"""Adaptive graph filtering layer"""

from typing import Optional
import torch.nn as nn, torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange


class AGFAttention(nn.Module):
    """
    Adaptive Graph Filtering Layer with OV Circuit and Multi-Head support.

    Flow:
    1. Estimator computes Adjacency A from Q, K.
    2. Input projected to Values V.
    3. Filter computes X' = Poly(A, V).
    4. Heads concatenated and projected via Output Linear.
    """

    _act_map = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": F.relu,
        "gelu": F.gelu,
        "softmax": lambda x: F.softmax(x, dim=0),
        "identity": lambda x: x,
        "none": lambda x: x,
    }

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        order: int = 3,
        top_k: Optional[int] = None,
        basis: str = "monomial",
        alphas_act: str = "gelu",
    ):
        super().__init__()
        self.num_heads, self.dim_head, self.order, self.k = num_heads, dim_head, order, top_k
        self.alphas_act_name = alphas_act.lower()
        self.basis, self.scale = basis.lower(), dim_head**-0.5

        inner = num_heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=True)
        self.to_out = nn.Linear(inner, dim, bias=True)

        # Coefficients
        self.alphas_raw = nn.Parameter(torch.empty(order, num_heads))
        self.act = self._act_map.get(alphas_act.lower(), lambda x: x)

        # State keys for regularization (ephemeral dict is risky for DDP/JIT, used properties)
        self.register_buffer("last_adj", None, persistent=False)
        self.register_buffer("last_x", None, persistent=False)
        self.reset_parameters()
    
    @property
    def alphas(self):
        return self.act(self.alphas_raw)

    @torch.no_grad()
    def reset_parameters(self):
        """
        Initializes alpha coefficients with a decay factor to prevent 
        vanishing gradients and focus initially on low-order terms.
        """
        indices = torch.arange(self.order, device=self.alphas_raw.device).float()
        
        if self.alphas_act_name == "softmax":
            init_values = torch.exp(-indices) # [1.0, 0.36, 0.13, ...]
        elif self.alphas_act_name in ["sigmoid", "tanh"]:
            init_values = 0.5 * torch.exp(-indices) 
        else:
            init_values = torch.ones(self.order) / self.order

        init_values = init_values.unsqueeze(-1).repeat(1, self.num_heads)
        self.alphas_raw.copy_(init_values)
        
        return self

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.num_heads
        self.last_x = x
        # Unpack: (B, N, 3*H*D) -> 3 * (B, H, N, D)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        attn_scores: Tensor = (q @ k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask_bc = mask.view(b, 1, 1, n)
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask_bc, float("-inf"))
            else:
                attn_scores = attn_scores.masked_fill(mask_bc == 0, float("-inf"))

        if self.k and self.k < n:
            top_val = attn_scores.topk(self.k, dim=-1)[0][..., -1:]
            attn_scores = attn_scores.masked_fill(attn_scores < top_val, float("-inf"))

        attn = attn_scores.softmax(dim=-1)
        self.last_adj = attn

        with torch.autocast(device_type=x.device.type, enabled=False):
            attn, v = attn.float(), v.float()
            alphas = self.act(self.alphas_raw).view(-1, 1, h, 1, 1).float()

            res = 0
            v_prev, v_curr = None, v

            for i in range(self.order):
                if self.basis == "monomial":
                    v_curr = attn @ v_curr
                else:
                    # Chebyshev Recurrence: T_k = 2(2A - I)T_{k-1} - T_{k-2}
                    # l_v represents (2A - I)T_{k-1}
                    l_v = 2 * (attn @ v_curr) - v_curr
                    v_next = (2 * l_v - v_prev) if v_prev is not None else l_v
                    v_prev, v_curr = v_curr, v_next

                res = res + alphas[i] * v_curr

        out = res.to(x.dtype).transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

    def get_reg_loss(self):
        """Computes Spectral Smoothness Loss"""
        if self.last_adj is None:
            return torch.tensor(0.0, device=self.to_out.weight.device)

        # Dirichlet Energy on Normalized Features
        adj = self.last_adj.float()
        x_n = F.normalize(self.last_x.detach().float(), dim=-1)

        # "bnd,bhnm,bmd->" calculates Tr(X^T L X) efficiently
        smooth = torch.einsum("bnd,bhnm,bmd->", x_n, adj, x_n)

        bs, heads, n, _ = adj.shape
        return -smooth / (bs * heads * n)


class AGFAttentionLayer(nn.Module):
    """Adapter that matches the project's attention layer interface."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        order: int = 3,
        top_k: Optional[int] = None,
        basis: str = "monomial",
        alphas_act: str = "gelu",
        output_attention: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.output_attention = output_attention
        self.inner_attention = AGFAttention(
            dim=d_model,
            num_heads=n_heads,
            dim_head=d_model // n_heads,
            order=order,
            top_k=top_k,
            basis=basis,
            alphas_act=alphas_act,
        )

    def _to_padding_mask(self, attn_mask: Optional[Tensor]):
        if attn_mask is None:
            return None
        if attn_mask.dim() == 2:
            return (~attn_mask) if attn_mask.dtype == torch.bool else (attn_mask != 0)
        if attn_mask.dim() == 3:
            # [B, L, L] with True meaning "masked" in this codebase.
            return ~attn_mask[:, 0, :]
        if attn_mask.dim() == 4:
            # [B, 1, L, L] with True meaning "masked" in this codebase.
            return ~attn_mask[:, 0, 0, :]
        raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        mask = self._to_padding_mask(attn_mask)
        out = self.inner_attention(queries, mask=mask)
        attn = self.inner_attention.last_adj if self.output_attention else None
        return out, attn
