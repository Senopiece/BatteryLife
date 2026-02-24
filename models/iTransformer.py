import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        if self.task_name == 'classification':
            self.enc_embedding = DataEmbedding_inverted(configs.early_cycle_threshold * configs.charge_discharge_length, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, 1)

    def classification(self, x_enc, x_mark_enc, return_embedding):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        embedding = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(embedding)  # (batch_size, num_classes)
        if return_embedding:
            output, embedding
        return output

    def forward(self, cycle_curve_data, curve_attn_mask, x_mark_enc=None, return_embedding=False):
        '''
        cycle_curve_data: [B, early_cycle, num_var, fixed_len]
        '''


        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        B, num_vars = cycle_curve_data.shape[0], cycle_curve_data.shape[2]
        cycle_curve_data = cycle_curve_data.transpose(2, 3)
        cycle_curve_data = cycle_curve_data.reshape(B, -1, num_vars) # [B, L*fixed_len, num_vars]
        
        x_enc = cycle_curve_data
        if return_embedding:
            dec_out, embedding = self.classification(x_enc, None, return_embedding=return_embedding)
            return dec_out, embedding  # [B, N]
        else:
            dec_out = self.classification(x_enc, None, return_embedding=return_embedding)
            return dec_out  # [B, N]
