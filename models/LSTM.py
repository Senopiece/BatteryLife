import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Model(nn.Module):
    """
    Use BiLSTM as the baseline model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.charge_discharge_length = configs.charge_discharge_length
        self.early_cycle_threshold = configs.early_cycle_threshold
        self.drop_rate = configs.dropout
        # Embedding
        self.embedding = nn.Linear(3, configs.d_model)
        self.LSTM = nn.LSTM(configs.d_model, configs.d_ff, configs.lstm_layers, batch_first=True, dropout=self.drop_rate, bidirectional=False)
        self.projection = nn.Linear(configs.d_ff, configs.output_num)

    def classification(self, x_enc, rnn_mask, return_embedding=False):
        # x_enc: [B, L*fixed_len; num_var]
        lengths = torch.sum(rnn_mask, dim=1).cpu() # [N]
        x_enc = self.embedding(x_enc)

        x_enc = pack_padded_sequence(x_enc, lengths=lengths, batch_first=True, enforce_sorted=False)
        x_enc, (_,_) = self.LSTM(x_enc)
        x_enc, lens_unpacked = pad_packed_sequence(x_enc, True)
        idx = (torch.as_tensor(lengths, device=x_enc.device, dtype=torch.long) - 1).view(-1, 1).expand(
            len(lengths), x_enc.size(2))
        idx = idx.unsqueeze(1)
        x_enc = x_enc.gather(1, idx).squeeze(1) # [B, 2*d_ff]
        output = self.projection(x_enc)
        if return_embedding:
            return output, x_enc
        return output

    def forward(self,  cycle_curve_data, curve_attn_mask, return_embedding=False):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask: [B, L]
        '''
        fixed_len = cycle_curve_data.shape[-1]
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        B, num_vars = cycle_curve_data.shape[0], cycle_curve_data.shape[2]
        cycle_curve_data = cycle_curve_data.transpose(2, 3)
        cycle_curve_data = cycle_curve_data.reshape(B, -1, num_vars) # [B, L*fixed_len, num_vars]
        rnn_mask = torch.repeat_interleave(curve_attn_mask, dim=1, repeats=fixed_len)
        if return_embedding:
            dec_out, embedding = self.classification(cycle_curve_data, rnn_mask, return_embedding=return_embedding)
            return dec_out, embedding
        else:
            dec_out = self.classification(cycle_curve_data, rnn_mask, return_embedding=return_embedding)
            return dec_out  # [B, N]
