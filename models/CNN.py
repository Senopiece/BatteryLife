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
        
        # stack the CNN
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=configs.d_model, kernel_size=5)
        self.cnn2 = nn.Conv2d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=5)
        self.flatten = nn.Flatten(start_dim=2)
        self.flatten_projection = nn.Linear(1311, 1)
        self.head = nn.Linear(configs.d_model, 1)


    def classification(self, x_enc, curve_attn_mask, return_embedding):
        '''
        x_enc: [B, L, num_variables, fixed_length_of_curve]
        '''
        # Embedding
        # Intra-cycle modelling
        B, L, num_var = x_enc.shape[0], x_enc.shape[1], x_enc.shape[2]
        x_enc = x_enc.reshape(B, num_var, L, -1)
        output = self.cnn1(x_enc)
        output = F.relu(output)
        output = F.avg_pool2d(output, kernel_size=5, stride=2)
        output = self.cnn2(output)
        output = F.relu(output)
        output = F.avg_pool2d(output, kernel_size=5, stride=2)
        output = self.flatten(output)

        embedding = self.flatten_projection(output).squeeze(-1)
        
        output = self.head(embedding)
        if return_embedding:
            return output, embedding
        return output

    def forward(self,  cycle_curve_data, curve_attn_mask, return_embedding=False):
        '''
        params:
            cycle_curve_data: [B, L, num_variables, fixed_length_of_curve]
            curve_attn_mask: [B, L]
        '''
        # tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
        # cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
        if return_embedding:
            dec_out, embedding = self.classification(cycle_curve_data, curve_attn_mask, return_embedding)
            return dec_out, embedding  # [B, N]
        else:
            dec_out = self.classification(cycle_curve_data, curve_attn_mask, return_embedding)
            return dec_out  # [B, N]
