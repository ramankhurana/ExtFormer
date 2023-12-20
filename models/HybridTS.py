import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import BandPassFilter, StaticEmbedding, CombineOutputs
import numpy as np


class Model(nn.Module):
    """
    HybridTS:
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_static = False 
        
        '''
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)
        '''
        self.bandpassfilter =  BandPassFilter(10,1000)
        self.linear = nn.Sequential(nn.Linear(self.seq_len,self.pred_len),  # First layer
                        nn.ReLU(),                    # ReLU activation after the first layer
                        #nn.Linear(128, 256),  # Second layer
                        #nn.ReLU(),    # ReLU activation after the second layer
                        #nn.Linear(256, 128),  # third layer
                        #nn.ReLU(),                           # ReLU activation after the second layer
                        #nn.Linear(128, self.pred_len)  # forth layer
                        )
        self.linear1 = nn.Linear(self.seq_len,self.pred_len)
        self.linear2 = nn.Linear(self.seq_len,self.pred_len)
        self.linear3 = nn.Linear(self.seq_len, self.pred_len)
        self.relu = nn.ReLU()  # ReLU activation function


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x = self.bandpassfilter(x_enc)
        x_orig = x
        #print ("after bandpass x_enc.shape: ", x[0][0])
        x = x.real.float()
        #print ("x.shape-----", x.shape, x.dtype)
        x = self.linear(x.permute(0,2,1))

        x1 = x_orig.abs().float()
        x1 = self.linear(x1.permute(0,2,1))

        x2 = x_orig.angle().float()
        x2 = self.linear(x2.permute(0,2,1))


        x3 = x + x1 + x2 
        x3 = self.linear(x3)

        return x3.permute(0,2,1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            #print ("dec_out final result shape is: ", dec_out[:, -self.pred_len:, :].shape)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
