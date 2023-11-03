import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


## added by Raman
from layers.Embed import StaticEmbedding
## added by Raman ends here 

class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        if (False): print ("--------------------------------------------------")
        if (False): print ("Inside Autoformer init: self.task_name, self.seq_len, self.label_len, self.pred_len, self.output_attention, kernel_size", self.task_name, self.seq_len, self.label_len, self.pred_len, self.output_attention, kernel_size )

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        if (False): print ("input for embedding: ", configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        if (False): print ("output for embedding:", self.enc_embedding)


        ## Raman code starts
        self.static_embeding = StaticEmbedding() ## start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        self.static_embeding1 = StaticEmbedding(512,256) ## start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        self.static_embeding2 = StaticEmbedding(256,7) ## start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension 
        ## Raman code ends here 

        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if (False): print ("size of x_enc: ", x_enc.shape, x_mark_enc.shape, x_dec.shape, x_mark_dec.shape)


        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        
        print ("size of trend_init before concating the zeros/mean: ", seasonal_init.shape, trend_init.shape)
        print ("size of mean and zeros : ",mean.shape, zeros.shape)
        
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)


        if (False): print ("trend_init, seasonal_init: ", trend_init.shape, seasonal_init.shape)


        # enc
        if (False):print ("enc_out.shape before  embedding: ", x_enc.shape)

        ## Code From Raman Starts here
        static_raw = torch.tensor([1, 1, 2, 1, 2, 2, 1])
        static_raw = static_raw.repeat((32,144,1))   ## for input it should 96, for output it should be 144 
        static_raw = static_raw.float()
        if (False): print ("shape of static_raw", static_raw.shape)
        static_out =  static_raw ## self.static_embeding(static_raw)
        if (False): print ("shape of static_out", static_out.shape)
        if (False): print ("some instances of the static embedding: ", static_out[0][0])
        ## the static_out is not yet used, it has to be embed to the temporal data an then it will be fed to the encoder 
        ## Code from Raman ends here 



        

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if (False): print ("result of embedding: ", enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if (False): print ("result of encoder: ", enc_out.shape)


        
        
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        if (False):        print ("result of decoder embedding: ", dec_out.shape)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)


        ## Raman code starts
        ### add the static and temporal data embeddings, in a later version it can be concatinated instead of adding
        ### it is added only tot he seasonal part of the decoder output instead of total output.
        ### Adding to total output can also be tested.
        print ("result of seasonal_part----------: ", seasonal_part.shape)
        print ("result of static_part----------: ", static_out.shape)

        if (False):
            seasonal_part  = static_out + seasonal_part
            seasonal_part = self.static_embeding(seasonal_part)
            seasonal_part = self.static_embeding1(seasonal_part)
            seasonal_part = self.static_embeding2(seasonal_part)
        
        if (False): print ("result of encoder with static embedding added to it: ", enc_out.shape)
        ## Raman code ends here 

        
        if (False): print ("result of decoder: ", seasonal_part.shape, trend_part.shape)
        # final
        dec_out = trend_part + seasonal_part
        if (False): print ("final result: ", dec_out.shape)
        return dec_out

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
            print ("dec_out final result shape is: ", dec_out[:, -self.pred_len:, :].shape)
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