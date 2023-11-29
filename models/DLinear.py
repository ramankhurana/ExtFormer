import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

## added by Raman
import numpy as np 
from layers.Embed import StaticEmbedding
## added by Raman ends here 

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)

        self.use_static = True

        ## Raman code starts
        # 7 for ETTh1
        # start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        # 200 for Divvy
        if (self.use_static):

            n_input = 200
            self.static_embeding = StaticEmbedding(n_input) 
            #self.static_embeding3 = StaticEmbedding(512,256) 
            #self.static_embeding4 = StaticEmbedding(256,128) 
            #self.static_embeding5 = StaticEmbedding(128,n_input) 
        ## Raman code ends

        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        seasonal_part = seasonal_output
        print ("shape os seasonal data is", seasonal_part.shape)
        # static 
        ## Code From Raman Starts here
        if (self.use_static):

            # static_raw = torch.tensor([1, 1, 2, 1, 2, 2, 1])   ## synthetic data for ETTh1 
            static_raw = torch.tensor(np.load('auxutils/divvy_static.npy').tolist() )  ## static real data for Divvy Bikes
        
            #static_raw = static_raw.repeat((32,72,1))   ## for input it should 96, for output it should be 144
            #static_raw = static_raw.repeat((32,144,1))  # for Auto and FED former  ## for input it should 96, for output it should be 144 
            static_raw = static_raw.repeat((32,96,1))   ## for DLinear for input it should 96, for output it should be 144 
            static_raw = static_raw.float()
            static_raw = static_raw.permute(0, 2, 1)

            static_out =  static_raw ## self.static_embeding(static_raw)
        ## the static_out is not yet used, it has to be embed to the temporal data and then it will be fed to the encoder 
        ### add the static and temporal data embeddings, in a later version it can be concatinated instead of adding
        ### it is added only tot he seasonal part of the decoder output instead of total output.
        ### Adding to total output can also be tested.

        if (self.use_static):
            print ("shape of static data: ", static_out.shape)
            seasonal_part  = static_out + seasonal_part
            seasonal_part = self.static_embeding(seasonal_part.permute(0, 2, 1)) ## the permutation is needed for DLinear only 

        seasonal_output = seasonal_part.permute(0,2,1)

        ## Raman code ends here 

        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
