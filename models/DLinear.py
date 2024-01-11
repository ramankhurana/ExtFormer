import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

## added by Raman
import numpy as np 
from layers.Embed import StaticEmbedding, CombineOutputs
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
        self.static1    = configs.static=="static1"
        self.static2    = configs.static=="static2"
        self.static4    = configs.static=="static4"
        self.static6    = configs.static=="static6"
        self.static7    = configs.static=="static7"
        self.data       = configs.data 
        
        print ("self.static1, self.static2, self.static4, self.static6, self.static7", self.static1, self.static2, self.static4, self.static6, self.static7)

        ## Raman code starts
        # 7 for ETTh1
        # start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        # 200 for Divvy
        if (self.use_static):

            # static_raw = torch.tensor([1, 1, 2, 1, 2, 2, 1])   ## synthetic data for ETTh1 
            if self.data == "Divvy":
                self.static_raw = torch.tensor(np.load('auxutils/divvy_static.npy').tolist() )  ## static real data for Divvy Bikes
            if self.data == "M5":  
                    self.static_raw = torch.tensor(np.load('auxutils/M5_static.npy')[1].tolist() )  ## static real data for Divvy Bikes

            #static_raw = static_raw.repeat((32,72,1))   ## for input it should 96, for output it should be 144
            #static_raw = static_raw.repeat((32,144,1))  # for Auto and FED former  ## for input it should 96, for output it should be 144 
            
            #self.static_raw = self.static_raw.repeat((32,96,1))   ## for DLinear for input it should 96, for output it should be 144 
            self.static_raw = self.static_raw.repeat((32,self.pred_len,1))   ## for DLinear for input it should 96, for output it should be 144 
            self.static_raw = self.static_raw.float()
            self.static_raw = self.static_raw.permute(0, 2, 1)
            n_input = 1000
            self.DLinear_output_dim = n_input
            self.static_output_dim = n_input
            ## model 
            self.static_embeding = StaticEmbedding(n_input) 
            self.combiner = CombineOutputs(self.DLinear_output_dim, self.static_output_dim )

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
        print ("shape of seasonal data is", seasonal_part.shape)
        # static 
        ## Code From Raman Starts here
        if (self.static4 & self.use_static ):
            print ("This is now running for staitc4")
            static_out =   self.static_raw
            seasonal_part = self.combiner(seasonal_part.permute(0,2,1), static_out.permute(0,2,1) ) # static 4
            seasonal_part = self.static_embeding(seasonal_part)
            seasonal_output = seasonal_part.permute(0,2,1)

        if (self.static1 & self.use_static ):
            print ("This is now running for staitc1")
            static_out =   self.static_raw
            print ("shapes=---------",static_out.shape, seasonal_part.shape)
            seasonal_part  = static_out + seasonal_part
            seasonal_part = self.static_embeding(seasonal_part.permute(0,2,1))
            seasonal_output = seasonal_part.permute(0,2,1)

        if (self.static2 & self.use_static ):
            print ("This is now running for staitc2")
            static_out =   self.static_raw
            print ("shapes=---------",static_out.shape, seasonal_part.shape)
            static_out = self.static_embeding(static_out.permute(0,2,1)) 
            seasonal_part  = static_out.permute(0,2,1) + seasonal_part 
            seasonal_output = seasonal_part


        if self.static6 & self.use_static: 
            print ("This is now running for staitc6")

            static_out =   self.static_raw
            static_out = self.static_embeding(static_out.permute(0,2,1))
            seasonal_part = self.combiner(seasonal_part.permute(0,2,1), static_out) # static 6
            seasonal_output = seasonal_part.permute(0, 2, 1)


        if (self.static7 & self.use_static ):
            print ("This is now running for staitc7")
            static_out =   self.static_raw
            seasonal_part_orig = seasonal_part  
            seasonal_part = self.combiner(seasonal_part.permute(0,2,1), static_out.permute(0,2,1)) 
            seasonal_part = self.static_embeding(seasonal_part)
            seasonal_part = seasonal_part_orig + seasonal_part.permute(0,2,1)
            seasonal_output = seasonal_part

        ## Raman code ends here 

        print ("shapes: ", seasonal_output.shape, trend_output.shape)
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
