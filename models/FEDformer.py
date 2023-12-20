import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
## added by Raman
import numpy as np 
from layers.Embed import StaticEmbedding
## added by Raman ends here 


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        self.use_static = False
        self.static1    = configs.static=="static1"
        self.static2    = configs.static=="static2"
        self.static4    = configs.static=="static4"
        self.static6    = configs.static=="static6"
        self.static7    = configs.static=="static7"


        # Decomp
        self.decomp = series_decomp(configs.moving_avg)

        print ("self.static1, self.static2, self.static4, self.static6, self.static7", self.static1, self.static2, self.static4, self.static6, self.static7)

        ## Raman code starts
        # 7 for ETTh1
        # start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        # 200 for Divvy
        if (self.use_static):

            # static_raw = torch.tensor([1, 1, 2, 1, 2, 2, 1])   ## synthetic data for ETTh1 
            self.static_raw = torch.tensor(np.load('auxutils/divvy_static.npy').tolist() )  ## static real data for Divvy Bikes
            #static_raw = static_raw.repeat((32,72,1))   ## for input it should 96, for output it should be 144
            #static_raw = static_raw.repeat((32,144,1))  # for Auto and FED former  ## for input it should 96, for output it should be 144 
            self.static_raw = self.static_raw.repeat((32,96,1))   ## for DLinear for input it should 96, for output it should be 144 
            self.static_raw = self.static_raw.float()
            self.static_raw = self.static_raw.permute(0, 2, 1)
            n_input = 200
            self.DLinear_output_dim = n_input
            self.static_output_dim = n_input
            ## model 
            self.static_embeding = StaticEmbedding(n_input) 
            self.combiner = CombineOutputs(self.DLinear_output_dim, self.static_output_dim )
        ## Raman code ends


        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
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
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)


        print ("shape of seasonal data is", seasonal_part.shape)
        # static 
        ## Code From Raman Starts here
        if (self.static4 & self.use_static ):
            print ("This is now running for staitc4")
            static_out =   self.static_raw
            seasonal_part = self.combiner(seasonal_part, static_out )
            seasonal_part = self.static_embeding(seasonal_part)

        if (self.static1 & self.use_static ):
            print ("This is now running for staitc1")
            static_out =   self.static_raw
            print ("shapes=---------",static_out.shape, seasonal_part.shape)
            seasonal_part  = static_out + seasonal_part
            seasonal_part = self.static_embeding(seasonal_part)

        if (self.static2 & self.use_static ):
            print ("This is now running for staitc2")
            static_out =   self.static_raw
            print ("shapes=---------",static_out.shape, seasonal_part.shape)
            static_out = self.static_embeding(static_out) 
            seasonal_part  = static_out + seasonal_part 


        if self.static6 & self.use_static: 
            print ("This is now running for staitc6")

            static_out =   self.static_raw
            static_out = self.static_embeding(static_out)
            seasonal_part = self.combiner(seasonal_part, static_out) 

        if (self.static7 & self.use_static ):
            print ("This is now running for staitc7")
            static_out =   self.static_raw
            seasonal_part_orig = seasonal_part  
            seasonal_part = self.combiner(seasonal_part, static_out) 
            seasonal_part = self.static_embeding(seasonal_part)
            seasonal_part = seasonal_part_orig + seasonal_part

        ## Raman code ends here 

        # static 
        ## Code From Raman Starts here
        if (self.use_static):

            # static_raw = torch.tensor([1, 1, 2, 1, 2, 2, 1])   ## synthetic data for ETTh1 
            static_raw = torch.tensor(np.load('auxutils/divvy_static.npy').tolist() )  ## static real data for Divvy Bikes
        
            #static_raw = static_raw.repeat((32,72,1))   ## for input it should 96, for output it should be 144
            static_raw = static_raw.repeat((32,144,1))   ## for input it should 96, for output it should be 144 
            static_raw = static_raw.float()
            static_out =  static_raw ## self.static_embeding(static_raw)
        
            ## the static_out is not yet used, it has to be embed to the temporal data and then it will be fed to the encoder 
            ### add the static and temporal data embeddings, in a later version it can be concatinated instead of adding
            ### it is added only tot he seasonal part of the decoder output instead of total output.
            ### Adding to total output can also be tested.
            if False: 
                seasonal_part  = static_out + seasonal_part
                seasonal_part = self.static_embeding(seasonal_part)
            
            static_out = self.static_embeding(static_out)
            seasonal_part  = static_out + seasonal_part

            #seasonal_part = self.static_embeding3(seasonal_part)
            #seasonal_part = self.static_embeding4(seasonal_part)
            #seasonal_part = self.static_embeding5(seasonal_part)
            
        ## Raman code ends here 


        # final
        dec_out = trend_part + seasonal_part
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
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
