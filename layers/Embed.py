import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        if (False): print ("shape of x in TokenEmbedding before: ", x.shape)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        if (False): print ("shape of x in TokenEmbedding after: ", x.shape)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        if (False): print ("shape of temporal embedding: ", (hour_x + weekday_x + day_x + month_x + minute_x).shape )
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        if (False): print ("shape of x in TimeFeatureEmbedding: -------------", x.shape)
        x = self.embed(x)
        if (False): print ("shape of data after TimeFeatureEmbedding: ", x.shape)
        return (x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        if (False): print ("embed_type: ", embed_type)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        if (False):print ("shape of x in DataEmbedding_wo_pos after: ", x.shape)
        
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class StaticEmbedding(nn.Module):
    def __init__(self, d_inp=7, d_output=512, dropout=0.2):
        super(StaticEmbedding, self).__init__()
        ## Linear Embedding
        self.static_embed = nn.Linear(d_inp, d_output, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()  # ReLU activation function
        
        self.static_embedL1 = nn.Linear(d_output,256) ## start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        self.static_embedL2 = nn.Linear(256,128) ## start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension
        self.static_embedL3 = nn.Linear(128,d_inp) ## start with default parameters 7, 512 in the begining, this is matched with the temporal embed data dimension 

    def forward(self, x):
        #print ("size of x in StaticEmbedding", x.shape)
        x = self.static_embed(x)
        x = self.relu(x)  # Apply ReLU after linear transformation
        x = self.dropout(x)  # Apply dropout after activation
        #print ("in forward pass of StaticEmbedding")
        x = self.dropout(self.relu (self.static_embedL1(x))  )
        x = self.dropout(self.relu (self.static_embedL2(x))  )
        x = self.dropout(self.relu (self.static_embedL3(x))  )

        return (x)



class CombineOutputs(nn.Module):
    ''' mode=1; addition 
        mode=2; concatination 
    '''
    def __init__(self, autoformer_dim, static_dim,mode=2):
        super(CombineOutputs, self).__init__()
        self.mode=mode
        self.dimension_match = nn.Linear( autoformer_dim + static_dim, autoformer_dim ) 

    def forward(self, autoformer_output, static_output):
        #print ("shape of autoformer_output:", autoformer_output.shape)
        #print ("shape of static_output:", static_output.shape)
  
        if self.mode==1:
            combined_output = autoformer_output + static_output
        if self.mode==2:
            #print ("this is in the mode 2")
            
            print ("shape of combined_output:", static_output.shape, autoformer_output.shape)

            combined_output = torch.cat((autoformer_output, static_output), dim=-1)
            

            self.dimension_match(combined_output)
        return self.dimension_match(combined_output)



    

## Bandpass filter for HybridTS
class BandPassFilter(nn.Module):
    def __init__(self, low_freq=10, high_freq=1000 ):
        super(BandPassFilter, self).__init__()
        # Initialize parameters for band-pass filter here, if needed
        self.low_freq = low_freq
        self.high_freq = high_freq

    def fourier_transform(self, x):
        # Apply Fourier transform
        # Assuming x is a numpy array. 
        # If x is a tensor, convert it 
        # to numpy array before applying fft
        return np.fft.fft(x)

    def apply_filter(self, freq_components):
        # Implement band-pass filter logic here
        # This is a placeholder implementation. 
        # You should replace it with your actual filter logic.
        # Example: Zero out components not within the desired 
        # frequency range
        filtered_freq_components = np.copy(freq_components)
        # Define your frequency range
        #print ("frequency range: ", freq_components.shape[1])
        #freq_range = np.arange(freq_components.shape[1])
        #mask = (freq_range < self.low_freq) | (freq_range > self.high_freq)
        #filtered_freq_components[:, mask] = 0
        return filtered_freq_components

    def forward(self, x):
        # Check if input is a tensor, convert to numpy array for fft
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        freq_components = self.fourier_transform(x)
        filtered_output = self.apply_filter(freq_components)
        #print ("filtered_output in Bandpass filter *******************: ", filtered_output)
        # Convert back to tensor for further processing in PyTorch
        return torch.from_numpy(filtered_output)

# check the freq vs amplitude for the time series, 
# see which freq have high amplitudes. 
