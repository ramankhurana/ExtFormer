import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)



# Define a class for the Static Model
class StaticModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StaticModel, self).__init__()
        self.linear = LinearModel(input_size, hidden_size)
        self.output_layer = LinearModel(hidden_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.output_layer(x)
        return x


class ExogenousModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_lags):
        super(ExogenousModel, self).__init__()
        self.num_lags = num_lags
        self.lagged_linear = nn.Linear(input_size * num_lags, hidden_size)
        self.output_layer = LinearModel(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input to include lagged variables
        x = x.view(x.size(0), -1)
        x = self.lagged_linear(x)
        x = self.output_layer(x)
        return x



class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        print ("self.seq_len, self.pred_len: ",self.seq_len, self.pred_len )
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = False
        self.dropout_prob = 0.5
        
        print ("individual flag: ",self.individual) 
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
                        
            self.Linear = nn.Linear(self.seq_len, self.pred_len)


    def forward(self,  x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x_enc
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            

        return x # [Batch, Output length, Channel]




