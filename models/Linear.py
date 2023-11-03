import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        ###newly added 
        #self.n_locs = 2   ## unique values of the location
        #self.n_series = 7 ## size of embedding vector, this can be tuned 
        #self.sensor_to_location = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        ###newly added finished
        
        print ("individual flag: ",self.individual) 
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
                        
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            ### newly added
            #self.location_embedding = nn.Embedding(self.n_locs, self.n_series)
            ###newly added finished

    def forward(self,  x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        #print ("x_enc-----", x_enc )
        #print ("--------------------")
        #print ("x_mark_enc", x_mark_enc)
        
        x = x_enc
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            #print ("flow is in the forward",self.sensor_to_location) 
            ###newly added 
            #loc_emb = self.location_embedding(self.sensor_to_location)
            #print ("location embedding worked and it is: ", loc_emb)
            #x = x + loc_emb
            
            ###newly added finished
            
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            

        return x # [Batch, Output length, Channel]
