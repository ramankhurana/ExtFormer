
import torch
import torch.nn as nn
import numpy as np 
from layers.Embed import StaticEmbedding, CombineOutputs
from models import Autoformer

class Model(nn.Module):
    ''' This is an extension of the Autoformer model to deal with the static datasets'''
    def __init__(self, configs):
        super(Model, self).__init__()
        # Initialize Autoformer model
        self.static_input = configs.c_out
        self.static_output = configs.c_out
        self.autoformer_output = configs.c_out

        self.autoformer = Autoformer.Model(configs)

        # Initialize Static Network
        # input and output will be same dimension
        self.static_network = StaticEmbedding(self.static_input)#,self.static_output)

        # Initialize Combiner
        self.combiner = CombineOutputs(self.autoformer_output, self.static_output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Process time series data through Autoformer
        autoformer_output = self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec)

        ######
        static_raw = torch.tensor(np.load('auxutils/divvy_static.npy').tolist() )  ## static real data for Divvy Bikes
        #static_raw = static_raw.repeat((32,72,1))   ## for input it should 96, for output it should be 144
        static_raw = static_raw.repeat((32,96,1))   ## for input it should 96, for output it should be 144 
        static_raw = static_raw.float()
        static_out =  static_raw ## self.static_embeding(static_raw)
        static_data = static_out
        ####

        # Process static data through the static network
        static_output = self.static_network(static_data)

        # Combine and match dimensions
        combined_transformed = self.combiner(autoformer_output, static_output)

        # Apply residual connection
        #return autoformer_output + combined_transformed
        return combined_transformed 


'''
class ResidualConnection(nn.Module)
    def __init__(self, autoformer, static_network, combiner):
        super(ResidualConnection, self).__init__()
        self.autoformer = autoformer
        self.static_network = static_network
        self.combiner = combiner

    def forward(self, time_series_data, static_data):
        autoformer_output = self.autoformer(time_series_data)
        static_output = self.static_network(static_data)
        combined_transformed = self.combiner(autoformer_output, static_output)
        return autoformer_output + combined_transformed



# Define the Static Network class
class StaticNetwork(nn.Module):
    # ... (as previously defined)

# Define the CombineOutputs class
class CombineOutputs(nn.Module):
    # ... (as previously defined)



## Using the unified model 

# Initialize your Autoformer model
autoformer = YourAutoformerModel(...)  # Replace with your specific Autoformer model

# Create the unified model
unified_model = UnifiedModel(autoformer, static_input_dim, static_output_dim, autoformer_output_dim)

# Now you can use unified_model in your training and inference pipelines

'''