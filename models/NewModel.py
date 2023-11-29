class StaticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StaticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Add more layers as needed
        )

    def forward(self, x):
        return self.network(x)

class CombineOutputs(nn.Module):
    def __init__(self, autoformer_dim, static_dim):
        super(CombineOutputs, self).__init__()
        self.dimension_match = nn.Linear(autoformer_dim + static_dim, autoformer_dim)

    def forward(self, autoformer_output, static_output):
        combined_output = torch.cat((autoformer_output, static_output), dim=1)
        return self.dimension_match(combined_output)


class ResidualConnection(nn.Module):
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


import torch
import torch.nn as nn

class UnifiedModel(nn.Module):
    def __init__(self, autoformer, static_input_dim, static_output_dim, autoformer_output_dim):
        super(UnifiedModel, self).__init__()
        # Initialize Autoformer model
        self.autoformer = autoformer

        # Initialize Static Network
        self.static_network = StaticNetwork(static_input_dim, static_output_dim)

        # Initialize Combiner
        self.combiner = CombineOutputs(autoformer_output_dim, static_output_dim)

    def forward(self, time_series_data, static_data):
        # Process time series data through Autoformer
        autoformer_output = self.autoformer(time_series_data)

        # Process static data through the static network
        static_output = self.static_network(static_data)

        # Combine and match dimensions
        combined_transformed = self.combiner(autoformer_output, static_output)

        # Apply residual connection
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
