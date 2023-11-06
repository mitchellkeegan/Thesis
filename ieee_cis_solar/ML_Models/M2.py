from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F

class featureExtractionFCN(nn.Module):
    def __init__(self,n_timesteps,batch_norm,dropout=False):
        super().__init__()

        channel_sizes = [64,128,64]
        pool_kernel = 32

        self.conv1 = nn.Conv1d(3,channel_sizes[0],8)
        self.conv2 = nn.Conv1d(channel_sizes[0],channel_sizes[1],5)
        self.conv3 = nn.Conv1d(channel_sizes[1],channel_sizes[2],3)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)

        L1 = n_timesteps - 7
        L2 = L1 - 4
        L3 = L2 - 2
        L4 = floor((L3-(pool_kernel-1)-1)/pool_kernel + 1)

        self.bn1 = torch.nn.BatchNorm1d(channel_sizes[0])
        self.bn2 = torch.nn.BatchNorm1d(channel_sizes[1])
        self.bn3 = torch.nn.BatchNorm1d(channel_sizes[2])

        self.n_out = channel_sizes[2] * L4

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        # Add some pooling?
        return x.view(-1,self.n_out)

class outputANN(nn.Module):
    def __init__(self, n_features, n_out, layer_widths=[1024,512], output_shape=None, batch_norm=True, dropout=False):
        super().__init__()
        if output_shape:
            self.output_shape = output_shape
        else:
            raise Exception('Output ANN must be provided with an output shape!!\n')

        self.fc1 = nn.Linear(n_features, layer_widths[0])
        self.fc2 = nn.Linear(layer_widths[0], layer_widths[1])
        self.fc3 = nn.Linear(layer_widths[1], n_out)

        self.bn1 = torch.nn.BatchNorm1d(layer_widths[0]) if batch_norm else lambda x: x
        self.bn2 = torch.nn.BatchNorm1d(layer_widths[1]) if batch_norm else lambda x: x

    def forward(self, x):
        # x = x.view(-1,3,2880)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        return torch.reshape(self.fc3(x),self.output_shape)

class classScheduleNN(nn.Module):
    def __init__(self,model_params,fixed_params):
        super().__init__()
        self.model_details = {'Name': 'FCN -> Individual Task Specific Networks',
                              'Model Iteration': 2}
        self.fixed_params = fixed_params

        use_batch_norm = model_params.get('Batch Norm',True)
        use_dropout = model_params.get('Dropout',False)

        n_timesteps = model_params['n_features']
        n_batteries = model_params['n_batteries']
        T_r = model_params['T_r']
        T_o = model_params['T_o']
        self.n_recurring = model_params['n_recurring']
        self.n_oneoff = model_params['n_oneoff']

        A = fixed_params['A']
        r_small = [fixed_params['r_small'][a] for a in A]
        r_large = [fixed_params['r_large'][a] for a in A]
        value = [0 + fixed_params['value'].get(a, 0) for a in A]
        penalty = [0 + fixed_params['penalty'].get(a, 0) for a in A]
        dur = fixed_params['dur']
        p = fixed_params['p']
        eff = fixed_params['eff']
        m = fixed_params['m']
        base_load = fixed_params['base_load']
        solar_supply = fixed_params['solar_supply']

        fixed_times_series = [base_load, solar_supply]
        fixed_features = [r_small, r_large, value, dur, p, m, eff]

        self.fixed_times_series = torch.cat((self.normalise_fixed_features(torch.tensor(base_load)).reshape(1,1,-1),
                                             self.normalise_fixed_features(torch.tensor(solar_supply)).reshape(1,1,-1)), dim=1)

        self.fixed_features = torch.cat(tuple(self.normalise_fixed_features(torch.tensor(feature)).reshape(1,-1) for feature in fixed_features), dim=1)

        self.feature_extractor = featureExtractionFCN(n_timesteps, use_batch_norm)

        latent_size = 1024

        self.feature_mixer = outputANN(self.feature_extractor.n_out + self.fixed_features.shape[1],
                                       latent_size,
                                       [2048,latent_size],
                                       output_shape=(-1,latent_size))

        layer_widths = [512,256]


        self.battery_networks = nn.ModuleList([outputANN(latent_size,
                                           3*n_timesteps,
                                           layer_widths=layer_widths,
                                           output_shape=(-1,3,2880)) for b in fixed_params['B']])

        self.recurring_networks = nn.ModuleList([outputANN(latent_size,
                                             T_r,
                                             layer_widths=layer_widths,
                                             output_shape=(-1,T_r)) for a in fixed_params['A_r']])

        #TODO Add one to the output size?

        self.oneoff_networks = nn.ModuleList([outputANN(latent_size,
                                          T_o,
                                          layer_widths=layer_widths,
                                          output_shape=(-1, T_o)) for a in fixed_params['A_o']])

        feature_extractor_params = sum(p.numel() for p in self.feature_extractor.parameters())
        feature_mixer_params = sum(p.numel() for p in self.feature_mixer.parameters())
        battery_network_params = sum(sum(p.numel() for p in model.parameters()) for model in self.battery_networks)
        recurring_network_params = sum(sum(p.numel() for p in model.parameters()) for model in self.recurring_networks)
        oneoff_network_params = sum(sum(p.numel() for p in model.parameters()) for model in self.oneoff_networks)

        # Can use this to sanity check that the optimiser is actually picking up all the parameters in the model (E.g. a submodel inside a list will be hidden)
        self.total_paramaters = feature_extractor_params + feature_mixer_params + battery_network_params + recurring_network_params + oneoff_network_params

    def forward(self,x):

        batch_size = x.shape[0]
        fixed_times_series = self.fixed_times_series.expand(batch_size,-1,-1)
        x = torch.cat((fixed_times_series,x),dim=1)

        x = self.feature_extractor(x)

        fixed_features = self.fixed_features.expand(batch_size,-1)
        x = torch.cat((fixed_features,x),dim=1)

        x = self.feature_mixer(x)

        xb = torch.stack(tuple(self.battery_networks[b](x) for b in self.fixed_params['B']), dim=3)
        xr = torch.stack(tuple(self.recurring_networks[a](x) for a in self.fixed_params['A_r']), dim=2)
        xo = torch.stack(tuple(self.oneoff_networks[a](x) for a in range(len(self.fixed_params['A_o']))), dim=2)

        return xr, xo, xb

    def normalise_fixed_features(self,X):
        X_max, X_min = X.max(dim=0)[0], X.min(dim=0)[0]
        return (X-X_min)/(X_max-X_min)