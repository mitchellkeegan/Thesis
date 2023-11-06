from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F


class featureExtractionFCN(nn.Module):
    def __init__(self,n_timesteps,n_out,batch_norm,dropout=False):
        super().__init__()

        channel_sizes = [32,64,32]
        pool_kernel = 8

        self.conv1 = nn.Conv1d(1,channel_sizes[0],8)
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

class featureExtractionANN(nn.Module):
    def __init__(self,n_features,n_out,batch_norm,dropout=False):
        super().__init__()

        self.fc1 = nn.Linear(n_features, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, n_out)

        self.bn1 = torch.nn.BatchNorm1d(8192) if batch_norm else lambda x: x
        self.bn2 = torch.nn.BatchNorm1d(4096) if batch_norm else lambda x: x

    def forward(self,x):
        x = self.b1(F.relu(self.fc1(x)))
        x = self.b2(F.relu(self.fc2(x)))
        return self.fc3(x)

class outputANN(nn.Module):
    def __init__(self, n_features, n_out, layer_widths, output_shape, batch_norm, dropout=False):
        super().__init__()

        self.output_shape = output_shape

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
        self.model_details = {'Name': 'FCN -> Shared Task Specific Networks',
                              'Model Iteration': 1}
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
        r_small = [0 + fixed_params['r_small'].get(a,0) for a in A]
        r_large = [0 + fixed_params['r_large'].get(a, 0) for a in A]
        value = [0 + fixed_params['value'].get(a, 0) for a in A]
        penalty = [0 + fixed_params['penalty'].get(a, 0) for a in A]
        dur = fixed_params['dur']
        p = fixed_params['p']
        n_small = fixed_params['n_small']
        n_large = fixed_params['n_large']

        self.battery_features = torch.tensor([fixed_params['m'],
                                              fixed_params['cap']]).T
        self.class_features = torch.tensor([r_small,
                                            r_large,
                                            dur,
                                            p]).T

        self.oneoff_features = torch.tensor([value,
                                             penalty]).T

        self.battery_features = self.normalise_fixed_features(self.battery_features)
        self.class_features = self.normalise_fixed_features(self.class_features)
        self.oneoff_features = self.normalise_fixed_features(self.oneoff_features)

        self.feature_extractor = featureExtractionFCN(n_timesteps,4096, use_batch_norm)

        self.battery_network = outputANN(self.feature_extractor.n_out + self.battery_features.shape[1],
                                         3*n_timesteps,
                                         [1024,512],
                                         (-1,3,2880),
                                         use_batch_norm)

        self.recurring_network = outputANN(self.feature_extractor.n_out + self.class_features.shape[1],
                                           T_r,
                                           [1024,512],
                                           (-1,T_r),
                                           use_batch_norm)

        self.oneoff_network = outputANN(self.feature_extractor.n_out + self.class_features.shape[1] + self.oneoff_features.shape[1],
                                        T_o,
                                        [1024, 512],
                                        (-1,T_o),
                                        use_batch_norm)

    def forward(self,x):
        x = self.feature_extractor(x)

        xb = torch.stack(tuple(self.battery_network(self.augment_battery(x,b))
                               for b in self.fixed_params['B']),
                         dim=3)
        xr = torch.stack(tuple(self.recurring_network(self.augment_recurring(x,a))
                               for a in self.fixed_params['A_r']),
                         dim=2)
        xo = torch.stack(tuple(self.oneoff_network(self.augment_oneoff(x, a))
                               for a in self.fixed_params['A_o']),
                         dim=2)

        return xr, xo, xb

    def augment_battery(self,x,b):
        batch_size, n_features = x.shape
        new_columns = torch.ones((batch_size,1)) * self.battery_features[b,:]
        return torch.cat((x,new_columns),dim=1)

    def augment_recurring(self,x,a):
        batch_size, n_features = x.shape
        new_columns = torch.ones((batch_size, 1)) * self.class_features[a, :]
        return torch.cat((x, new_columns), dim=1)

    def augment_oneoff(self,x,a):
        batch_size, n_features = x.shape
        new_columns1 = torch.ones((batch_size, 1)) * self.class_features[a, :]
        new_columns2 = torch.ones((batch_size, 1)) * self.oneoff_features[a, :]
        return torch.cat((x, new_columns1, new_columns2), dim=1)

    def normalise_fixed_features(self,X):
        X_max, X_min = X.max(dim=0)[0], X.min(dim=0)[0]
        return (X-X_min)/(X_max-X_min)