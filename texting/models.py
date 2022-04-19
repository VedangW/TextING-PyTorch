from turtle import forward
import layers
from texting.layers import Dense, ReadoutLayer

import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.name = args.name

    
class MLP(Model):
    
    def __init__(self, configs, input_dim, hidden_dim, output_dim):
        self.configs = configs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_features_nonzero = self.configs['num_features_nonzero']
        self.dropout = self.configs['dropout'] # TODO: Seriously, check how these are passed.
        self.softmax = nn.Softmax()

        self.build()

    def build(self):
        self.fc1 = Dense(input_dim=self.input_dim, 
                         hidden_dim=self.hidden_dim, 
                         num_features_nonzero=self.num_features_nonzero, 
                         activation=self.dropout, 
                         sparse_inputs=False)

        self.readout = ReadoutLayer(input_dim=self.hidden_dim, 
                                    output_dim=self.output_dim, 
                                    configs=self.configs, 
                                    dropout=True, 
                                    act=lambda x: x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.readout(x)

        return self.softmax(x)

    


