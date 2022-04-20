import torch.nn as nn

from layers import Dense, GraphLayer, ReadoutLayer


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
    
class MLP(Model):
    def __init__(self, configs, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.configs = configs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_features_nonzero = self.configs['num_features_nonzero']
        self.dropout = self.configs['dropout'] # TODO: Seriously, check how these are passed.
        self.softmax = nn.Softmax()

        self._build()

    def _build(self):
        self.fc1 = Dense(input_dim=self.input_dim, 
                         output_dim=self.hidden_dim, 
                         num_features_nonzero=self.num_features_nonzero, 
                         activation=self.dropout, 
                         sparse_inputs=False)

        self.readout = ReadoutLayer(input_dim=self.hidden_dim, 
                                    output_dim=self.output_dim, 
                                    args=self.configs, 
                                    dropout=True, 
                                    act=lambda x: x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.readout(x)
        x = self.softmax(x)

        return x


class GatedGNN(Model):
    def __init__(self, args, output_dim):
        super().__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden
        self.output_dim = output_dim
        self.softmax = nn.Softmax()
        self.args = args
        self._build()

    def _build(self):
        self.gl1 = GraphLayer(input_dim=self.input_dim, 
                              output_dim=self.hidden_dim, 
                              args=self.args, 
                              dropout=True, 
                              steps=self.args.gnn_steps, 
                              sparse_inputs=False, 
                              activation=nn.Tanh())

        self.readout = ReadoutLayer(input_dim=self.hidden_dim, 
                                    output_dim=self.output_dim, 
                                    args=self.args, 
                                    act=nn.Tanh(), 
                                    sparse_inputs=False, 
                                    dropout=True)
    
    def l2_loss(self):
        loss = None
        for layer in [self.gl1, self.readout]:
            for p in layer.parameters():
                if loss:
                    loss += self.args.weight_decay * p.pow(2).sum()
                else:
                    loss = self.args.weight_decay * p.pow(2).sum()

        return loss / 2

    def forward(self, x, mask, support):
        x = self.gl1(x, mask, support)
        x = self.readout(x, mask)
        x = self.softmax(x)

        return x


