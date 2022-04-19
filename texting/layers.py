import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from inits import glorot, zeros


def sparse_dropout(x, keep_prob, noise_shape):
    """
    x: Input sparse tensor
    keep_prob: 1 - dropout probability
    noise_shape: number of non zero elements in the sparse tensor
    """
    random_tensor = keep_prob
    random_tensor += torch.rand(noise_shape)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]
    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1./ (keep_prob))
    return out


def sparse_dense_matmul_batch(sp_a, b):
    """
    Wrapper around torch.sparse.mm.

    Parameters
    ----------
    sp_a: torch SparseTensor
        mat1
    b: torch Tensor
        mat2
    
    Returns
    -------
    c: torch Tensor
        mat1 x mat2
    """
    return torch.sparse.mm(sp_a, b)


def dot(x, y, sparse=False):
    """
    Multiplies tensors x and y

    Parameters
    ----------
    x: torch Tensor or SparseTensor
        mat1
    y: torch Tensor or SparseTensor
        mat2
    
    Returns
    -------
    res: torch Tensor or SparseTensor
        mat1 x mat2

    """
    if sparse:
        res = sparse_dense_matmul_batch(x, y)
    else:
        res = torch.mm(x, y)

    return res

def gru_unit(support, x, var, act, mask, dropout, sparse_inputs=False):
    return x


class Dense(nn.Module):
    """
    Implements a Dense layer where inputs can be Tensors or SparseTensors
    """

    def __init__(self, input_dim, output_dim, num_features_nonzero, activation=nn.ReLU, 
                dropout=0., sparse_inputs=False, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.num_features_nonzero = num_features_nonzero
        
        self.weight = Parameter(glorot((output_dim, input_dim)))

        if bias:
            self.bias = Parameter(zeros((output_dim)))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            dropout = nn.Dropout(p=1-self.dropout)
            x = dropout(x)

        output = dot(x, self.weights, sparse=self.sparse_inputs)

        if self.bias:
            output += self.bias

        return self.activation(output)


class GraphLayer(nn.Module):
    """
    Implements a GraphLayer which can have sparse or dense inputs.
    """

    def __init__(self, input_dim, output_dim, configs, dropout=False,
                 sparse_inputs=False, activation=nn.ReLU, bias=False,
                 featureless=False, steps=2, **kwargs):
        """ Note: dropout value is passed from config, and not dropout. """

        super(GraphLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = configs['dropout']
        else:
            self.dropout = 0.
        
        self.activation = activation
        self.support = configs['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mask = configs['mask']
        self.steps = steps

        # helper variable for sparse dropout
        self.num_features_nonzero = configs['num_features_nonzero']

        self.vars = nn.ParameterDict({
            'weights_encode': Parameter(glorot([input_dim, output_dim])),
            'weights_z0': Parameter(glorot([output_dim, output_dim])),
            'weights_z1': Parameter(glorot([output_dim, output_dim])),
            'weights_r0': Parameter(glorot([output_dim, output_dim])),
            'weights_r1': Parameter(glorot([output_dim, output_dim])),
            'weights_h0': Parameter(glorot([output_dim, output_dim])),
            'weights_h1': Parameter(glorot([output_dim, output_dim])),
            'bias_encode': Parameter(zeros([output_dim])),
            'bias_z0': Parameter(zeros([output_dim])),
            'bias_z1': Parameter(zeros([output_dim])),
            'bias_r0': Parameter(zeros([output_dim])),
            'bias_r1': Parameter(zeros([output_dim])),
            'bias_h0': Parameter(zeros([output_dim])),
            'bias_h1': Parameter(zeros([output_dim]))
        })

    def forward(self, x):
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            dropout = nn.Dropout(p=1-self.dropout)
            x = dropout(x)

        x = dot(x, self.vars['weights_encode'], self.sparse_inputs) + self.vars['bias_encode']
        output = self.mask * self.activation(x)

        for _  in range(self.steps):
            output = gru_unit(self.support, output, self.vars, 
                              self.activation, self.mask, 1-self.dropout, 
                              self.sparse_inputs)

        return output


class ReadoutLayer(nn.Module):
    """ Implements Readout layer of TextING. """

    def __init__(self, input_dim, output_dim, configs, dropout=0.,
                 sparse_inputs=False, act=nn.ReLU, bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = configs['dropout']
        else:
            self.dropout = 0.
        
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.mask = configs['mask']

        self.vars = nn.ParameterDict({
            'weights_att': glorot([input_dim, 1]),
            'weights_emb': glorot([input_dim, input_dim]),
            'weights_mlp': glorot([input_dim, output_dim]),
            'bias_att': zeros([1]),
            'bias_emb': zeros([input_dim]),
            'bias_mlp': zeros([output_dim])
        })

    def forward(self, x):
        att = nn.Sigmoid(dot(x, self.vars['weights_att']) + self.vars['bias_att'])
        emb = self.act(dot(x, self.vars['weights_emb']) + self.vars['bias_emb'])

        N = torch.sum(self.mask, dim=1)
        M = (self.mask-1) * 1e9
    
        # TODO: Re-implement for sparse inputs
        g = self.mask * att * emb
        g = torch.sum(g, dim=1) / N + torch.sum(g + M, dim=1)
        g = nn.Dropout(g, 1-self.dropout)

        # Classify
        # TODO: Re-implement using dot
        output = torch.mm(g, self.vars['weights_mlp']) + self.vars['bias_mlp']        

        return output