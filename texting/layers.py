import torch


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