"""
Implementation of complement cross-entropy loss
"""


import torch
import torch.nn.functional as F

def complement_entropy(y_hat, y):
    assert len(y_hat.shape) == len(y.shape) == 2

    not_y = torch.bitwise_not(y.to(torch.bool)).to(torch.uint8)
    y_hat_wrong = torch.mul(y_hat, not_y)
    
    y_hat_corr = torch.mul(y_hat, y)
    y_hat_corr = torch.sum(y_hat_corr, axis=1)
    y_hat_ccomp = 1 - y_hat_corr
    y_hat_ccomp = torch.repeat_interleave(y_hat_ccomp.reshape(-1, 1), y_hat.size(1), 1)

    assert y_hat_wrong.shape == y_hat_ccomp.shape

    y_hat_wrong_norm = torch.div(y_hat_wrong, y_hat_ccomp + 1e-20)
    log_y_hat_wrong_norm = torch.log(y_hat_wrong_norm + 1e-20)

    y_hat_mult = torch.mul(y_hat_wrong_norm, log_y_hat_wrong_norm)
    # Setting nans to 0 as per https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/10
    y_hat_mult = torch.where(torch.isnan(y_hat_mult), torch.zeros_like(y_hat_mult), y_hat_mult)

    sum_over_K = torch.sum(y_hat_mult, axis=1)
    mean_over_N = torch.mean(sum_over_K)

    return mean_over_N

def complement_cross_entropy_loss(y_hat, y, gamma=-1):
    assert y_hat.size() == y.size(), "Input tensors must have equal shape"

    h = F.cross_entropy(y_hat, torch.argmax(y, dim=1))
    y_hat = F.softmax(y_hat, dim=1)
    c = gamma / (y.size(1) - 1) * complement_entropy(y_hat, y)
    return h + c


if __name__ == "__main__":
    # 1
    y_hat = torch.tensor([[0.3, 0.3, 0.4], [0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.8, 0.1]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)

    loss = complement_entropy(y_hat, y)
    print(loss)

    loss.backward()

    # 2
    y_hat = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]], requires_grad=True)
    y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]], requires_grad=True)

    loss = complement_entropy(y_hat, y)
    print(loss)

    # 3
    y_hat = torch.tensor([[0.3, 0.3, 0.4], [0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.8, 0.1]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)

    loss = complement_cross_entropy_loss(y_hat, y)
    print(loss)

    loss.backward()

    # 4
    y_hat = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]], requires_grad=True)
    y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.1]], requires_grad=True)

    loss = complement_cross_entropy_loss(y_hat, y)
    print(loss)
