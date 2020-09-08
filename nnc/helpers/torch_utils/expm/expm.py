import torch
import  numpy as np

from nnc.helpers.torch_utils.expm.expm32 import expm32
from nnc.helpers.torch_utils.expm.expm64 import expm64
from nnc.helpers.torch_utils.expm.expm_taylor import expm_taylor as expm_taylor_forward


def expm_frechet(A, E, expm):
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=A.dtype, device=A.device, requires_grad=False)
    M[:n, :n] = A
    M[n:, n:] = A
    M[:n, n:] = E
    return expm(M)[:n, n:]


class expm_class(torch.autograd.Function):
    @staticmethod
    def _expm_func(A):
        if A.element_size() > 4:
            return expm64
        else:
            return expm32

    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        expm = expm_class._expm_func(A)
        return expm(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        expm = expm_class._expm_func(A)
        return expm_frechet(A.t(), G, expm)


class expm_taylor_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return expm_taylor_forward(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return expm_frechet(A.t(), G, expm_taylor_forward)


expm_taylor = expm_taylor_class.apply
expm = expm_class.apply



#######################################
#        Some very basic tests        #
#######################################

# Basic naïve implementation that about works
def scale_square(X, exp):
    """
    Scale-squaring trick
    Note: This is for just for testing. Apparently, differentiating the scale_squaring trick
            gives a good approximation to the derivative of the exponential (I have not proved this tho)
    """
    norm = X.norm()
    if norm < 1./32.:
        return exp(X)

    # The + 5 makes the norm be less or equal 1/32, small enough so that it is very accurate
    k = int(np.ceil(np.log2(float(norm)))) + 5
    B = X * (2.**-k)
    E = exp(B)
    for _ in range(k):
        E = torch.mm(E, E)
    return E


def taylor(X, n=30):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    coeff = [Id, X]
    for i in range(2, n):
        coeff.append(coeff[-1].mm(X) / i)
    return sum(coeff)

A = torch.rand(5, 5, requires_grad=True)
A = torch.tensor([[0.2438, 0.3366, 0.6083, 0.4208, 0.2997],
                  [0.4911, 0.9196, 0.7790, 0.6629, 0.9682],
                  [0.4104, 0.3005, 0.1019, 0.9837, 0.2015],
                  [0.6454, 0.6973, 0.7667, 0.6931, 0.2697],
                  [0.9324, 0.4042, 0.8409, 0.7221, 0.7703]], requires_grad=True)
E = torch.tensor([[0.3891, 0.1785, 0.9886, 0.8972, 0.6448],
                  [0.9298, 0.3912, 0.9970, 0.2925, 0.2157],
                  [0.1791, 0.0150, 0.5072, 0.5781, 0.0153],
                  [0.2724, 0.5619, 0.8964, 0.2883, 0.5064],
                  [0.7171, 0.1772, 0.8602, 0.4367, 0.2689]])

def ret(A, B, E):
    return torch.autograd.grad([B], A, grad_outputs=(E,))[0]


if __name__ == '__main__':

    # Test gradients with some random 32-bit vectors
    B = expm(A)
    ret1 = ret(A, B, E)

    # Test gradients with the 64 bit algorithm
    A = A.double()
    B = expm(A)
    E = E.double()
    ret2 = ret(A, B, E)

    # Compare against the gradients from differentiating the taylor expansion + scale & square
    B = scale_square(A, taylor)
    ret3 = ret(A, B, E)

    B = expm_taylor(A)
    ret4 = ret(A, B, E)


    print(torch.norm(ret1-ret2.float()))
    print(torch.norm(ret1-ret3.float()))
    print(torch.norm(ret2-ret3))

    print("Taylor")
    print(torch.norm(ret1-ret4.float()))
    print(torch.norm(ret1-ret3))
    print(torch.norm(ret2-ret3))


