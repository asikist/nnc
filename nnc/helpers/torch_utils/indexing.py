import math
import torch


def multi_unsqueeze(tensor, times, prepend=True):
    """
    Unsqueeze a tensor multiple times towards direction.
    Useful for broadcasting operations.
    :param tensor: The tensor to unsqueeze
    :param times: The number of singular dimensions to add.
    :param prepend: whether to prepend unsqueezes or append them.
    :return: The unsqueezed tensor, which is a view and not a copy.
    """
    old_shape = list(tensor.shape)
    extra_dims = [1]*times
    if prepend:
        return tensor.view(extra_dims + old_shape)
    else:
        return tensor.view(old_shape, extra_dims)


def get_off_diagonal_elements(matrix):
    """
    A vector of the off-diagonal elemetns of the matrix
    :param matrix: The matrix of interest. Can also be rectangular and the diagonal of the
    smaller axis is used.
    :return:
    """
    shape = matrix.shape
    axis_index = list(range(min(shape[0], shape[1])))
    res = matrix.clone()
    res[axis_index, axis_index] = 0
    return res


def get_diagonal_elements(matrix):
    """
    A vector of the diagonal elemetns of the matrix
    :param matrix: The matrix of interest. Can also be rectangular and the diagonal of the
    smaller axis is used.
    :return:
    """
    shape = matrix.shape
    axis_index = list(range(min(shape[0], shape[1])))
    return matrix.clone()[axis_index, axis_index]


def calc_driver_mask(n_nodes, driver_nodes: set, device='cpu', dtype=torch.float):
    """
    Calculates a binary vector mask over graph nodes with unit value on the drive indeces.
    :param n_nodes: numeber of driver nodes in graph
    :param driver_nodes: driver node indeces.
    :param device: the device of the `torch.Tensor`
    :param dtype: the data type of the `torch.Tensor`
    :return: the driver mask vector.
    """
    driver_mask = torch.zeros(n_nodes, device=device, dtype=dtype)
    driver_mask[list(driver_nodes)] = 1
    return driver_mask




# Convolution Utilities below are taken from
# https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
# from user DuaneNielsen

def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
                                              num2tuple(kernel_size), num2tuple(stride), num2tuple(
        pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return h, w

