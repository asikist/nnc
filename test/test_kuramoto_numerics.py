import unittest
import math
import torch
from nnc.helpers.torch_utils.numerics import cos_difference,\
                                             sin_difference,\
                                             sin_difference_mem,\
                                             cos_difference_mem

class MyTestCase(unittest.TestCase):
    def test_sin(self):
        A = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]).to(torch.float)
        x = torch.tensor([2, 3, 6]).to(torch.float)
        res = torch.tensor([
            1*math.sin(3.-2.) + 0*math.sin(6.-2.),
            1*math.sin(2.-3.) + 1*math.sin(6.-3),
            0*math.sin(2.-6.) + 1*math.sin(3.-6.)
        ])
        y = sin_difference_mem(x,x,A)#torch.sin((x.unsqueeze(-1)-x.unsqueeze(-2))*A).sum(-1)
        y2 = sin_difference(x, x, A)
        self.assertTrue(torch.allclose(y, y2,atol=1e-6))
        self.assertTrue(torch.allclose(res, y2,atol=1e-6))
        self.assertTrue(torch.allclose(res, y,atol=1e-6))

    def test_sin_batch(self):

        A = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]).to(torch.float)
        x = torch.tensor([[2, 3, 6]]).to(torch.float)
        y = sin_difference_mem(x, x, A)#torch.sin((x.unsqueeze(-1) - x.unsqueeze(-2)) * A).sum(-1)
        y2 = sin_difference(x, x, A)
        self.assertTrue(torch.allclose(y, y2, atol=1e-6))




    def test_cos(self):
        A = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]).to(torch.float)
        x = torch.tensor([2, 3, 6]).to(torch.float)
        y = cos_difference_mem(x,x,A)#torch.sin((x.unsqueeze(-1)-x.unsqueeze(-2))*A).sum(-1)
        y2 = cos_difference(x, x, A)
        print(y)
        print(y2)
        self.assertTrue(torch.allclose(y, y2,atol=1e-6))

    def test_cos_batch(self):

        A = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]).to(torch.float)
        x = torch.tensor([[2, 3, 6]]).to(torch.float)
        y = cos_difference_mem(x, x, A)#torch.sin((x.unsqueeze(-1) - x.unsqueeze(-2)) * A).sum(-1)
        y2 = cos_difference(x, x, A)
        y3 = sin_difference(x+math.pi/2,x, A)
        self.assertTrue(torch.allclose(y, y2, atol=1e-6))
        self.assertTrue(torch.allclose(y3, y2, atol=1e-6))

    def test_random_batch(self):
        A = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ]).to(torch.float)
        x = torch.rand(5, 4)

        y = cos_difference_mem(x, x,
                               A)  # torch.sin((x.unsqueeze(-1) - x.unsqueeze(-2)) * A).sum(-1)
        y2 = cos_difference(x, x, A)
        y3 = sin_difference(x + math.pi / 2, x, A)
        self.assertTrue(torch.allclose(y, y2, atol=1e-6))
        self.assertTrue(torch.allclose(y3, y2, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
