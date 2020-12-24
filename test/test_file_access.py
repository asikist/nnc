import unittest
import torch
from nnc.helpers.torch_utils.file_helpers import read_tensor_from_collection, \
    save_tensor_to_collection


class FileAccessTests(unittest.TestCase):
    def test_file_access(self):
        collection_file = './resources/collection'
        file_a = 'a'
        file_b = 'b'
        a = torch.rand([10, 10])
        b = torch.rand([10, 10])
        save_tensor_to_collection(collection_file, file_a, a)
        save_tensor_to_collection(collection_file, file_b, b)

        aa = read_tensor_from_collection(collection_file, file_a)
        self.assertTrue(torch.all(a==aa))

        bb = read_tensor_from_collection(collection_file, file_b)
        self.assertTrue(torch.all(b==bb))


if __name__ == '__main__':
    unittest.main()
