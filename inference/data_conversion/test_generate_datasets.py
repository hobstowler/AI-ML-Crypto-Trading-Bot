
import unittest

from generate_datasets import generate_start_indicies


class TestGenerateStartIndicies(unittest.TestCase):

    # Test for invalid splits
    def test_invalid_splits(self):
        with self.assertRaises(ValueError):
            generate_start_indicies(100, 3, 2, 0.8, 0.2, 0.1, seed=None)

    # Test for invalid length
    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            generate_start_indicies(4, 3, 2, 0.8, 0.1, 0.1, seed=None)

    # Test lengths for an evenly divisible split
    def test_train_length(self):
        indicies = generate_start_indicies(100, 3, 2, 0.8, 0.1, 0.1, seed=None)
        len_train = len(indicies[0])
        expected = 16
        self.assertEqual(len_train, expected)

    def test_val_length(self):
        indicies = generate_start_indicies(100, 3, 2, 0.8, 0.1, 0.1, seed=None)
        len_val = len(indicies[1])
        expected = 2
        self.assertEqual(len_val, expected)

    def test_test_length(self):
        indicies = generate_start_indicies(100, 3, 2, 0.8, 0.1, 0.1, seed=None)
        len_test = len(indicies[2])
        expected = 2
        self.assertEqual(len_test, expected)

    # Test lengths for a non-evenly divisible split (Should ignore last non-full set)
    def test_train_length2(self):
        indicies = generate_start_indicies(104, 3, 2, 0.8, 0.1, 0.1, seed=None)
        len_train = len(indicies[0])
        expected = 16
        self.assertEqual(len_train, expected)

    def test_val_length2(self):
        indicies = generate_start_indicies(104, 3, 2, 0.8, 0.1, 0.1, seed=None)
        len_val = len(indicies[1])
        expected = 2
        self.assertEqual(len_val, expected)

    def test_test_length2(self):
        indicies = generate_start_indicies(104, 3, 2, 0.8, 0.1, 0.1, seed=None)
        len_test = len(indicies[2])
        expected = 2
        self.assertEqual(len_test, expected)

    # Test actual values of indicies with seed of 0
    def test_train_indicies(self):
        indicies = generate_start_indicies(30, 2, 1, 0.6, 0.3, 0.1, seed=0)
        train_indicies = indicies[0]
        expected = [6, 24, 12, 27,  3, 18]
        self.assertEqual(train_indicies.tolist(), expected)

    def test_val_indicies(self):
        indicies = generate_start_indicies(30, 2, 1, 0.6, 0.3, 0.1, seed=0)
        val_indicies = indicies[1]
        expected = [21, 9, 0]
        self.assertEqual(val_indicies.tolist(), expected)

    def test_test_indicies(self):
        indicies = generate_start_indicies(30, 2, 1, 0.6, 0.3, 0.1, seed=0)
        test_indicies = indicies[2]
        expected = [15]
        self.assertEqual(test_indicies.tolist(), expected)

if __name__ == '__main__':
    unittest.main()