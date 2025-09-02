#!/usr/bin/env python3

import unittest

import torch
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode


class TestLocalTensor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device("cpu")
        self.shape = (2, 3)
        self.dtype = torch.float32

        # Create sample local tensors for different ranks
        self.local_tensors = {
            0: torch.randn(self.shape, dtype=self.dtype, device=self.device),
            1: torch.randn(self.shape, dtype=self.dtype, device=self.device),
            2: torch.randn(self.shape, dtype=self.dtype, device=self.device),
        }

        # Create identical local tensors for consistency tests
        base_tensor = torch.randn(self.shape, dtype=self.dtype, device=self.device)
        self.identical_local_tensors = {
            0: base_tensor.clone(),
            1: base_tensor.clone(),
            2: base_tensor.clone(),
        }

    def test_local_tensor_creation(self):
        """Test basic LocalTensor creation."""
        lt = LocalTensor(self.local_tensors)

        self.assertIsInstance(lt, LocalTensor)
        self.assertEqual(lt.shape, self.shape)
        self.assertEqual(lt.dtype, self.dtype)
        self.assertEqual(lt.device, self.device)
        self.assertFalse(lt.requires_grad)
        self.assertEqual(len(lt._local_tensors), 3)

    def test_local_tensor_no_grad(self):
        """Test LocalTensor creation - requires_grad is always False."""
        lt = LocalTensor(self.local_tensors)
        self.assertFalse(lt.requires_grad)

    def test_local_tensor_creation_fails_with_grad_tensors(self):
        """Test that LocalTensor creation fails when local tensors have requires_grad=True."""
        grad_tensors = {
            rank: tensor.requires_grad_(True)
            for rank, tensor in self.local_tensors.items()
        }

        with self.assertRaises(AssertionError):
            LocalTensor(grad_tensors)

    def test_local_tensor_shape_consistency(self):
        """Test that LocalTensor enforces shape consistency."""
        inconsistent_tensors = {
            0: torch.randn((2, 3), dtype=self.dtype, device=self.device),
            1: torch.randn(
                (3, 2), dtype=self.dtype, device=self.device
            ),  # Different shape
        }

        with self.assertRaises(AssertionError):
            LocalTensor(inconsistent_tensors)

    def test_local_tensor_dtype_consistency(self):
        """Test that LocalTensor enforces dtype consistency."""
        inconsistent_tensors = {
            0: torch.randn(self.shape, dtype=torch.float32, device=self.device),
            1: torch.randn(
                self.shape, dtype=torch.float64, device=self.device
            ),  # Different dtype
        }

        with self.assertRaises(AssertionError):
            LocalTensor(inconsistent_tensors)

    def test_local_tensor_repr(self):
        """Test LocalTensor string representation."""
        lt = LocalTensor(self.local_tensors)
        repr_str = repr(lt)
        self.assertIn("LocalTensor", repr_str)
        self.assertIn("local_tensors", repr_str)

    def test_tensor_flatten_unflatten(self):
        """Test tensor flatten/unflatten protocol for PT2 tracing."""
        lt = LocalTensor(self.local_tensors)

        # Test flatten
        inner_tensors, flatten_spec = lt.__tensor_flatten__()
        self.assertEqual(inner_tensors, ["_local_tensors"])
        self.assertEqual(flatten_spec, ())

        # Test unflatten
        flattened_tensors = {"_local_tensors": self.local_tensors}
        reconstructed = LocalTensor.__tensor_unflatten__(
            flattened_tensors, flatten_spec, lt.shape, lt.stride()
        )

        self.assertIsInstance(reconstructed, LocalTensor)
        self.assertFalse(reconstructed.requires_grad)
        self.assertEqual(len(reconstructed._local_tensors), 3)

    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations on LocalTensors."""
        lt1 = LocalTensor(self.identical_local_tensors)
        lt2 = LocalTensor(self.identical_local_tensors)

        # Test addition
        result_add = lt1 + lt2
        self.assertIsInstance(result_add, LocalTensor)
        self.assertEqual(len(result_add._local_tensors), 3)

        # Verify the operation was applied to each local tensor
        for rank in self.identical_local_tensors.keys():
            expected = (
                self.identical_local_tensors[rank] + self.identical_local_tensors[rank]
            )
            torch.testing.assert_close(result_add._local_tensors[rank], expected)

        # Test multiplication
        result_mul = lt1 * 2.0
        self.assertIsInstance(result_mul, LocalTensor)
        for rank in self.identical_local_tensors.keys():
            expected = self.identical_local_tensors[rank] * 2.0
            torch.testing.assert_close(result_mul._local_tensors[rank], expected)

    def test_tensor_operations(self):
        """Test various tensor operations on LocalTensors."""
        lt = LocalTensor(self.identical_local_tensors)

        # Test reshape
        reshaped = lt.reshape(-1)
        self.assertIsInstance(reshaped, LocalTensor)
        self.assertEqual(reshaped.shape, (6,))

        # Test transpose
        transposed = lt.transpose(0, 1)
        self.assertIsInstance(transposed, LocalTensor)
        self.assertEqual(transposed.shape, (3, 2))

        # Test sum
        summed = lt.sum()
        self.assertIsInstance(summed, LocalTensor)

        # Test mean
        mean_result = lt.mean()
        self.assertIsInstance(mean_result, LocalTensor)

    def test_mixed_operations_with_regular_tensors(self):
        """Test operations between LocalTensors and regular tensors."""
        lt = LocalTensor(self.identical_local_tensors)
        regular_tensor = torch.ones_like(self.identical_local_tensors[0])

        # Test LocalTensor + regular tensor
        result = lt + regular_tensor
        self.assertIsInstance(result, LocalTensor)

        for rank in self.identical_local_tensors.keys():
            expected = self.identical_local_tensors[rank] + regular_tensor
            torch.testing.assert_close(result._local_tensors[rank], expected)

    def test_gradient_propagation(self):
        """Test that gradients work correctly with LocalTensors."""
        # Create LocalTensor
        lt = LocalTensor(self.identical_local_tensors)

        # Simple operation - LocalTensors always have requires_grad=False
        result = lt * 2.0
        self.assertFalse(result.requires_grad)

    def test_local_tensor_mode(self):
        """Test LocalTensorMode functionality."""
        lt = LocalTensor(self.identical_local_tensors)

        with LocalTensorMode():
            # Operations within the mode should work the same
            result = lt + 1.0
            self.assertIsInstance(result, LocalTensor)

            # Regular tensor operations should still work
            regular = torch.ones(2, 2)
            regular_result = regular + 1.0
            self.assertIsInstance(regular_result, torch.Tensor)
            self.assertNotIsInstance(regular_result, LocalTensor)

    def test_different_rank_sets(self):
        """Test LocalTensors with different sets of ranks."""
        lt1_ranks = {0: torch.randn(self.shape), 1: torch.randn(self.shape)}
        lt2_ranks = {1: torch.randn(self.shape), 2: torch.randn(self.shape)}

        lt1 = LocalTensor(lt1_ranks)
        lt2 = LocalTensor(lt2_ranks)

        # Operations should work on the intersection of ranks
        result = lt1 + lt2
        self.assertIsInstance(result, LocalTensor)
        # Should only have rank 1 in common
        self.assertEqual(set(result._local_tensors.keys()), {1})

    def test_empty_local_tensors(self):
        """Test behavior with empty local tensors dict."""
        with self.assertRaises(StopIteration):  # next() on empty iterator
            LocalTensor({})

    def test_single_rank_tensor(self):
        """Test LocalTensor with only one rank."""
        single_rank_tensors = {0: torch.randn(self.shape)}
        lt = LocalTensor(single_rank_tensors)

        self.assertIsInstance(lt, LocalTensor)
        self.assertEqual(len(lt._local_tensors), 1)

        # Operations should still work
        result = lt * 2.0
        self.assertIsInstance(result, LocalTensor)
        self.assertEqual(len(result._local_tensors), 1)

    def test_complex_operations(self):
        """Test more complex tensor operations."""
        lt = LocalTensor(self.identical_local_tensors)

        # Chain multiple operations
        result = ((lt + 1.0) * 2.0).transpose(0, 1).sum(dim=0)
        self.assertIsInstance(result, LocalTensor)

        # Verify the result makes sense
        for rank in self.identical_local_tensors.keys():
            expected = (
                ((self.identical_local_tensors[rank] + 1.0) * 2.0)
                .transpose(0, 1)
                .sum(dim=0)
            )
            torch.testing.assert_close(result._local_tensors[rank], expected)


if __name__ == "__main__":
    unittest.main()
