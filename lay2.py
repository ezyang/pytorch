"""
Layout utilities for computing covered indices
"""

import sys


sys.path.append("third_party/cutlass/python")

from pycute.int_tuple import is_int, is_tuple
from pycute.layout import Layout


def _generate_coordinates(shape):
    """Generate all coordinate tuples for the given shape (assumes simple tuple of ints)"""
    if is_int(shape):
        for i in range(shape):
            yield i
    elif is_tuple(shape) and all(is_int(s) for s in shape):
        # Use itertools.product for simple multi-dimensional case
        import itertools

        for coord in itertools.product(*[range(s) for s in shape]):
            yield coord
    else:
        raise ValueError(
            f"Unsupported shape: {shape}. Only integer shapes and tuples of integers are supported."
        )


def covered_indices(layout):
    indices = []
    for coord in _generate_coordinates(layout.shape):
        idx = layout(coord)
        indices.append(idx)

    indices.sort()
    return indices


def indices_to_layout(indices):
    """
    Convert a sorted list of indices to a pycute Layout using recursive decomposition.

    Algorithm:
    1. Find the smallest gap between adjacent elements - that's the innermost stride
    2. Group consecutive elements separated by this smallest stride
    3. Each group forms one dimension (group_size, smallest_stride)
    4. Take first element of each group to form a new sequence
    5. Recursively solve the new sequence and combine results

    Args:
        indices: A sorted list of integers starting from 0

    Returns:
        Layout: A pycute Layout that generates the given indices
    """
    if not indices:
        return Layout(0, 0)

    if len(indices) == 1:
        if indices[0] == 0:
            return Layout(1, 1)
        else:
            raise ValueError("Single index must be 0")

    if indices[0] != 0:
        raise ValueError("Indices must start with 0")

    # Base case: consecutive integers starting from 0
    if indices == list(range(len(indices))):
        return Layout(len(indices), 1)

    # Find the smallest gap between adjacent elements
    diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    min_stride = min(diffs)

    # Group consecutive elements that are separated by min_stride
    groups = []
    current_group = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] == min_stride:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]
    groups.append(current_group)

    # All groups must have the same size for a valid layout
    group_size = len(groups[0])
    if not all(len(g) == group_size for g in groups):
        raise ValueError("Inconsistent indices: groups have different sizes")

    # If we only have one group, we're done
    if len(groups) == 1:
        return Layout(group_size, min_stride)

    # Recursively solve for the leaders (first element of each group)
    leaders = [g[0] for g in groups]
    sub_layout = indices_to_layout(leaders)

    # Combine: prepend our dimension to the sub-layout
    if hasattr(sub_layout, "shape") and isinstance(sub_layout.shape, tuple):
        shapes = (group_size,) + sub_layout.shape
        strides = (min_stride,) + sub_layout.stride
    else:
        shapes = (group_size, sub_layout.shape)
        strides = (min_stride, sub_layout.stride)

    return Layout(shapes, strides)


def test_indices_to_layout():
    """Test the indices_to_layout function with various examples"""

    # Test basic cases that should work
    test_cases = [
        ("1D stride 1", Layout(4, 1)),
        ("2D row major", Layout((2, 3), (3, 1))),
        ("2D column major", Layout((2, 3), (1, 2))),
        ("3D simple", Layout((2, 2, 2), (4, 2, 1))),
        ("3D simple 2", Layout((2, 3, 2), (6, 2, 1))),
        ("3D complex", Layout((2, 3, 2), (12, 4, 1))),
        ("4D simple", Layout((2, 2, 2, 2), (8, 4, 2, 1))),
        ("4D complex", Layout((2, 3, 2, 2), (24, 8, 4, 1))),
        ("5D simple", Layout((2, 2, 2, 2, 2), (16, 8, 4, 2, 1))),
        ("5D complex", Layout((2, 3, 2, 2, 2), (48, 16, 8, 4, 1))),
    ]

    all_passed = True
    for name, layout in test_cases:
        indices = covered_indices(layout)
        reconstructed = indices_to_layout(indices)
        back_indices = covered_indices(reconstructed)
        match = back_indices == indices

        print(f"{name}: {'PASS' if match else 'FAIL'}")
        print(f"  Original: {layout}")
        print(f"  Indices: {indices}")
        print(f"  Reconstructed: {reconstructed}")
        if not match:
            print(f"  Back indices: {back_indices}")
            all_passed = False
        print()

    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


def test_malformed_indices():
    """Test that malformed indices properly error without infinite loops"""

    malformed_cases = [
        ("Empty list", []),  # Actually valid
        ("Doesn't start with 0", [1, 2, 3, 4]),
        ("Single non-zero", [5]),
        ("Has gaps", [0, 1, 3, 4]),  # Actually valid pattern
        (
            "Inconsistent groups",
            [0, 1, 4, 5, 8, 10],
        ),  # Last group has different pattern
        ("Duplicate indices", [0, 1, 1, 2]),  # Gets deduplicated to valid pattern
        ("Unsorted indices", [0, 2, 1, 3]),
        ("Negative indices", [-1, 0, 1, 2]),
        ("Non-consecutive with holes", [0, 2, 4, 6, 8, 10, 12]),  # Actually valid
        ("Truly inconsistent 1", [0, 1, 4, 5, 8]),  # Incomplete last group
        ("Truly inconsistent 2", [0, 1, 4, 6]),  # Mixed strides within groups
        ("Random pattern", [0, 3, 7, 12]),  # No clear pattern
        ("Large gap then small", [0, 100, 101]),  # Valid but edge case
    ]

    print("\nTesting malformed indices:")

    for name, indices in malformed_cases:
        try:
            if name == "Duplicate indices":
                # Remove duplicates for this test since covered_indices doesn't create them
                indices = sorted(list(set(indices)))
            if name == "Unsorted indices":
                # Test with unsorted, but the function expects sorted
                pass  # We'll pass unsorted to see what happens

            result = indices_to_layout(indices)
            print(f"{name}: UNEXPECTED SUCCESS - {result}")
        except ValueError as e:
            print(f"{name}: PROPERLY ERRORED - {e}")
        except Exception as e:
            print(f"{name}: UNEXPECTED ERROR TYPE - {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_indices_to_layout()
    test_malformed_indices()
