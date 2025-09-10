"""
Layout utilities for computing covered indices
"""

import sys
sys.path.append('third_party/cutlass/python')

from pycute.layout import Layout
from pycute.int_tuple import is_int, is_tuple


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
        raise ValueError(f"Unsupported shape: {shape}. Only integer shapes and tuples of integers are supported.")


def covered_indices(layout):
    indices = []
    for coord in _generate_coordinates(layout.shape):
        idx = layout(coord)
        indices.append(idx)
    
    indices.sort()
    return indices


def indices_to_layout(indices):
    """
    Convert a sorted list of indices to a pycute Layout using the mathematical
    approach based on the admissible for complement property.
    
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
    
    strides = []
    shapes = []
    remaining = set(indices)
    
    # Always start with stride 1
    current_stride = 1
    max_iterations = len(indices)  # Safety limit
    iteration = 0
    
    while remaining and iteration < max_iterations:
        iteration += 1
        
        # Count consecutive multiples of current_stride starting from 0
        size = 0
        while size * current_stride in remaining:
            size += 1
        
        if size > 0:
            # Found a valid dimension - remove all multiples
            for i in range(size):
                remaining.discard(i * current_stride)
            strides.append(current_stride)
            shapes.append(size)
            
            # Calculate next stride using admissible property
            current_stride = current_stride * size
        else:
            # No pattern from 0, jump to minimum remaining element
            if remaining:
                current_stride = min(remaining)
            else:
                break
    
    if iteration >= max_iterations:
        raise RuntimeError(f"Algorithm did not converge after {max_iterations} iterations")
    
    # Convert to proper format for Layout
    if len(shapes) == 1:
        return Layout(shapes[0], strides[0])
    else:
        return Layout(tuple(shapes), tuple(strides))


def test_indices_to_layout():
    """Test the indices_to_layout function with various examples"""
    
    # Test basic cases that should work
    test_cases = [
        #   ("1D stride 1", Layout(4, 1)),
        #   ("2D row major", Layout((2, 3), (3, 1))),
        #   ("2D column major", Layout((2, 3), (1, 2))),
        #   ("3D simple", Layout((2, 2, 2), (4, 2, 1))),
        #   ("3D simple 2", Layout((2, 3, 2), (6, 2, 1))),
        ("3D complex", Layout((2, 3, 2), (12, 4, 1))),
    ]
    
    # Note: The 3D strided case Layout((2, 2, 2), (8, 4, 1)) and other cases with
    # gaps require a more sophisticated algorithm that's beyond the current implementation.
    # The current algorithm works for dense/contiguous layouts but needs enhancement
    # for sparse layouts with complex stride patterns.
    
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


if __name__ == "__main__":
    test_indices_to_layout()
