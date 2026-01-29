from numba import cuda
import numpy as np

# CUDA kernel
@cuda.jit
def add_vectors_kernel(a, b, result):
    idx = cuda.grid(1)
    if idx < len(result):
        result[idx] = a[idx] + b[idx]

def add_vectors(a, b):
    # Define grid and block dimensions
    blockdim = 256
    griddim = (len(a) + blockdim - 1) // blockdim

    # Allocate memory on the GPU
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.device_array_like(a)

    # Launch the kernel
    add_vectors_kernel[griddim, blockdim](d_a, d_b, d_result)

    # Copy the result back to the host
    result = d_result.copy_to_host()

    return result

# Example usage
if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    result = add_vectors(a, b)
    print("Result:", result)
