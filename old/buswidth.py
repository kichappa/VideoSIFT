import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA kernel code to query memory bus width
cuda_code = """
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void query_memory_bus_width()
{
    int width = 0;
    cudaDeviceGetAttribute(&width, cudaDevAttrMemoryBusWidth, 0);
    // printf("Memory Bus Width: %d bits\\n", width);
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)

# Get function pointer to CUDA kernel
query_bus_width = mod.get_function("query_memory_bus_width")

# Call the CUDA kernel
query_bus_width(block=(1, 1, 1), grid=(1, 1))

# Synchronize to ensure the kernel has completed
cuda.Context.synchronize()