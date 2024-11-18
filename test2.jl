using CUDA
using BenchmarkTools

# arr = rand(32, 32, 1200)

# function mykernel2(inp)
#     x = threadIdx().x
#     y = threadIdx().y
#     z = blockIdx().x

#     if x <= 32 && y <= 32
#         @inbounds inp[y, x, z] += 1
#     end

#     return
# end

# arr_GPU = CuArray(arr)

# @cuda threads = (32, 32) blocks = 1200 mykernel2(arr_GPU)

# time_taken = 0
# for i in 1:100
#     start_t = time_ns()
#     global arr_GPU = CuArray(arr)
#     @cuda threads = (32, 32) blocks = 1200 mykernel2(arr_GPU)
#     end_t = time_ns()
#     global time_taken += (end_t - start_t) / 1e6
# end

# println("Time: ", time_taken / 100, " ms")

function mykernel()
    @cuprintln("blockdim: ", blockDim().x, " ", blockDim().y, " ", blockDim().z)
    return
end

@cuda threads = (2, 3) blocks = 1 mykernel()