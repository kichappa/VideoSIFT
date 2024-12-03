using CUDA

# function kernel(a, b, c)
#     point1, point2 = let
#         if c == 1
#             a, b
#         else
#             b, a
#         end
#     end
#     point1[2, 2] = 5
#     point2[2, 2] = 6
#     return
# end
# # d = CuDynamicSharedArray(Float32, 2, sizeof(Float32)*2)

# function kernel2(a)
#     @assert a == 1 "a should be 1"
#     c_shared = CuDynamicSharedArray(Float32, 2)
#     if threadIdx().x == 1
#         c_shared[1] = 0.0
#         c_shared[2] = 0.0
#     end
#     sync_threads()

#     if threadIdx().x % 2 == 0
#         CUDA.atomic_add!(pointer(c_shared, 1), Float32(1.0))
#     else
#         CUDA.atomic_add!(CUDA.pointer(c_shared, 2), Float32(1.0))
#     end
#     sync_threads()
#     return
# end

function kernel()
    a1 = 0
    if threadIdx().x == 1
        a1 = threadIdx().y
    end
    sync_warp()
    a2 = CUDA.shfl_sync(CUDA.FULL_MASK, a1, 1)
    @cuprintln("Thread ($(threadIdx().x), $(threadIdx().y)) a = ($a1==>$a2)")
    return
end
# define a 2x2 matrix {{1, 2}, {3, 4}}
# a = CuArray([1 2; 3 4])
# b = CuArray([1 2; 3 4])
# @cuda threads = 5 blocks = 1 kernel(a, b, 1)

# println(collect(a))
# println(collect(b))

a = CUDA.zeros(1024)
@cuda threads = (32, 2) blocks = 1 kernel()
# println(collect(a))