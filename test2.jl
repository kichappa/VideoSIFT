using CUDA

# define a new arrays of 64 elements, and fill it with random ones and zeros
a = rand(0:1, 1024*5)

a_gpu = CuArray(a)
b_gpu = CUDA.zeros(Int64, 1024*5)
count = CUDA.zeros(Int64, 1)

function mykernel!(in, out, count)
	threadNum = threadIdx().x + blockDim().x * (blockIdx().x-1) # 1-indexed
	warpNum = (threadIdx().x - 1) ÷ 32 # 0-indexed
	laneNum = (threadIdx().x - 1) % 32 # 0-indexed

    shared_count = CuDynamicSharedArray(Int64, 1)
    
    if threadNum == 1
        shared_count[1] = 0
    end
    sync_threads()

    if threadNum <= 1024*5
        is_nonzero = in[threadNum] != 0
        mask = CUDA.vote_ballot_sync(0xffffffff, is_nonzero)
        warp_count = count_ones(mask)

        warp_offset = 0
        if laneNum == 0
            warp_offset = CUDA.atomic_add!(pointer(shared_count), warp_count)
        end
        warp_offset = CUDA.shfl_sync(0xffffffff, warp_offset, 1) #<<<<< This is the BUG code.

        if is_nonzero
            index = count_ones(mask & ((1 << laneNum) - 1)) + warp_offset
            out[index+1] = threadNum
        end
    end
    sync_threads()

    if threadIdx().x == 1
        CUDA.atomic_add!(CUDA.pointer(count), shared_count[1])
    end
	return
end

@cuda threads = 1024 blocks = 5 shmem=sizeof(Int64) mykernel!(a_gpu, b_gpu, count)

println("nonzeros:$(collect(count))")
println(a)
println("----------------")
println(collect(b_gpu))

