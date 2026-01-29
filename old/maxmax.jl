using CUDA

vid_time = 2 * 60 #min
f
A = CUDA.rand(Float32, vid_time * 60 * 30 * 50, 50)

function maxmax_kernel_shared(inp, max, maxloc, width, height)
    threadNum::Int16 = threadIdx().x # one based
    threads::Int16 = blockDim().x # number of threads per block
    blockNum::Int32 = blockIdx().x # one based

    maxmaxval = CuDynamicSharedArray(Float32, (2, threads))
    maxmaxloc = CuDynamicSharedArray(Int32, (2, threads))

    sharedA = CuDynamicSharedArray(Float32, (threads, height))
    if (threadNum + (blockNum - 1) * threads) <= width
        # coalesced copy to shared memory
        for i in 1:height
            if blockNum == 1 && i == height
                # @cuprintln("threadNum: $threadNum, blockNum: $blockNum, i: $i, inp[$(threadNum+(blockNum-1)*threads+(i-1)*width)]: (inp[threadNum+(blockNum-1)*threads+i*width])")
            end
            # sharedA[threadNum, i] = rand(Float32) #inp[threadNum+(blockNum-1)*threads+(i-1)*width]
            sharedA[threadNum, i] = 0.0
            if threadNum == 1
                @cuprintln("copying inp[$(threadNum+(blockNum-1)*threads+(i-1)*width)]: $(inp[threadNum+(blockNum-1)*threads+(i-1)*width]) to sharedA[$threadNum, $i]: $(sharedA[threadNum, i])")
            end
            sharedA[threadNum, i] = inp[threadNum+(blockNum-1)*threads+(i-1)*width]
            if threadNum == 1
                @cuprintln("\tcopied inp[$(threadNum+(blockNum-1)*threads+(i-1)*width)]: $(inp[threadNum+(blockNum-1)*threads+(i-1)*width]) to sharedA[$threadNum, $i]: $(sharedA[threadNum, i])")
            end
            sync_threads()
        end

        if threadNum == 1
            @cuprint("sharedA:[\n")
            for j in 1:threads
                @cuprint("[")
                for i in 1:height
                    @cuprint("$(sharedA[j, i]), ")
                end
                @cuprintln("]")
            end
            @cuprintln("]")
        end

        maxmaxval[1, threadNum] = -10
        maxmaxloc[1, threadNum] = 0

        sync_threads()

        for i in 1:height
            if threadNum == 1
                @cuprintln("checking for sharedA[$threadNum, $i]: $(sharedA[threadNum, i]), current max: $(maxmaxval[1, threadNum]) at $(maxmaxloc[1, threadNum])")
            end
            if sharedA[threadNum, i] > maxmaxval[1, threadNum]
                if threadNum == 1
                    @cuprintln("\tfound new max: sharedA[$threadNum, $i]: $(sharedA[threadNum, i]), writing on to maxmaxval[1, $threadNum]: $(maxmaxval[1, threadNum]) at $(maxmaxloc[1, threadNum])")
                end
                maxmaxval[2, threadNum] = 100 #maxmaxval[1, threadNum]
                maxmaxloc[2, threadNum] = 10 #maxmaxloc[1, threadNum]
                maxmaxval[1, threadNum] = 200 #sharedA[threadNum, i]
                maxmaxloc[1, threadNum] = i
                if threadNum == 1
                    @cuprintln("\t\tupdated max: maxmaxval[1, $threadNum]: $(maxmaxval[1, threadNum]) at $(maxmaxloc[1, threadNum])")
                end
            elseif sharedA[threadNum, i] > maxmaxval[2, threadNum]
                maxmaxval[2, threadNum] = 101 #sharedA[threadNum, i]
                maxmaxloc[2, threadNum] = i
            end
        end

        sync_threads()

        # coalesced write to global memory
        max[threadNum+(blockNum-1)*threads, 1] = maxmaxval[1, threadNum]
        sync_threads()
        max[threadNum+(blockNum-1)*threads, 2] = maxmaxval[2, threadNum]
        sync_threads()
        maxloc[threadNum+(blockNum-1)*threads, 1] = maxmaxloc[1, threadNum]
        sync_threads()
        maxloc[threadNum+(blockNum-1)*threads, 2] = maxmaxloc[2, threadNum]
        sync_threads()
    end
    return
end

function maxmax_kernel(inp, max, maxloc, width, height)

    threadNum::Int16 = threadIdx().x # one based
    threads::Int16 = blockDim().x # number of threads per block
    blockNum::Int32 = blockIdx().x # one based

    if (threadNum + (blockNum - 1) * threads) <= width
        max1 = -10
        maxloc1 = 0
        max2 = -11
        maxloc2 = 0

        for i in 1:height
            if (threadNum + (blockNum - 1) * threads) <= width
                if inp[threadNum+(blockNum-1)*threads+(i-1)*width] > max1
                    # @cuprintln("($threadNum, $blockNum): new max found inp[$(threadNum+(blockNum-1)*threads+(i-1)*width)]: $(inp[threadNum+(blockNum-1)*threads+(i-1)*width]) > max1: $max1 at $maxloc1, updating max1: ")
                    max2 = max1
                    maxloc2 = maxloc1
                    max1 = inp[threadNum+(blockNum-1)*threads+(i-1)*width]
                    maxloc1 = i
                    # @cuprintln("\t($threadNum, $blockNum): $max1 at $maxloc1")
                elseif inp[threadNum+(blockNum-1)*threads+(i-1)*width] > max2
                    max2 = inp[threadNum+(blockNum-1)*threads+(i-1)*width]
                    maxloc2 = i
                end
            end
            sync_threads()
        end
        max[threadNum+(blockNum-1)*threads, 1] = max1
        sync_threads()
        max[threadNum+(blockNum-1)*threads, 2] = max2
        sync_threads()
        maxloc[threadNum+(blockNum-1)*threads, 1] = maxloc1
        sync_threads()
        maxloc[threadNum+(blockNum-1)*threads, 2] = maxloc2
        sync_threads()
    end
    return
end

function maxmax_kernel_nosync(inp, max, maxloc, width, height)

    threadNum::Int16 = threadIdx().x # one based
    threads::Int16 = blockDim().x # number of threads per block
    blockNum::Int32 = blockIdx().x # one based

    if (threadNum + (blockNum - 1) * threads) <= width
        max1 = -10
        maxloc1 = 0
        max2 = -11
        maxloc2 = 0

        for i in 1:height
            if (threadNum + (blockNum - 1) * threads) <= width
                if inp[threadNum+(blockNum-1)*threads+(i-1)*width] > max1
                    # @cuprintln("($threadNum, $blockNum): new max found inp[$(threadNum+(blockNum-1)*threads+(i-1)*width)]: $(inp[threadNum+(blockNum-1)*threads+(i-1)*width]) > max1: $max1 at $maxloc1, updating max1: ")
                    max2 = max1
                    maxloc2 = maxloc1
                    max1 = inp[threadNum+(blockNum-1)*threads+(i-1)*width]
                    maxloc1 = i
                    # @cuprintln("\t($threadNum, $blockNum): $max1 at $maxloc1")
                elseif inp[threadNum+(blockNum-1)*threads+(i-1)*width] > max2
                    max2 = inp[threadNum+(blockNum-1)*threads+(i-1)*width]
                    maxloc2 = i
                end
            end
            # sync_threads()
        end
        max[threadNum+(blockNum-1)*threads, 1] = max1
        # sync_threads()
        max[threadNum+(blockNum-1)*threads, 2] = max2
        # sync_threads()
        maxloc[threadNum+(blockNum-1)*threads, 1] = maxloc1
        # sync_threads()
        maxloc[threadNum+(blockNum-1)*threads, 2] = maxloc2
        # sync_threads()
    end
    return
end

function maxmax(A)
    width, height = size(A)
    max = CUDA.zeros(Float32, width, 2)
    maxloc = CUDA.zeros(Int32, width, 2)
    threads = min(width, 640)
    blocks = cld(width, threads)
    # shmem = 2 * threads * (sizeof(Float32) + sizeof(Int32)) + threads * height * sizeof(Float32)
    # println("Shared Memory Size: $shmem bytes")
    @cuda threads = threads blocks = blocks maxmax_kernel(A, max, maxloc, width, height)
    # kernel = @cuda maxmax_kernel(A, max, maxloc, width, height)
    # println(launch_configuration(kernel.fun))
    return Array(max), Array(maxloc)
end

function maxmax_nosync(A)
    width, height = size(A)
    max = CUDA.zeros(Float32, width, 2)
    maxloc = CUDA.zeros(Int32, width, 2)
    threads = min(width, 640)
    blocks = cld(width, threads)
    @cuda threads = threads blocks = blocks maxmax_kernel_nosync(A, max, maxloc, width, height)
    # kernel = @cuda maxmax_kernel_nosync(A, max, maxloc, width, height)
    # println(launch_configuration(kernel.fun))
    return Array(max), Array(maxloc)
end

function maxmax_nokernel(A)
    maxmax = zeros(Float32, size(A, 1), 2)
    maxloc = zeros(Int32, size(A, 1), 2)
    width, height = size(A)
    for i in axes(A, 1)
        max1 = -10
        maxloc1 = 0
        max2 = -11
        maxloc2 = 0
        for j in axes(A, 2)
            if A[i+(j-1)*width] > max1
                max2 = max1
                maxloc2 = maxloc1
                max1 = A[i+(j-1)*width]
                maxloc1 = j
            elseif A[i+(j-1)*width] > max2
                max2 = A[i+(j-1)*width]
                maxloc2 = j
            end
        end
        maxmax[i, 1] = max1
        maxmax[i, 2] = max2
        maxloc[i, 1] = maxloc1
        maxloc[i, 2] = maxloc2
    end
end

# for i in axes(A, 1)
#     println(A[i, :])
# end
# println()
time_taken = 0
maxmax(A)
maxmax_nosync(A)
iterations = 100
for i in 1:iterations
    start_t = time_ns()
    maxv, maxloc = maxmax(A)
    end_t = time_ns()
    global time_taken += (end_t - start_t)
end
println("Time taken for sync kernel: $(time_taken/1.0e9) seconds")

time_taken = 0
iterations = 100
for i in 1:iterations
    start_t = time_ns()
    maxv, maxloc = maxmax_nosync(A)
    end_t = time_ns()
    global time_taken += (end_t - start_t)
end
println("Time taken for no sync kernel: $(time_taken/1.0e9) seconds")
# CUDA.allowscalar(true)
# time_taken = 0
# for i in 1:iterations
#     start_t = time_ns()
#     maxmax_nokernel(A)
#     end_t = time_ns()
#     global time_taken += (end_t - start_t)
# end
# println("Time taken in non-kernel: $(time_taken/1.0e9) seconds")
# CUDA.allowscalar(false)
# maxv, maxloc = maxmax(A)
# println(maxv)
# println(maxloc)