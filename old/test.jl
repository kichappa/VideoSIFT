using CUDA, StaticArrays

function test_kernel(A, shmemA)
    n = threadIdx().x
    # data = Vector{CuPtr{Float32}}(undef, length(A))
    data1 = CuDynamicSharedArray(Float64, 5, 5)
    data = SVector{length(A), CuDeviceVector{Float64, 3}}(data1, data1, data1)
    @cuprintln("typeof data1: ", typeof(data1))
    @cuprintln("typeof pointer of data1: ", typeof(pointer(data1)))
    @cuprintln("typeof shmemA[1] ", typeof(data))

    # shmemA[1] = data1

    # data1[1] = pointer(data1)
    # @cuprintln("typeof data[1]: ", typeof(data[1]))
		# @cuprintln("Matrix $n")
        # @cuprintln("size of A[n]: ", size(A[n]))
		# for i in 1:size(A[n],1)
		# 	for j in 1:size(A[n],2)
		# 		@cuprint(A[n][i, j], " ")
		# 	end
		# 	@cuprintln()
		# end
	return
end

A = Vector{Matrix}()
push!(A, rand(2, 2))
push!(A, rand(3, 3))
push!(A, rand(4, 4))

for i in eachindex(A)
	println("Matrix $i")
	println(A[i])
end

ptrs = CuArray([pointer(x) for x in A])
sizes = CuArray([size(x) for x in A])

A_gpu = CuArray(A)

shmemA = CuArray{Core.LLVMPtr{Float64, 3}}(undef, length(A))

@cuda threads = size(A, 1) blocks = 1 shmem=5*5*sizeof(Float64) test_kernel(CuArray([cudaconvert(CuArray(x)) for x in A]), CuArray(shmemA))
 