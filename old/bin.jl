using CUDA

nParts = 50
width = 1920
height = 1080
# array of X, Y coordinates for each point
pos = CUDA.rand(Float32, 2, nParts) .* CuArray([width, height])



function bin(pos, width, height)
    # binning
    bins = zeros(Int32, 32, 32)
    bin_sizes = CuArray([width / 32, height / 32])
    for i in axes(pos, 2)
        x, y = Int32.(pos[:, i] .÷ bin_sizes)
        # println(x," ", y)
        # append this particle to the bin
        # bins[x+1+(y)*32] += 1
        bins[x+1, y+1] += 1
    end
    return bins
end

bin(pos, width, height)

let
    iterations = 100
    time_sum = 0
    bins = []
    for i in 1:iterations
        start_t = time()
        bins = bin(pos, width, height)
        end_t = time()
        time_sum += end_t - start_t
    end
    println("Average time: ", time_sum / iterations)

    println(bins)
end