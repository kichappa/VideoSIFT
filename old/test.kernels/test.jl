include("kernels.jl")
using FileIO, Images, Printf, CUDA
using JLD2, UnPack

function normalizeArray(arr)
    return (arr .- minimum(arr)) ./ (maximum(arr) - minimum(arr) + eps(Float32))
end
function format_number(x)
    if abs(x) == 0 || (abs(x) >= 0.0001 && abs(x) < 100000 && (round(x, digits=5) - x )<= 0.00001)
        return round(x, digits=5)
    else
        return @sprintf("%.4e", x)
    end
end

# get all png files in root directory
# png_files = readdir(".") |> filter(f -> endswith(f, ".png"))
# filename(x) = "../assets/testing/gaussian_o1_l$x.png"
# imgs = [FileIO.load(filename(x)) for x in 1:5]
@unpack imgs = jldopen("../assets/testing/gaussian_o1.jld2")

h, w = size(imgs[1])
imgWidth = Integer(w/2)
scales = length(imgs)
k = 2.0f0^(1/(scales - 3))

imgs_cuda = []
for i in eachindex(imgs)
    imgs[i] = Gray.(imgs[i])
    push!(imgs_cuda, CuArray{Float32}(imgs[i]))
end

out1 = similar(imgs_cuda[1])
out2 = similar(imgs_cuda[1])

DoG1 = similar(imgs_cuda[1])
DoG2 = similar(imgs_cuda[1])
DoG3 = similar(imgs_cuda[1])
DoG4 = similar(imgs_cuda[1])

out1 .= 0.0f0
out2 .= 0.0f0
DoG1 .= 0.0f0
DoG2 .= 0.0f0
DoG3 .= 0.0f0
DoG4 .= 0.0f0
CUDA.synchronize()

imgs_cuda = CuArray(reverse([cudaconvert(x) for x in imgs_cuda]))  # reverse order to match original code

threads_blobs = (32, 1024 ÷ 32)
blocks_blobs = (cld(h, threads_blobs[1] - 2), cld(w, threads_blobs[2] - 2))
shmem_blobs = threads_blobs[1] * threads_blobs[2] * sizeof(Float32) * 4
@cuda threads=threads_blobs blocks=blocks_blobs shmem=shmem_blobs maxregs=32 blobs_2_rewrite(imgs_cuda, out2, out1, h, w, imgWidth, k-1, DoG4, DoG3, DoG2, DoG1)
CUDA.synchronize()
println("k=$(format_number(k-1))")
println("h, w, imw = $(h), $(w), $(imgWidth)")
let
    # normDoG1 = normalizeArray(DoG1)
    # normDoG2 = normalizeArray(DoG2)
    # normDoG3 = normalizeArray(DoG3)
    # normDoG4 = normalizeArray(DoG4)

    # normout2 = normalizeArray(out2)
    # normout1 = normalizeArray(out1)

    # println("Max/Min DoG1: ", maximum(normDoG1), "/", minimum(normDoG1))
    # println("Max/Min DoG2: ", maximum(normDoG2), "/", minimum(normDoG2))
    # println("Max/Min DoG3: ", maximum(normDoG3), "/", minimum(normDoG3))
    # println("Max/Min DoG4: ", maximum(normDoG4), "/", minimum(normDoG4))
    # println("Max/Min out2: ", maximum(normout2), "/", minimum(normout2))
    # println("Max/Min out1: ", maximum(normout1), "/", minimum(normout1))
end

# save 
FileIO.save("DoGs/DoG1.png", collect(Gray.(normalizeArray(DoG1))))
FileIO.save("DoGs/DoG2.png", collect(Gray.(normalizeArray(DoG2))))
FileIO.save("DoGs/DoG3.png", collect(Gray.(normalizeArray(DoG3))))
FileIO.save("DoGs/DoG4.png", collect(Gray.(normalizeArray(DoG4))))

FileIO.save("DoGs/out2.png", collect(Gray.(normalizeArray(out2))))
FileIO.save("DoGs/out1.png", collect(Gray.(normalizeArray(out1))))

@unpack DoGo, minmaxo = jldopen("../assets/testing/DoG_o1.jld2")
for i in 1:4
    img_cuda = CuArray(DoGo[i])
    compared = normalizeArray(img_cuda) .== normalizeArray(getfield(Main, Symbol("DoG$i")))
    compared = .!compared
    println("DoG $i differences: ", sum(compared))
    FileIO.save("DoGs/DoG_compare_$i.png", collect(Gray.(normalizeArray(Float32.(compared)))))
end

for i in 1:2
    img_cuda = CuArray(minmaxo[i])
    compared = normalizeArray(img_cuda) .== normalizeArray(getfield(Main, Symbol("out$i")))
    compared = .!compared
    println("out $i differences: ", sum(compared))
    FileIO.save("DoGs/out_compare_$i.png", collect(Gray.(normalizeArray(Float32.(compared)))))
end

# garbage collection
GC.gc()