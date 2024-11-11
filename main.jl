using Images, FileIO, DelimitedFiles
include("helper.jl")
include("kernels.jl")

function doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    time_taken = 0
    for j in 1:octaves
        # println("performing octave $j")
        for i in 1:layers
            # assuming height <= 1024
            threads_column = 1024 #32 * 32
            threads_row = (16, 1024 ÷ 16)
            while threads_row[2] - 2 * aprons[i] <= 0 && threads_row[1] > 4
                threads_row = (threads_row[1] ÷ 2, threads_row[2] * 2)
            end
            # println("threads_column: $threads_column, threads_row: $threads_row")
            # println(cld(height, prod(threads_column)))
            if cld(height, prod(threads_column)) >= 1
                blocks_column = makeThisNearlySquare((cld(height - 2 * aprons[i], threads_column - 2 * aprons[i]), width))
                # println("org_blocks_column: $((cld(height-2*aprons[i], threads_column-2*aprons[i]), width))")
                # println("blocks_column: $blocks_column")
                blocks_row = makeThisNearlySquare((cld(height - 2 * aprons[i], threads_row[1]) * cld(width - 2 * aprons[i], threads_row[2] - 2 * aprons[i]) + cld(height - 2 * aprons[i], threads_row[1]) / 2 * cld(imgWidth - 2 * aprons[i], threads_row[2] - 2 * aprons[i]), 1))
                # println("blocks_row: $blocks_row")  
                shmem_column = threads_column * sizeof(Float32)
                shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)


                time_taken += CUDA.@elapsed buffer .= 0
                time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(img_gpu, conv_gpus[i], buffer, Int32(width), Int16(height), Int8(aprons[i]))
                # kernel = @cuda name = "col" launch = false col_kernel_strips(img_gpu, conv_gpus[1], buffer, Int32(width), Int16(height), Int8(aprons[i]))
                # println(launch_configuration(kernel.fun))
                # kernel = @cuda name = "row" launch = false row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
                # println(launch_configuration(kernel.fun))
                # println("h-2ap:$(Int16(height - 2 * aprons[i])), h: $(Int16(height)), w: $(Int32(width)), imW: $(Int16(imgWidth)), apron: $(Int8(aprons[i]))")
                time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
                save("assets/gaussian_j_o$(j)_l$(i)_r.png", colorview(Gray, collect(buffer)))
                save("assets/gaussian_j_o$(j)_l$(i)_rc.png", colorview(Gray, collect(out_gpus[j][i])))
            end
        end
        time_taken += CUDA.@elapsed buffer = CUDA.zeros(Float32, cld(height, 2), cld(width, 2))
        time_taken += CUDA.@elapsed img_gpu = CUDA.zeros(Float32, cld(height, 2), cld(width, 2))
        time_taken += CUDA.@elapsed @cuda threads = 1024 blocks = makeThisNearlySquare((cld(height * width ÷ 4, 1024), 1)) shmem = 1024 * sizeof(Float32) resample_kernel(out_gpus[j][3], img_gpu)
        for i in 1:(layers-1)
            time_taken += CUDA.@elapsed out_gpus[j][i] = out_gpus[j][i+1] .- out_gpus[j][i]
            time_taken += CUDA.@elapsed out_gpus[j][i] = out_gpus[j][i] .* (out_gpus[j][i] .> 0.0)
        end
        height = height ÷ 2
        width = width ÷ 2
    end
    return time_taken
end

function getGPUElements(img, height, width, layers, octaves, sigma0=1.6, k=-1)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    conv_gpus = []
    out_gpus = []
    for octave in 1:octaves
        for layer in 1:layers
            if layer == 1
                push!(out_gpus, [CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1))))])
            else
                push!(out_gpus[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
            end
            if octave == 1
                sigma = sigma0 * k^(layer - 1)
                apron = ceil(Int, 3 * sigma)
                push!(conv_gpus, CuArray(getGaussianKernel(2 * apron + 1, sigma)))
            end
        end
    end
    return CuArray(img), out_gpus, conv_gpus, CUDA.zeros(Float32, height, width), CUDA.zeros(Float32, height ÷ 2, width ÷ 2)
end

function findBlobs(img_gpu, out_gpus, conv_gpus, buffer, resample_buffer, height, width, imgWidth, octaves, scales=5, sigma0=1.6, k=-1)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    time_taken = 0
    for octave in 1:octaves
        for layer in 1:scales
            threads_column = 1024 #32 * 32
            threads_row = (16, 1024 ÷ 16)
            # sigma = sigma0 * k^(layer-1)
            # apron = ceil(3*sigma)
            sigma = sigma0 * k^(layer - 1)
            apron = ceil(Int, 3 * sigma)
            while threads_row[2] - 2 * apron <= 0 && threads_row[1] > 4
                threads_row = (threads_row[1] ÷ 2, threads_row[2] * 2)
            end
            if cld(height, prod(threads_column)) >= 1
                blocks_column = makeThisNearlySquare((cld(height - 2 * apron, threads_column - 2 * apron), width))
                blocks_row = makeThisNearlySquare((cld(height - 2 * apron, threads_row[1]) * cld(width - 2 * apron, threads_row[2] - 2 * apron) + cld(height - 2 * apron, threads_row[1]) / 2 * cld(imgWidth - 2 * apron, threads_row[2] - 2 * apron), 1))

                shmem_column = threads_column * sizeof(Float32)
                shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)

                time_taken += CUDA.@elapsed buffer .= 0
                if layer == 1
                    if octave == 1
                        # take the image as input
                        time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(img_gpu, conv_gpus[layer], buffer, Int32(width), Int16(height), Int8(apron))
                    else
                        # take the previous octave's third last output, resample it and take it as input
                        time_taken += CUDA.@elapsed @cuda threads = 1024 blocks = makeThisNearlySquare((cld(height * width ÷ 4, 1024), 1)) shmem = 1024 * sizeof(Float32) resample_kernel(out_gpus[octave-1][scales-2], resample_buffer)
                        time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(resample_buffer, conv_gpus[layer], buffer, Int32(width), Int16(height), Int8(apron))
                    end
                else
                    # take the previous layer's output as input
                    time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(out_gpus[octave][layer-1], conv_gpus[layer], buffer, Int32(width), Int16(height), Int8(apron))
                end
                time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel(buffer, conv_gpus[layer], out_gpus[octave][layer], Int16(height - 2 * apron), Int16(height), Int32(width), Int16(imgWidth), Int8(apron))
            end
        end
        height = height ÷ 2
        width = width ÷ 2
    end
    return time_taken
end

# let
#     println("Here we go!")
#     nImages = 128
#     img = []
#     imgWidth = 0
#     time_taken = 0
#     # load the images
#     for i in 1:nImages
#         # img_temp = Float32.(Gray.(FileIO.load("assets/images/DJI_20240328_234918_14_null_beauty.mp4_frame_$(i+900).png")))

#         img_temp = Float32.(Gray.(FileIO.load("assets/images/DJI_20240329_154936_17_null_beauty.mp4_frame_$(i+900).png")))
#         if i == 1
#             img = img_temp
#             imgWidth = size(img, 2)
#         else
#             img = cat(img, img_temp, dims=2)
#         end
#     end

#     height, width = size(img)
#     println(size(img))

#     octaves = 3
#     layers = 5

#     sigma0 = 1.6
#     k = 2^(1 / (layers - 3))

#     img_gpu, out_gpu, convolution_gpu, buffer, resample_buffer = getGPUElements(img, height, width, layers, octaves, sigma0, k)
#     iterations = 10
#     for i in 1:iterations
#         time_taken += findBlobs(img_gpu, out_gpu, convolution_gpu, buffer, resample_buffer, height, width, imgWidth, octaves, layers, sigma0, k)
#         # findBlobs(img_gpu, height, width, imgWidth, octaves, layers, sigma0, k)
#     end
#     println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time")
# end


let
    println("Here we go!")
    nImages = 1
    img = []
    imgWidth = 0
    time_taken = 0
    # load the images
    for i in 1:nImages
        # img_temp = Float32.(Gray.(FileIO.load("assets/images/DJI_20240328_234918_14_null_beauty.mp4_frame_$(i+900).png")))

        img_temp = Float32.(Gray.(FileIO.load("assets/images/DJI_20240329_154936_17_null_beauty.mp4_frame_$(i+900).png")))
        if i == 1
            img = img_temp
            imgWidth = size(img, 2)
        else
            img = cat(img, img_temp, dims=2)
        end
    end

    height, width = size(img)
    println(size(img))
    save("assets/gaussian_new_0.png", colorview(Gray, collect(img)))

    schemaBase = Dict(:name => "gaussian1D", :epsilon => 0.1725)

    layers = 5
    octaves = 3
    schemas = getSchemas(schemaBase, 1.6, sqrt(2), layers)
    aprons = getApron(schemas)

    # create GPU elements
    img_gpu = CuArray(img)
    # buffer_resample = CUDA.zeros(Float32, height ÷ 2, width ÷ 2)
    # @cuda threads = 1024 blocks = makeThisNearlySquare((cld(height * width ÷ 4, 1024), 1)) shmem=1024*sizeof(Float32) resample_kernel(img_gpu, buffer_resample)
    # save("assets/resample.png", colorview(Gray, Array(buffer_resample)))

    buffer = CUDA.zeros(Float32, height, width)
    conv_gpus = []
    out_gpus = []
    for j in 1:octaves
        out_gpus_octave = []
        for i in 1:layers
            # out_gpu = CUDA.zeros(Float32, height - 2 * aprons[i], width - 2 * nImages * aprons[i])
            out_gpu = CUDA.zeros(Float32, cld(height, (2^(j - 1))), cld(width, (2^(j - 1))))
            push!(out_gpus_octave, out_gpu)
            if j == 1
                # kernel = reshape(getGaussianKernel(2 * aprons[i] + 1, schemas[i][:sigma]), 2 * aprons[i] + 1)
                # push!(conv_gpus, CuArray(kernel))
                kernel = getGaussianKernel(2 * aprons[i] + 1, schemas[i][:sigma])
                push!(conv_gpus, CuArray(kernel))
            end
        end
        push!(out_gpus, out_gpus_octave)
    end


    # i = 1
    # warmup_inp = CUDA.rand(Float32, 1080, 1920)
    # warmupout_gpus = []
    # for i in 1:layers
    #     warmupout_gpu = CUDA.zeros(Float32, 1080 - 2 * aprons[i], 1920 - 2 * aprons[i])
    #     push!(warmupout_gpus, warmupout_gpu)
    # end
    # doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    # println("Warmup done!")
    iterations = 1
    for i in 1:iterations
        time_taken += doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    end
    println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time")
    for j in 1:octaves
        for i in 1:(layers-1)
            # save("assets/gaussian_new_$([i])_1.png", colorview(Gray, collect(buffer)))
            # save("assets/gaussian_new_$([i]).png", colorview(Gray, collect(out_gpus[j][i])))
            save("assets/DoG_o$(j)l$(i).png", colorview(Gray, Array(out_gpus[j][i])))
            # out = collect(out_gpus[j][i])
            # save("assets/DoG_$([i]).txt", collect(out_gpus[j][i]))
            # writedlm("assets/DoG_$([i]).csv", Array(out_gpus[j][i]), ',')
        end
    end
    # println(aprons)
end