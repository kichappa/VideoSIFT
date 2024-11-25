using Images, FileIO, DelimitedFiles, CSV, DataFrames
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
                blocks_row = makeThisNearlySquare((
                    cld(height - 2 * aprons[i], threads_row[1]) * cld(width - 2 * aprons[i], threads_row[2] - 2 * aprons[i]) + cld(height - 2 * aprons[i], threads_row[1]) / 2 * cld(imgWidth - 2 * aprons[i], threads_row[2] - 2 * aprons[i]),
                    1,
                ))
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
                time_taken +=
                    CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
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

function getGPUElements(img, height, width, layers, octaves, nImages, sigma0=1.6, k=-1)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    conv_gpu = []
    out_gpu = []
    DoG_gpu = []
    XY_gpu = nothing
    for octave in 1:octaves
        prev_mid_size = [height, width]

        for layer in 1:layers
            if layer == 1
                push!(out_gpu, [CUDA.zeros(Float32, cld(prev_mid_size[1], 2^(octave - 1)), cld(prev_mid_size[2], 2^(octave - 1)))])
                push!(DoG_gpu, [CUDA.zeros(Float32, cld(prev_mid_size[1], 2^(octave - 1)), cld(prev_mid_size[2], 2^(octave - 1)))])
                if octave ==1
                    XY_gpu=CUDA.zeros(Int32, 6, ceil(Integer, 0.0025 * height * width))
                end
            else
                push!(out_gpu[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
                if layer < layers
                    push!(DoG_gpu[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
                end
            end
            # create the convolution kernel
            if octave == 1 && layer < layers
                sigma = sigma0 * k^(layer - 1)
                apron = ceil(Int, 3 * sigma / 2) * 2
                push!(conv_gpu, CuArray(getGaussianKernel(2 * apron + 1, sigma)))
            end
        end
    end
    return CuArray(img), out_gpu, DoG_gpu, conv_gpu, CUDA.zeros(Float32, height, width), XY_gpu
end

function findBlobs(img_gpu, out_gpu, DoG_gpu, conv_gpu, buffer, height, width, imgWidth, octaves, scales=5, sigma0=1.6, k=-1)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    time_taken = 0
    accumulative_apron::Int8 = 0
    resample_apron::Int8 = 0
    nImages = width ÷ imgWidth
    for octave in 1:octaves
        for layer in 1:scales
            threads_column = 1024 #32 * 32
            threads_row = (32, 1024 ÷ 32)

            if layer == 1
                sigma = sigma0 * k^(layer - 1)
                apron::Int8 = ceil(Int, 3 * sigma / 2) * 2
            else
                sigma = sigma0 * k^(layer - 2)
                apron = ceil(Int, 3 * sigma / 2) * 2
            end

            while threads_row[2] - 2 * apron <= 0 && threads_row[1] > 4
                threads_row = (threads_row[1] ÷ 2, threads_row[2] * 2)
            end

            if cld(height, prod(threads_column)) >= 1
                blocks_column = makeThisNearlySquare((
                    cld(height - 2 * (apron + accumulative_apron), threads_column - 2 * apron),
                    width - 2 * accumulative_apron * nImages,
                ))
                blocks_row = makeThisNearlySquare((
                    cld(height - 2 * (apron + accumulative_apron), threads_row[1]),
                    cld(width - 2 * (apron + accumulative_apron), threads_row[2] - 2 * apron),
                ))

                shmem_column = threads_column * sizeof(Float32)
                shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)

                time_taken += CUDA.@elapsed buffer .= 0
                if layer == 1
                    if octave == 1
                        # take the image as input, blur with conv_gpu[1]
                        time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips_2(
                            img_gpu,
                            conv_gpu[1],
                            buffer,
                            Int32(width),
                            Int16(height),
                            Int16(imgWidth),
                            Int8(accumulative_apron),
                            Int8(apron),
                        )
                        CUDA.synchronize()
                        save("assets/gaussian_o$(octave)_l$(layer)_c.png", colorview(Gray, collect(buffer)))
                        time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel_2(
                            buffer,
                            conv_gpu[1],
                            out_gpu[1][1],
                            Int16(height),
                            Int32(width),
                            Int16(imgWidth),
                            Int8(accumulative_apron),
                            Int8(apron),
                        )
                    else
                        # take the previous octave's third last output, resample it and take it as input
                        apron = 0
                        time_taken += CUDA.@elapsed @cuda threads = 1024 blocks = makeThisNearlySquare(((height * width) ÷ (1024), 1)) resample_kernel_2(out_gpu[octave-1][scales-2], out_gpu[octave][1], height * 2, width * 2)
                        accumulative_apron = resample_apron
                        # CUDA.synchronize()
                        # save the resampled image
                        # save("assets/resampled_g_o$(octave-1)_l$(scales-2).png", colorview(Gray, collect(out_gpu[octave][1])))
                    end
                else
                    # take the previous layer's output as input
                    time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips_2(
                        out_gpu[octave][layer-1],
                        conv_gpu[layer-1],
                        buffer,
                        Int32(width),
                        Int16(height),
                        Int16(imgWidth / 2^(octave - 1)),
                        Int8(accumulative_apron),
                        Int8(apron),
                    )
                    CUDA.synchronize()
                    save("assets/gaussian_o$(octave)_l$(layer)_c.png", colorview(Gray, collect(buffer)))
                    time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel_2(
                        buffer,
                        conv_gpu[layer-1],
                        out_gpu[octave][layer],
                        Int16(height),
                        Int32(width),
                        Int16(imgWidth / 2^(octave - 1)),
                        Int8(accumulative_apron),
                        Int8(apron),
                    )
                end
                CUDA.synchronize()
                save("assets/gaussian_o$(octave)_l$(layer)_rc.png", colorview(Gray, collect(out_gpu[octave][layer])))
                if layer == 5
                    threads_blobs = (32, 1024 ÷ 32)
                    blocks_blobs = (cld(height - 2 * (accumulative_apron + 1), threads_blobs[1] - 2), cld(width - 2 * (accumulative_apron + 1) * nImages, threads_blobs[2] - 2))
                    shmem_blobs = threads_blobs[1] * threads_blobs[2] * sizeof(Float32) * 3

                    time_taken += CUDA.@elapsed @cuda threads = threads_blobs blocks = blocks_blobs shmem = shmem_blobs maxregs = 32 blobs(
                        out_gpu[octave][5],
                        out_gpu[octave][4],
                        out_gpu[octave][3],
                        out_gpu[octave][2],
                        out_gpu[octave][1],
                        DoG_gpu[octave][2],
                        DoG_gpu[octave][1],
                        height,
                        width,
                        imgWidth / 2^(octave - 1),
                        accumulative_apron,
                        accumulative_apron + apron,
                        k - 1,
                    )
                end
                accumulative_apron += apron
                if layer == scales - 2
                    resample_apron = accumulative_apron / 2
                end
            end
        end
        height = height ÷ 2
        width = width ÷ 2
        time_taken += CUDA.@elapsed buffer = CUDA.zeros(Float32, height, width)
    end
    return time_taken
end

function extractBlobXYs(DoG_gpu, XY_gpu, octaves, scales, height, width, imgWidth, sigma0=1.6, k=-1)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    time_taken = 0
    count_gpu = CuArray{UInt64}([0])
    counts = []
    radii = []
    px = 0
    for octave in 1:octaves
        for layer in 1:scales-3
            threads = 1024
            blocks = cld(height * width, threads)
            time_taken += CUDA.@elapsed @cuda threads = threads blocks = blocks shmem = (sizeof(Int64) * (1 + 32 * 2)) stream_compact(
                DoG_gpu[octave][layer],
                XY_gpu,
                height,
                width,
                Int16(imgWidth / 2^(octave - 1)),
                count_gpu,
                octave,
                layer+1
            )
            push!(counts, collect(count_gpu)[1])
            sigma = sigma0 * k^(layer - 1)
            push!(radii, ceil(Int, 1.5 * sigma0 * k^(layer) / 2) * 2)
            px = (2*radii[end] + 1) ^2 * counts[end]
            println("O$(octave)L$(layer) radius: $(radii[end]) px: $(px/counts[end])")
        end
        height = height ÷ 2
        width = width ÷ 2
    end
    count = collect(count_gpu)[1]
    threads = 1024
    blocks = cld(px, threads)
    return time_taken, collect(count_gpu)[1]
end

let
    println("Here we go!")
    nImages = 2
    img = []
    imgWidth = 0
    time_taken = 0
    # load the images
    for i in 1:nImages
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

    octaves = 3
    layers = 5

    sigma0 = 1.6
    k = 2^(1 / (layers - 3))

    img_gpu, out_gpu, DoG_gpu, convolution_gpu, buffer, XY_gpu = getGPUElements(img, height, width, layers, octaves, nImages, sigma0, k)
    println("Got the GPU elements...")
    iterations = 1
    count = nothing
    for i in 1:iterations
        time_taken += findBlobs(img_gpu, out_gpu, DoG_gpu, convolution_gpu, buffer, height, width, imgWidth, octaves, layers, sigma0, k)
        time_taken_here, count = extractBlobXYs(DoG_gpu, XY_gpu, octaves, layers, height, width, imgWidth)
        time_taken += time_taken_here
    end
    println("Got the blobs...")
    for j in 1:octaves
        for i in 1:2
            max = CUDA.maximum(DoG_gpu[j][i])
            if max .!= 0
                DoG_gpu = DoG_gpu ./ max
            end
            save("assets/DoG_nov24_o$(j)l$(i).png", colorview(Gray, Array(DoG_gpu[j][i])))
        end
    end

    XY = collect(XY_gpu[:, 1:count])
    println("Total potential blobs: $count, utilization: $(round(count / size(XY_gpu, 2)*100, digits=2))%")
    CSV.write("assets/blobs.csv", DataFrame(collect(transpose(XY)), :auto))

    println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time. Total time taken: $(round(time_taken/iterations, digits=5))s")
end


# let
#     println("Here we go!")
#     nImages = 1
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
#     save("assets/gaussian_new_0.png", colorview(Gray, collect(img)))

#     schemaBase = Dict(:name => "gaussian1D", :epsilon => 0.1725)

#     layers = 5
#     octaves = 3
#     schemas = getSchemas(schemaBase, 1.6, sqrt(2), layers)
#     aprons = getApron(schemas)

#     # create GPU elements
#     img_gpu = CuArray(img)
#     # buffer_resample = CUDA.zeros(Float32, height ÷ 2, width ÷ 2)
#     # @cuda threads = 1024 blocks = makeThisNearlySquare((cld(height * width ÷ 4, 1024), 1)) shmem=1024*sizeof(Float32) resample_kernel(img_gpu, buffer_resample)
#     # save("assets/resample.png", colorview(Gray, Array(buffer_resample)))

#     buffer = CUDA.zeros(Float32, height, width)
#     conv_gpus = []
#     out_gpus = []
#     for j in 1:octaves
#         out_gpus_octave = []
#         for i in 1:layers
#             # out_gpu = CUDA.zeros(Float32, height - 2 * aprons[i], width - 2 * nImages * aprons[i])
#             out_gpu = CUDA.zeros(Float32, cld(height, (2^(j - 1))), cld(width, (2^(j - 1))))
#             push!(out_gpus_octave, out_gpu)
#             if j == 1
#                 # kernel = reshape(getGaussianKernel(2 * aprons[i] + 1, schemas[i][:sigma]), 2 * aprons[i] + 1)
#                 # push!(conv_gpus, CuArray(kernel))
#                 kernel = getGaussianKernel(2 * aprons[i] + 1, schemas[i][:sigma])
#                 push!(conv_gpus, CuArray(kernel))
#             end
#         end
#         push!(out_gpus, out_gpus_octave)
#     end


#     # i = 1
#     # warmup_inp = CUDA.rand(Float32, 1080, 1920)
#     # warmupout_gpus = []
#     # for i in 1:layers
#     #     warmupout_gpu = CUDA.zeros(Float32, 1080 - 2 * aprons[i], 1920 - 2 * aprons[i])
#     #     push!(warmupout_gpus, warmupout_gpu)
#     # end
#     # doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
#     # println("Warmup done!")
#     iterations = 1
#     for i in 1:iterations
#         time_taken += doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
#     end
#     println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time")
#     for j in 1:octaves
#         for i in 1:(layers-1)
#             # save("assets/gaussian_new_$([i])_1.png", colorview(Gray, collect(buffer)))
#             # save("assets/gaussian_new_$([i]).png", colorview(Gray, collect(out_gpus[j][i])))
#             save("assets/DoG_o$(j)l$(i).png", colorview(Gray, Array(out_gpus[j][i])))
#             # out = collect(out_gpus[j][i])
#             # save("assets/DoG_$([i]).txt", collect(out_gpus[j][i]))
#             # writedlm("assets/DoG_$([i]).csv", Array(out_gpus[j][i]), ',')
#         end
#     end
#     # println(aprons)
# end