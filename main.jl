using Images, FileIO, DelimitedFiles, CSV, DataFrames, Format
include("helper.jl")
include("kernels.jl")
# include("kernels_inbounds.jl")

fmt = "%.8f"

function getGPUElements(img, height, width, layers, octaves, nImages, sigma0=1.6, k=-1)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    conv_gpu = []
    out_gpu = []
    DoG_gpu = []
    DoG_prev_gpu = []
    DoGo_gpu = []
    XY_gpu = nothing
    for octave in 1:octaves
        prev_mid_size = [height, width]

        for layer in 1:layers
            if layer == 1
                push!(out_gpu, [CUDA.zeros(Float32, cld(prev_mid_size[1], 2^(octave - 1)), cld(prev_mid_size[2], 2^(octave - 1)))])
                push!(DoG_gpu, [CUDA.zeros(Float32, cld(prev_mid_size[1], 2^(octave - 1)), cld(prev_mid_size[2], 2^(octave - 1)))])
                push!(DoG_prev_gpu, [CUDA.zeros(Float32, cld(prev_mid_size[1], 2^(octave - 1)), cld(prev_mid_size[2], 2^(octave - 1)))])
                push!(DoGo_gpu, [CUDA.zeros(Float32, cld(prev_mid_size[1], 2^(octave - 1)), cld(prev_mid_size[2], 2^(octave - 1)))])
                if octave == 1
                    XY_gpu = CUDA.zeros(Int32, 6, ceil(Integer, 0.0128 * height * width))
                end
            else
                push!(out_gpu[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
                if layer < layers
                    push!(DoG_gpu[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
                    push!(DoG_prev_gpu[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
                    push!(DoGo_gpu[octave], CUDA.zeros(Float32, cld(height, (2^(octave - 1))), cld(width, (2^(octave - 1)))))
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
    return CuArray(img), out_gpu, DoG_gpu, conv_gpu, CUDA.zeros(Float32, height, width), XY_gpu, DoGo_gpu, DoG_prev_gpu
end

function findBlobs(img_gpu, out_gpu, DoG_gpu, DoGo_gpu, conv_gpu, buffer, height, width, imgWidth, octaves, scales=5, sigma0=1.6, k=-1)
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
                    cld(height, threads_column - 2 * apron),
                    width,
                ))
                blocks_row = makeThisNearlySquare((
                    cld(height, threads_row[1]),
                    cld(width, threads_row[2] - 2 * apron),
                ))

                shmem_column = threads_column * sizeof(Float32)
                shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)

                time_taken += CUDA.@elapsed buffer .= 0
                if layer == 1
                    if octave == 1
                        time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips_3(
                            img_gpu,
                            conv_gpu[1],
                            buffer,
                            Int32(width),
                            Int16(height),
                            Int16(imgWidth),
                            Int8(apron),
                        )
                        CUDA.synchronize()
                        save("assets/gaussian_o$(octave)_l$(layer)_c.png", colorview(Gray, collect(buffer)))
                        time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel_3(
                            buffer,
                            conv_gpu[1],
                            out_gpu[1][1],
                            Int16(height),
                            Int32(width),
                            Int16(imgWidth),
                            Int8(apron),
                        )
                    else
                        # take the previous octave's third last output, resample it and take it as input
                        apron = 0
                        time_taken += CUDA.@elapsed @cuda threads = 1024 blocks = makeThisNearlySquare(((height * width) ÷ (1024), 1)) resample_kernel_2(out_gpu[octave-1][scales-2], out_gpu[octave][1], height * 2, width * 2)
                        accumulative_apron = resample_apron
                    end
                else
                    # take the previous layer's output as input
                    time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips_3(
                        out_gpu[octave][layer-1],
                        conv_gpu[layer-1],
                        buffer,
                        Int32(width),
                        Int16(height),
                        Int16(imgWidth / 2^(octave - 1)),
                        Int8(apron),
                    )
                    CUDA.synchronize()
                    save("assets/gaussian_o$(octave)_l$(layer)_c.png", colorview(Gray, collect(buffer)))
                    time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel_3(
                        buffer,
                        conv_gpu[layer-1],
                        out_gpu[octave][layer],
                        Int16(height),
                        Int32(width),
                        Int16(imgWidth / 2^(octave - 1)),
                        Int8(apron),
                    )
                end
                CUDA.synchronize()
                # save("assets/gaussian_o$(octave)_l$(layer)_rc.png", colorview(Gray, collect(out_gpu[octave][layer])))
                save("assets/gaussian_o$(octave)_l$(layer).png", colorview(Gray, collect(out_gpu[octave][layer])))
                if octave == 1 && layer == 1
                    save("assets/go/gaussian_o$(octave)_l$(layer).png", colorview(Gray, Array(out_gpu[octave][layer])))
                end
                if layer == scales
                    threads_blobs = (32, 1024 ÷ 32)
                    blocks_blobs = (cld(height, threads_blobs[1] - 2), cld(width, threads_blobs[2] - 2))
                    shmem_blobs = threads_blobs[1] * threads_blobs[2] * sizeof(Float32) * 4
                    println("Launching blobs with threads: $threads_blobs, blocks: $blocks_blobs, height: $height, width: $width, imgWidth: $imgWidth, accumulative_apron: $accumulative_apron, apron: $apron, k: $k, octave: $octave")
                    time_taken += CUDA.@elapsed @cuda threads = threads_blobs blocks = blocks_blobs shmem = shmem_blobs maxregs = 32 blobs_2(
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
                        k - 1
                        # DoGo_gpu[octave][4],
                        # DoGo_gpu[octave][3],
                        # DoGo_gpu[octave][2],
                        # DoGo_gpu[octave][1]
                    )
                    CUDA.synchronize()
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

function extractBlobXYs(out_gpu, DoG_gpu, XY_gpu, octaves, scales, height, width, imgWidth, iter, sigma0=1.6, k=-1, bins=32)
    if k == -1
        k = 2^(1 / (scales - 3))
    end
    time_taken = 0
    count_gpu = CuArray{UInt64}([0])
    counts = Int[]
    radii = Int[]
    height_local = height
    width_local = width
    for octave in 1:octaves
        for layer in 1:scales-3
            threads = 1024
            blocks = cld(height_local * width_local, threads)
            println("O$(octave)L$(layer) max of DoG: $(maximum(DoG_gpu[octave][layer]))")
            time_taken += CUDA.@elapsed @cuda threads = threads blocks = blocks shmem = (sizeof(Int64) * (1 + 32 * 2)) stream_compact(
                DoG_gpu[octave][layer],
                XY_gpu,
                height_local,
                width_local,
                Int16(imgWidth / 2^(octave - 1)),
                count_gpu,
                octave,
                layer + 1,
            )
            CUDA.synchronize()
            push!(counts, Integer(collect(count_gpu)[1]))
            if layer > 1 || octave > 1
                println("O$(octave)L$(layer) count: $(counts[end]-counts[end-1])")
            else
                println("O$(octave)L$(layer) count: $(counts[end])")
            end
            sigma = sigma0 * k^((octave - 1) * 2 + layer)
            push!(radii, ceil(Int, 1.5 * sigma0 * k^(layer) + 1))
        end
        height_local = height_local ÷ 2
        width_local = width_local ÷ 2
    end
    count = collect(count_gpu)[1]
    println("counts: $counts, $count")
    counts_gpu = CuArray(counts)
    radii_gpu = CuArray(radii)
    threads = ((maximum(radii) * 2 + 1) + 2 * 1, (maximum(radii) * 2 + 1) + 2 * 1)
    orientation_gpu = CUDA.zeros(Float32, bins, count)

    # lets store the gradient orientations at each pixel in a 2D array
    gradient_orientations = []
    aligned_go = []
    for o in 1:octaves
        push!(gradient_orientations, CUDA.zeros(Float32, 4, cld(height, 2^(o - 1)), cld(width, 2^(o - 1))))
        push!(aligned_go, CUDA.zeros(Float32, 4, cld(height, 2^(o - 1)), cld(width, 2^(o - 1))))
        # push!(gradient_orientations, CUDA.zeros(Float32, 3, height, width))
    end

    check_count = CUDA.zeros(UInt64, 1)

    println("Size of XY_gpu: $(size(XY_gpu))")

    time_taken += CUDA.@elapsed @cuda threads = threads blocks = count shmem = (sizeof(Float32) * (((maximum(radii) * 2 + 1) + 2 * 1)^2 + bins)) maxregs = 32 find_orientations(
        out_gpu[3][2],
        out_gpu[2][2],
        out_gpu[1][2],
        XY_gpu,
        orientation_gpu,
        height,
        width,
        counts_gpu,
        radii_gpu,
        bins,
        gradient_orientations[3],
        gradient_orientations[2],
        gradient_orientations[1],
        aligned_go[3],
        aligned_go[2],
        aligned_go[1],
        check_count
    )
    CUDA.synchronize()
    println("check_count: $(collect(check_count)[1])")
    # save orientation_gpu as csv
    CSV.write("assets/orientations_i.csv", DataFrame(collect(transpose(collect(orientation_gpu))), :auto))
    # save gradient_orientations as images
    for o in 1:octaves
        save("assets/go/gradient_orientations_o$(o).png", colorview(RGBA, Array(gradient_orientations[o])))
        save("assets/go/aligned_go_o$(o).png", colorview(RGBA, map(clamp01nan, Array(aligned_go[o]))))
    end
    filtered_XY_gpu = CUDA.zeros(Float32, bins + 6, count)
    filtered_count_gpu = CuArray{UInt64}([0])
    println("count: $count, filtered count: $(collect(filtered_count_gpu)[1]), bins: $bins")

    time_taken += CUDA.@elapsed @cuda threads = (32, 512 ÷ 32) blocks = cld(count, 512 ÷ 32) shmem = (sizeof(Float32) * 32 * 32 + sizeof(UInt64)) filter_blobs(
        XY_gpu, orientation_gpu, filtered_XY_gpu, count, filtered_count_gpu, bins, 0.534)
    CUDA.synchronize()

    println("filtered count: $(collect(filtered_count_gpu)[1])")
    blank_slate = CUDA.zeros(Float32, 4, height, width)
    @cuda threads = 1 blocks = collect(filtered_count_gpu)[1] plot_blobs_f(filtered_XY_gpu, blank_slate, height, width, size(filtered_XY_gpu, 1))
    CUDA.synchronize()
    save("assets/filtered_blobs_$(iter).png", colorview(RGBA, Array(blank_slate)))
    if iter == 1
        save("assets/go/filtered_blobs.png", colorview(RGBA, Array(blank_slate)))
    end
    blank_slate .= 0
    @cuda threads = 1 blocks = count plot_blobs_uf(XY_gpu, blank_slate, height, width, size(XY_gpu, 1), 0)
    CUDA.synchronize()
    save("assets/blobs_$(iter).png", colorview(RGBA, Array(blank_slate)))
    if iter == 1
        save("assets/go/blobs.png", colorview(RGBA, Array(blank_slate)))
    end
    # CSV.write("assets/filtered_XY_i.csv", DataFrame(collect(transpose(collect(filtered_XY_gpu))), :auto))
    return time_taken, count, collect(orientation_gpu), collect(filtered_XY_gpu)
end

let
    println("Here we go!")
    nImages = 1
    img = []
    imgWidth = 0
    time_taken = 0
    # load the images
    for i in 1:nImages
        # img_temp = Float32.(Gray.(FileIO.load("assets/images/DJI_20240329_154936_17_null_beauty.mp4_frame_$(i+900).png")))
        img_temp = Float32.(Gray.(FileIO.load("assets/images/20241203_000635.mp4_frame_$(i).png")))
        # img_temp = Float32.(Gray.(FileIO.load("assets/images/$(i).png")))
        # img_temp = Float32.(Gray.(FileIO.load("assets/images/test blobs.png")))
        if i == 1
            img = img_temp
            imgWidth = size(img, 2)
        else
            img = cat(img, img_temp, dims=2)
        end
    end

    height, width = size(img)
    println(size(img))

    octaves = 3 # hard coded in the blobs kernel
    layers = 5 # hard coded in the extractBlobXYs function and orientation kernel

    sigma0 = 1.6
    k = 2^(1 / (layers - 3))

    img_gpu, out_gpu, DoG_gpu, convolution_gpu, buffer, XY_gpu, DoGo_gpu, DoG_prev_gpu = getGPUElements(img, height, width, layers, octaves, nImages, sigma0, k)
    println("Got the GPU elements...")
    iterations = 1
    count = nothing
    orientations = nothing
    blobs = nothing
    for i in 1:iterations
        for o in 1:octaves
            for l in 1:layers
                out_gpu[o][l] .= 0
                buffer .= 0
                if l < layers
                    DoG_gpu[o][l] .= 0
                    DoGo_gpu[o][l] .= 0
                end
            end
        end
        XY_gpu .= 0
        time_taken += findBlobs(img_gpu, out_gpu, DoG_gpu, DoGo_gpu, convolution_gpu, buffer, height, width, imgWidth, octaves, layers, sigma0, k)
        # for oct in 1:octaves
        #     for l in 1:layers
        #         save("iterations/gaussian_o$(oct)_l$(l)_i$(i).png", colorview(Gray, collect(out_gpu[oct][l])))
        #         if l < layers
        #             if l <= layers - 3
        #                 max = CUDA.maximum(DoG_gpu[oct][l])
        #                 if max .!= 0
        #                     # DoG = Array(DoG_gpu[oct][l]) ./ max
        #                     DoG = Array(DoG_gpu[oct][l])
        #                 end
        #                 save("iterations/DoG_nov24_o$(oct)l$(l)_i$(i).png", colorview(Gray, DoG))
        #                 save("iterations/DoG_nov24_o$(oct)l$(l)_i$(i)_binary.png", colorview(Gray, DoG.>0))
        #                 if i > 1
        #                     # save the absolute differences in the DoG[i-1] and DoG[i]
        #                     save("iterations/DoG_diff_nov24_o$(oct)l$(l)_i$(i).png", colorview(Gray, Array((DoG_gpu[oct][l].>0) .⊻ (DoG_prev_gpu[oct][l].>0))))
        #                 end
        #             end
        #             max = CUDA.maximum(DoGo_gpu[oct][l])
        #             min = CUDA.minimum(DoGo_gpu[oct][l])
        #             if max - min .!= 0
        #                 DoGo = (Array(DoGo_gpu[oct][l]) .- min) ./ (max - min)
        #                 # DoGo = (Array(DoGo_gpu[oct][l]) .- min)
        #             end
        #             save("iterations/DoG_raw_nov24_o$(oct)l$(l)_i$(i).png", colorview(Gray, DoGo))
        #         end
        #     end
        # end
        # keep bins < 32 so that one warp handles one point

        time_taken_here, count, orientations, blobs = extractBlobXYs(out_gpu, DoG_gpu, XY_gpu, octaves, layers, height, width, imgWidth, i)
        time_taken += time_taken_here
        # DoG_prev_gpu = copy(DoG_gpu)
    end
    println("Got the blobs...")
    # for j in 1:octaves
    #     for i in 1:2
    #         max = CUDA.maximum(DoG_gpu[j][i])
    #         if max .!= 0
    #             DoG_gpu[j][i] = DoG_gpu[j][i] ./ max
    #             # DoG_gpu[j][i] = DoG_gpu[j][i]
    #         end
    #         save("assets/DoG_nov24_o$(j)l$(i).png", colorview(Gray, Array(DoG_gpu[j][i])))

    #         max = CUDA.maximum(DoGo_gpu[j][i])
    #         min = CUDA.minimum(DoGo_gpu[j][i])
    #         if max - min .!= 0
    #             DoGo_gpu[j][i] = (DoGo_gpu[j][i] .- min) ./ (max - min)
    #             # DoGo_gpu[j][i] = (DoGo_gpu[j][i] .- min)
    #         end
    #         save("assets/DoG_raw_nov24_o$(j)l$(i).png", colorview(Gray, Array(DoGo_gpu[j][i])))
    #     end
    # end
    println("count: $count, size of XY_gpu: $(size(XY_gpu))")
    XY = collect(XY_gpu[:, 1:count])
    println("Total potential blobs: $count, utilization: $(round(count / size(XY_gpu, 2)*100, digits=2))%")
    CSV.write("assets/blobs.csv", DataFrame(collect(transpose(XY)), :auto))
    df = DataFrame(collect(transpose(collect(orientations))), :auto)
    CSV.write("assets/orientations.csv", df,
        transform=(col, val) -> typeof(val) <: AbstractFloat ? cfmt(fmt, val) : val)
    # CSV.write("assets/filtered_blobs.csv", DataFrame(collect(transpose(blobs)), :auto))
    df = DataFrame(collect(transpose(collect(blobs))), :auto)
    CSV.write("assets/filtered_blobs.csv", df,
        transform=(col, val) -> typeof(val) <: AbstractFloat ? cfmt(fmt, val) : val)

    println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time. Total time taken: $(round(time_taken/iterations, digits=5))s per iteration.")
end