using Images, FileIO, DelimitedFiles, CSV, DataFrames, Format, VideoIO, Glob
include("helper.jl")
include("kernels.jl")
include("blobs.jl")
include("header.jl")
include("calibration_helper.jl")

function gaussianPyramid_mod(img_gpu, out_gpu, DoG_gpu, DoGo_gpu, conv_gpu, buffer, height, width, imgWidth, octaves; scales = 5, sigma0 = 1.6, k = -1)
	if k == -1
		k = 2^(1 / (scales - 3))
	end
	time_taken = zeros(Float32, 2)
	accumulative_apron::Int8 = 0
	resample_apron::Int8 = 0
	nImages = width ÷ imgWidth
	blobs_input_l = []
	for octave in 1:octaves
		blobs_input_l = []
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

				time_taken[1] += CUDA.@elapsed buffer .= 0
				if layer == 1
					if octave == 1
						time_taken[1] += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(
							img_gpu,
							conv_gpu[1],
							buffer,
							Int32(width),
							Int16(height),
							Int16(imgWidth),
							Int8(apron),
						)
						CUDA.synchronize()
						error_count = CuArray([0])
						time_taken[1] += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel(
							buffer,
							conv_gpu[1],
							out_gpu[1][1],
							Int16(height),
							Int32(width),
							Int16(imgWidth),
							Int8(apron),
							# error_count,
						)
					else
						# take the previous octave's third last output, resample it and take it as input
						apron = 0
						time_taken[1] += CUDA.@elapsed @cuda threads = 1024 blocks = makeThisNearlySquare(((height * width) ÷ (1024), 1)) resample_kernel_2(out_gpu[octave-1][scales-2], out_gpu[octave][1], height * 2, width * 2)
						accumulative_apron = resample_apron
					end
				else
					# take the previous layer's output as input
					time_taken[1] += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(
						out_gpu[octave][layer-1],
						conv_gpu[layer-1],
						buffer,
						Int32(width),
						Int16(height),
						Int16(imgWidth / 2^(octave - 1)),
						Int8(apron),
					)
					CUDA.synchronize()
					error_count = CuArray([0])
					time_taken[1] += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel(
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
				push!(blobs_input_l, out_gpu[octave][layer])
				if layer == scales
					threads_blobs = (32, 1024 ÷ 32)
					blocks_blobs = (cld(height, threads_blobs[1] - 2), cld(width, threads_blobs[2] - 2))
					shmem_blobs = threads_blobs[1] * threads_blobs[2] * sizeof(Float32) * 4
					DoG_gpu[octave][2] .= 0
					DoG_gpu[octave][1] .= 0
					CUDA.synchronize()
                    time_taken[2] += CUDA.@elapsed DoGo_gpu[octave][1] .= out_gpu[octave][2] .- out_gpu[octave][1]
                    time_taken[2] += CUDA.@elapsed DoGo_gpu[octave][2] .= out_gpu[octave][3] .- out_gpu[octave][2]
                    time_taken[2] += CUDA.@elapsed DoGo_gpu[octave][3] .= out_gpu[octave][4] .- out_gpu[octave][3]
                    time_taken[2] += CUDA.@elapsed DoGo_gpu[octave][4] .= out_gpu[octave][5] .- out_gpu[octave][4]
					CUDA.synchronize()
				end
				accumulative_apron += apron
				if layer == scales - 2
					resample_apron = accumulative_apron ÷ 2
				end
			end
		end
		height = height ÷ 2
		width = width ÷ 2
		time_taken[1] += CUDA.@elapsed buffer = CUDA.zeros(Float32, height, width)
	end
	return time_taken
end


let
    octaves = 5 # not hard coded in the blobs kernel
    layers = 5 # hard coded in the extractBlobXYs function and orientation kernel

    sigma0 = 1.6
    k = 2^(1 / (layers - 3))

    max = 8
    v = []
    path = "assets/videos/cam3"
    f = vcat(glob("*.mp4", path), glob("*.mov", path), glob("*.MP4", path), glob("*.MOV", path))[1]
    for i in 1:2^max
        try
            push!(v, Float32.(Gray.(getFrame(f, i))))
        catch 
            push!(v, Float32.(Gray.(getFrame(f, i-256))))
        end
    end
    for nImages in  2 .^ Vector(1:max)
        # push!(v, Float32.(Gray.(getFrame(f, 339))))
        # push!(v, Float32.(Gray.(getFrame(f, 202))))

        # load the images
        img = []
        imgWidth = 0
        for i in 1:nImages
            img_temp = v[i]	
            # println("image $i: $(size(img_temp))")
            if i == 1
                img = img_temp
                imgWidth = size(img, 2)
            else
                img = cat(img, img_temp, dims = 2)
            end
        end

        height, width = size(img)

        img_gpu, out_gpu, DoG_gpu, conv_gpu, buffer, XY_gpu, DoGo_gpu, DoG_prev_gpu = getGPUElements(img, height, width, layers, octaves, nImages, sigma0, k)

        runs = 100

        # # ============================= Single Gaussian =============================
        # t = 0.0
        # gaussian_index = 1

        # for run in 1:runs
        #     for i in 1:nImages
        #         threads_column = 1024 #32 * 32
        #         threads_row = (32, 1024 ÷ 32)
        #         apron::Int8 = ceil(Int, 3 * sigma0 / 2) * 2
        #         time_taken = 0.0
        #         while threads_row[2] - 2 * apron <= 0 && threads_row[1] > 4
        #             threads_row = (threads_row[1] ÷ 2, threads_row[2] * 2)
        #         end
                

        #         if cld(height, prod(threads_column)) >= 1
        #             blocks_column = makeThisNearlySquare((
        #                 cld(height, threads_column - 2 * apron),
        #                 width,
        #             ))
        #             blocks_row = makeThisNearlySquare((
        #                 cld(height, threads_row[1]),
        #                 cld(width, threads_row[2] - 2 * apron),
        #             ))

        #             shmem_column = threads_column * sizeof(Float32)
        #             shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)
        #             buffer .= 0
        #             time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column maxregs = 64 col_kernel_strips(
        #                 img_gpu,
        #                 conv_gpu[gaussian_index],
        #                 buffer,
        #                 Int32(width),
        #                 Int16(height),
        #                 Int16(imgWidth),
        #                 Int8(apron),
        #             )
        #             CUDA.synchronize()
        #             time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel(
        #                 buffer,
        #                 conv_gpu[gaussian_index],
        #                 out_gpu[1][1],
        #                 Int16(height),
        #                 Int32(width),
        #                 Int16(imgWidth),
        #                 Int8(apron),
        #             )
        #             if run > 1
        #                 t += time_taken
        #             end
        #         end
        #     end
        # end

        # println("Gaussian time: $(t/(nImages * (runs - 1))) seconds per image @ $nImages images at once")
        
        # ============================= Gaussian Pyramid =============================
        
        t = zeros(Float32, 2)
        for run in 1:runs
            time_taken = gaussianPyramid_mod(img_gpu, out_gpu, DoG_gpu, DoGo_gpu, conv_gpu, buffer, height, width, imgWidth, octaves; scales = layers, sigma0 = sigma0)
            if run > 1
                t += time_taken
            end
        end
        println("Pyramid time: [$(t[1]/(nImages * (runs - 1))), $(t[2]/(nImages * (runs - 1)))] seconds per image @ $nImages images at once")
    
        ## ============================= Blobs =============================
        # t = 0
        # for run in 1:runs
		# 	time_taken = 0.0
        #     time_taken, count, orientations, blobs, XY_gpu = getBlobs(img, height, width, imgWidth, octaves, layers, nImages, sigma0, k, 1, time_taken)
        #     if run > 1
        #         t += time_taken
        #     end
        # end
        # println("Blobs time: $(t/(nImages * (runs - 1))) seconds per image @ $nImages images at once")
    end
end