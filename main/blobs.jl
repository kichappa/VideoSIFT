
function getGPUElements(img, height, width, layers, octaves, nImages, sigma0 = 1.6, k = -1)
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
					XY_gpu = CuArray([potential_blob() for i in 1:ceil(Integer, 0.0128 * height * width)])
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

function findBlobs(img_gpu, out_gpu, DoG_gpu, DoGo_gpu, conv_gpu, buffer, height, width, imgWidth, octaves, scales = 5, sigma0 = 1.6, k = -1)
	if k == -1
		k = 2^(1 / (scales - 3))
	end
	time_taken = 0
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
						error_count = CuArray([0])
						time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row maxregs = 32 row_kernel_3(
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
					error_count = CuArray([0])
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
				save("assets/debug/gaussian_o$(octave)_l$(layer).png", colorview(Gray, collect(out_gpu[octave][layer])))
				if octave == 1 && layer == 1
					save("assets/go/gaussian_o$(octave)_l$(layer).png", colorview(Gray, Array(out_gpu[octave][layer])))
				end
				push!(blobs_input_l, out_gpu[octave][layer])
				if layer == scales
					threads_blobs = (32, 1024 ÷ 32)
					blocks_blobs = (cld(height, threads_blobs[1] - 2), cld(width, threads_blobs[2] - 2))
					shmem_blobs = threads_blobs[1] * threads_blobs[2] * sizeof(Float32) * 4
					DoG_gpu[octave][2] .= 0
					DoG_gpu[octave][1] .= 0
					CUDA.synchronize()
					time_taken += CUDA.@elapsed @cuda threads = threads_blobs blocks = blocks_blobs shmem = shmem_blobs maxregs = 32 blobs_2_rewrite(
						CuArray(reverse([cudaconvert(x) for x in blobs_input_l])),
						DoG_gpu[octave][2],
						DoG_gpu[octave][1],
						height,
						width,
						imgWidth / 2^(octave - 1),
						k - 1,
						DoGo_gpu[octave][4],
						DoGo_gpu[octave][3],
						DoGo_gpu[octave][2],
						DoGo_gpu[octave][1],
					)
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
		time_taken += CUDA.@elapsed buffer = CUDA.zeros(Float32, height, width)
	end
	return time_taken
end

function extractBlobXYs(img_gpu, out_gpu, DoG_gpu, XY_gpu, octaves, scales, height, width, imgWidth, iter, sigma0 = 1.6, k = -1, bins = 32)
	println("Extracting blobs...")
	if k == -1
		k = 2^(1 / (scales - 3))
	end
	time_taken = 0
	count_gpu = CuArray{UInt64}([0])
	counts = Int[]
	radii = Int[]
	height_local = height
	width_local = width
	out_gpus = []
	for octave in 1:octaves
		push!(out_gpus, out_gpu[octave][2])
		for layer in 1:scales-3
			threads = 1024
			blocks = cld(height_local * width_local, threads)
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
			sigma = sigma0 * k^((octave - 1) * 2 + layer)
			push!(radii, ceil(Int, 1.5 * sigma0 * k^(layer) + 1))
		end
		height_local = height_local ÷ 2
		width_local = width_local ÷ 2
	end
	println("streams compacted")
	count = collect(count_gpu)[1]
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
	end

	check_count = CUDA.zeros(UInt64, 1)

	printcount = CUDA.zeros(UInt64, octaves)
	time_taken += CUDA.@elapsed @cuda threads = threads blocks = count shmem = (sizeof(Float32) * (((maximum(radii) * 2 + 1) + 2 * 1)^2 + bins)) maxregs = 32 find_orientations(
		CuArray([cudaconvert(x) for x in out_gpus]),
		XY_gpu,
		orientation_gpu,
		height,
		width,
		counts_gpu,
		radii_gpu,
		bins,
		check_count,
		printcount,
	)
	CUDA.synchronize()
	println("orientations found")
	println("check_count: $(collect(check_count)[1])")
	# save orientation_gpu as csv
	CSV.write("assets/debug/orientations_i.csv", DataFrame(collect(transpose(collect(orientation_gpu))), :auto))
	# save gradient_orientations as images
	for o in 1:octaves
		save("assets/go/gradient_orientations_o$(o).png", colorview(RGBA, Array(gradient_orientations[o])))
		save("assets/go/aligned_go_o$(o).png", colorview(RGBA, map(clamp01nan, Array(aligned_go[o]))))
	end
	filtered_XY_gpu = CUDA.zeros(Float32, bins + 6, count)
	filtered_count_gpu = CuArray{UInt64}([0])
	println("count: $count, filtered count: $(collect(filtered_count_gpu)[1]), bins: $bins")

	time_taken += CUDA.@elapsed @cuda threads = (32, 512 ÷ 32) blocks = cld(count, 512 ÷ 32) shmem = (sizeof(Float32) * 32 * 32 + sizeof(UInt64)) filter_blobs(
		XY_gpu, orientation_gpu, filtered_XY_gpu, count, filtered_count_gpu, bins, 0.6)#0.534) #1.3) 
	CUDA.synchronize()

	println("filtering done")

	println("filtered count: $(collect(filtered_count_gpu)[1])")
	blank_slate = CUDA.zeros(Float32, 4, size(img_gpu)...)
	blank_slate[1:3, :, :] .= reshape(img_gpu, 1, size(img_gpu)...) .* 0.2
	blank_slate[4, :, :] .= 1.0

	# blank_slate is currently a single channel image, we need to convert it to a 4 channel image, with channel as the first dimension

	# println(size(blank_slate))
	if collect(filtered_count_gpu)[1] >= 1
		@cuda threads = 1 blocks = collect(filtered_count_gpu)[1] plot_blobs_f(filtered_XY_gpu, blank_slate, height, width, size(filtered_XY_gpu, 1))
		CUDA.synchronize()
	end
	save("assets/debug/filtered_blobs_$(iter).png", colorview(RGBA, Array(blank_slate)))
	if iter == 1
		save("assets/go/filtered_blobs.png", colorview(RGBA, Array(blank_slate)))
	end
	blank_slate[1:3, :, :] .= reshape(img_gpu, 1, size(img_gpu)...) .* 0.2
	blank_slate[4, :, :] .= 1.0
	# blank_slate = Float32.(Gray.(FileIO.load("assets/images/20241203_000635.mp4_frame_1.png")))
	# println(size(blank_slate))
	@cuda threads = 1 blocks = count plot_blobs_uf(XY_gpu, blank_slate, height, width, size(XY_gpu, 1), 0)
	CUDA.synchronize()
	save("assets/debug/blobs_$(iter).png", colorview(RGBA, Array(blank_slate)))
	if iter == 1
		save("assets/go/blobs.png", colorview(RGBA, Array(blank_slate)))
	end
	# CSV.write("assets/filtered_XY_i.csv", DataFrame(collect(transpose(collect(filtered_XY_gpu))), :auto))
	println("Size of filtered_XY_gpu: $(size(collect(filtered_XY_gpu)))")
	return time_taken, count, collect(orientation_gpu), collect(filtered_XY_gpu)[:, 1:collect(filtered_count_gpu)[1]]
end

function getBlobs(img, height, width, imgWidth, octaves, layers, nImages, sigma0, k, iterations, time_taken)
	img_gpu, out_gpu, DoG_gpu, convolution_gpu, buffer, XY_gpu, DoGo_gpu, DoG_prev_gpu = getGPUElements(img, height, width, layers, octaves, nImages, sigma0, k)
	println("Got the GPU elements...")
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

		# save the image 
		save("assets/debug/base.png", colorview(Gray, collect(img_gpu)))

		# copy out_gpu to another variable
		out_gpu_prev = copy(out_gpu)
		DoG_prev_gpu = copy(DoG_gpu)
		XY_gpu = CuArray([potential_blob() for i in 1:ceil(Integer, 0.0128 * height * width)])

		time_taken += findBlobs(img_gpu, out_gpu, DoG_gpu, DoGo_gpu, convolution_gpu, buffer, height, width, imgWidth, octaves, layers, sigma0, k)
		println("Found the blobs...")
		CUDA.synchronize()
		if 1 < i
			# compare out_gpu with out_gpu_prev
			for o in 1:octaves
				for l in 1:layers
					check = similar(out_gpu[o][l])
					h, w = size(out_gpu[o][l])
					@cuda threads = 1024 blocks = cld(h * w, 1024) check_equal(check, out_gpu[o][l], out_gpu_prev[o][l], h, w)
					# save it into a file
					save("assets/check/check_o$(o)_l$(l)_i$(i).png", colorview(Gray, collect(check)))
					if l < layers
						check = similar(DoG_gpu[o][l])
						h, w = size(DoG_gpu[o][l])
						@cuda threads = 1024 blocks = cld(h * w, 1024) check_equal(check, DoG_gpu[o][l], DoG_prev_gpu[o][l], h, w)
						# save it into a file
						save("assets/check/check_DoG_o$(o)_l$(l)_i$(i).png", colorview(Gray, collect(check)))
					end
				end
			end
		end

		time_taken_here, count, orientations, blobs = extractBlobXYs(img_gpu, out_gpu, DoG_gpu, XY_gpu, octaves, layers, height, width, imgWidth, i)
		time_taken += time_taken_here
	end
	return time_taken, count, orientations, blobs, XY_gpu
end