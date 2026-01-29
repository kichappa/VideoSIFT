function blobs_2_rewrite(l, out2, out1, h, w, imgWidth, norm, DoG4, DoG3, DoG2, DoG1)
	tN::UInt16 = threadIdx().x + (threadIdx().y - 1) * blockDim().x
	threads = blockDim().x * blockDim().y

	data1 = CuDynamicSharedArray(Float32, threads)
	data2 = CuDynamicSharedArray(Float32, threads, 1 * sizeof(Float32) * threads)
	data3 = CuDynamicSharedArray(Float32, threads, 2 * sizeof(Float32) * threads)
	data4 = CuDynamicSharedArray(Float32, threads, 3 * sizeof(Float32) * threads)

	data1[tN] = 0.0
	data2[tN] = 0.0
	data3[tN] = 0.0
	data4[tN] = 0.0
	sync_threads()

	thisY::Int32, thisX::Int32 = let
		blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
		blocksInACol::Int32 = cld(h - 2, blockDim().x - 2)
		blocksInAnImage::Int32 = blocksInACol * cld(imgWidth - 2, blockDim().y - 2)

		(blockNum % blocksInACol) * (blockDim().x - 2) + threadIdx().x - 1, # 0-indexed
		(blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2) + threadIdx().y # 0-indexed
	end

	# shouldIProcess = (thisY < h && thisX % imgWidth < imgWidth)
	# if (0 < thisPX <= h * w)
	if (0 < thisY <= h && 0 <= thisX % imgWidth < imgWidth && 0 < thisX <= w)
		# @cuprintln("t($(threadIdx().x),$(threadIdx().y)) b($(blockIdx().x),$(blockIdx().y)), ($thisY,$thisX)")
		data1[tN] = @inbounds l[1][thisY, thisX]
		data2[tN] = @inbounds l[2][thisY, thisX]
		data1[tN] = (@inbounds data2[tN] - @inbounds data1[tN]) / norm
		data3[tN] = @inbounds l[3][thisY, thisX]
		data2[tN] = (@inbounds data3[tN] - @inbounds data2[tN]) / norm
		data4[tN] = @inbounds l[4][thisY, thisX]
		data3[tN] = (@inbounds data4[tN] - @inbounds data3[tN]) / norm
		data4[tN] = (@inbounds l[5][thisY, thisX] - @inbounds data4[tN]) / norm

		DoG1[thisY, thisX] = data1[tN]
		DoG2[thisY, thisX] = data2[tN]
		DoG3[thisY, thisX] = data3[tN]
		DoG4[thisY, thisX] = data4[tN]
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && thisY < h && thisX % imgWidth < imgWidth)
		max_layer_2 = 0.0
		min_layer_2 = 0.0
		max_layer_3 = 0.0
		min_layer_3 = 0.0
		max_layer1 =
			max(
				max(
					max(max(data1[tN-1-blockDim().x], data1[tN-blockDim().x]), data1[tN+1-blockDim().x]),
					max(max(data1[tN-1], data1[tN]), data1[tN+1]),
				),
				max(max(data1[tN-1+blockDim().x], data1[tN+blockDim().x]), data1[tN+1+blockDim().x]),
			)
		min_layer1 =
			min(
				min(
					min(min(data1[tN-1-blockDim().x], data1[tN-blockDim().x]), data1[tN+1-blockDim().x]),
					min(min(data1[tN-1], data1[tN]), data1[tN+1]),
				),
				min(min(data1[tN-1+blockDim().x], data1[tN+blockDim().x]), data1[tN+1+blockDim().x]),
			)
		max_layer2 =
			max(
				max(
					max(max(data2[tN-1-blockDim().x], data2[tN-blockDim().x]), data2[tN+1-blockDim().x]),
					max(max(data2[tN-1], data2[tN]), data2[tN+1]),
				),
				max(max(data2[tN-1+blockDim().x], data2[tN+blockDim().x]), data2[tN+1+blockDim().x]),
			)
		min_layer2 =
			min(
				min(
					min(min(data2[tN-1-blockDim().x], data2[tN-blockDim().x]), data2[tN+1-blockDim().x]),
					min(min(data2[tN-1], data2[tN]), data2[tN+1]),
				),
				min(min(data2[tN-1+blockDim().x], data2[tN+blockDim().x]), data2[tN+1+blockDim().x]),
			)
		max_layer3 =
			max(
				max(
					max(max(data3[tN-1-blockDim().x], data3[tN-blockDim().x]), data3[tN+1-blockDim().x]),
					max(max(data3[tN-1], data3[tN]), data3[tN+1]),
				),
				max(max(data3[tN-1+blockDim().x], data3[tN+blockDim().x]), data3[tN+1+blockDim().x]),
			)
		min_layer3 =
			min(
				min(
					min(min(data3[tN-1-blockDim().x], data3[tN-blockDim().x]), data3[tN+1-blockDim().x]),
					min(min(data3[tN-1], data3[tN]), data3[tN+1]),
				),
				min(min(data3[tN-1+blockDim().x], data3[tN+blockDim().x]), data3[tN+1+blockDim().x]),
			)
		max_layer4 =
			max(
				max(
					max(max(data4[tN-1-blockDim().x], data4[tN-blockDim().x]), data4[tN+1-blockDim().x]),
					max(max(data4[tN-1], data4[tN]), data4[tN+1]),
				),
				max(max(data4[tN-1+blockDim().x], data4[tN+blockDim().x]), data4[tN+1+blockDim().x]),
			)
		min_layer4 =
			min(
				min(
					min(min(data4[tN-1-blockDim().x], data4[tN-blockDim().x]), data4[tN+1-blockDim().x]),
					min(min(data4[tN-1], data4[tN]), data4[tN+1]),
				),
				min(min(data4[tN-1+blockDim().x], data4[tN+blockDim().x]), data4[tN+1+blockDim().x]),
			)
		max_all = max(max(max_layer1, max_layer2), max_layer3)
		min_all = min(min(min_layer1, min_layer2), min_layer3)
		if data2[tN] == max_all || data2[tN] == min_all
			@inbounds out1[thisY, thisX] = abs(data2[tN])
		end
		max_all = max(max(max_layer2, max_layer3), max_layer4)
		min_all = min(min(min_layer2, min_layer3), min_layer4)
		if data3[tN] == max_all || data3[tN] == min_all
			@inbounds out2[thisY, thisX] = abs(data3[tN])
		end
	end
	return
end