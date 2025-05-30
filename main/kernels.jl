include("helper.jl")

function col_kernel_strips(inp, conv, buffer, width::Int32, height::Int16, imgWidth::Int16, apron::Int8)
	let
		blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
		threadNum::UInt16 = threadIdx().x - 1

		thisX::Int32 =
			imgWidth * (blockNum ÷ UInt32(imgWidth * cld(height, blockDim().x - 2 * apron))) +
			((blockNum % UInt32(imgWidth * cld(height, blockDim().x - 2 * apron))) ÷ UInt32(cld(height, blockDim().x - 2 * apron))) + 1 # 1-indexed

		thisY::Int32 = blockNum % UInt32(cld(height, blockDim().x - 2 * apron)) * (blockDim().x - 2 * apron) - apron + threadNum + 1 # 1-indexed
		thisPX::Int32 = thisY + (thisX - 1) * height # 1-indexed

		data = CuDynamicSharedArray(Float32, blockDim().x)

		# fill the shared memory
		if 0 < thisY <= height && 0 < thisX <= width && 0 < thisPX <= height * width
			@inbounds data[threadNum+1] = @inbounds inp[thisPX]
		else
			data[threadNum+1] = 0.0
		end
		sync_threads()
		# convolution
		if 0 < thisY <= height && 0 < thisX <= width && apron <= threadNum < blockDim().x - apron
			sum::Float32 = 0.0
			for i in -apron:apron
				sum += @inbounds data[threadNum+1+i] * @inbounds conv[apron+1+i]
			end
			@inbounds buffer[thisPX] = sum
		end
	end
	return
end

# buffH is the height of the buffer including the black apron at the bottom
# inpH is the height of the image excluding the aprons, after the column kernel

function row_kernel_3(inp, conv, out, height::Int16, width::Int32, imgWidth::Int16, apron::Int8)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x
	threads::Int16 = blockDim().x * blockDim().y

	# number of blocks in a column and row, and total number of blocks in the image
	blocksInACol::UInt8 = cld(height, blockDim().x)
	blocksInARow::UInt16 = cld(imgWidth, blockDim().y - 2 * apron)
	blocksInAnImage::UInt32 = UInt32(blocksInACol) * UInt32(blocksInARow)

	# This thread's coordinates in the image
	thisY::Int16 = (blockNum % blocksInACol) * blockDim().x + threadIdx().x # 1-indexed
	thisX::Int32 = (blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2 * apron) + threadIdx().y - apron # 1-indexed

	data = CuDynamicSharedArray(Float32, threads)

	# fill the shared memory
	begin
		if 0 < thisY <= height
			if 0 < thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth
				thisPX::Int32 = thisY + (thisX - 1) * height
				@inbounds data[threadNum+1] = @inbounds inp[thisPX]
			else
				@inbounds data[threadNum+1] = 0.0
			end
		end
	end
	sync_threads()

	thisIsAComputationThread::Bool =
		(0 < thisY <= height) && (0 < thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth) && (apron < threadIdx().y <= blockDim().y - apron) && (0 < thisX <= width)

	# convolution
	if thisIsAComputationThread
		sum::Float32 = 0.0
		for i in -apron:apron
			sum += @inbounds data[threadNum+1+i*blockDim().x] * @inbounds conv[apron+1+i]
		end
		@inbounds out[thisY, thisX] = sum
	end
	return
end

# legacy kernel, no longer used
function resample_kernel(inp, out)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x - 1
	threads::Int16 = blockDim().x

	data = CuDynamicSharedArray(Float32, threads)

	h, w = size(inp)
	outPX::Int32 = blockNum * threads + threadNum + 1
	outX::Int32 = (outPX - 1) ÷ (h ÷ 2) # 0-indexed
	outY::Int16 = (outPX - 1) % (h ÷ 2) # 0-indexed

	thisX::Int32 = 2 * outX # 0-indexed
	thisY::Int16 = 2 * outY # 0-indexed
	thisPX::Int32 = thisY + thisX * h + 1

	# fill the shared memory
	if thisPX <= h * w
		data[threadNum+1] = inp[thisPX]
	end
	sync_threads()

	if outPX <= ((h ÷ 2) * (w ÷ 2))
		out[outPX] = data[threadNum+1]
	end
	return
end

function resample_kernel_2(inp, out, h, w)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x - 1
	threads::Int16 = blockDim().x

	outPX::Int32 = blockNum * threads + threadNum + 1
	outX::Int32 = (outPX - 1) ÷ (h ÷ 2) # 0-indexed
	outY::Int16 = (outPX - 1) % (h ÷ 2) # 0-indexed

	thisX::Int32 = 2 * outX # 0-indexed
	thisY::Int16 = 2 * outY # 0-indexed
	thisPX::Int32 = thisY + thisX * h + 1

	# fill the shared memory
	if thisPX <= h * w && outPX <= ((h ÷ 2) * (w ÷ 2))
		@inbounds out[outPX] = @inbounds inp[thisPX]
	end

	return
end

# legacy kernel, no longer used
function subtract(l1, l0, out, h, w, imgWidth, iApron, norm)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x - 1
	threads::Int16 = blockDim().x * blockDim().y

	thisAPPX::Int32 = blockNum * threads + threadNum # 0-indexed and indexed in the image without top and bottom aprons
	thisY::Int16 = iApron + thisAPPX % (h - 2 * iApron)  # 0-indexed
	thisX::Int32 = iApron + imgWidth * (thisAPPX ÷ ((imgWidth - 2 * iApron) * (h - 2 * iApron))) + (thisAPPX % ((imgWidth - 2 * iApron) * (h - 2 * iApron))) ÷ (h - 2 * iApron) # 0-indexed
	thisPX::Int32 = thisY + thisX * h + 1 # 1-indexed

	if (0 < thisPX <= h * w)
		@inbounds out[thisPX] = (@inbounds l1[thisPX] - @inbounds l0[thisPX]) / norm
	end
	return
end

@inline function max3(a, b, c, val)
	return val * (max(a, max(b, c)) <= val)
end

@inline function min3(a, b, c, val)
	return val * (min(a, min(b, c)) >= val)
end

function blobs(l5, l4, l3, l2, l1, out2, out1, h, w, imgWidth, ap4, ap5, norm)#, DoG4, DoG3, DoG2, DoG1)
	threadNum::UInt16 = threadIdx().x + (threadIdx().y - 1) * blockDim().x # 1-indexed
	threads = blockDim().x * blockDim().y

	data1 = CuDynamicSharedArray(Float32, threads)
	data2 = CuDynamicSharedArray(Float32, threads, sizeof(Float32) * threads)
	data3 = CuDynamicSharedArray(Float32, threads, 2 * sizeof(Float32) * threads)

	# ground truth
	# this thread has same x and y throughout the kernel. Blocklocal numbering is img - ap4 (top, bottom and verticals)
	# when I process extrema in [data1, data2, data3], I need to check if the thread is outside the ap4 + 1 in all directions
	# when I process extrema in [data2, data3, data4], I need to check if the thread is outside the ap5 + 1 in all directions

	thisY::Int32, thisX::Int32, thisPX::Int32 = let
		blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
		blocksInACol::Int32 = cld(h - 2 * (ap4 + 1), blockDim().x - 2)
		blocksInAnImage::Int32 = blocksInACol * cld(imgWidth - 2 * (ap4 + 1), blockDim().y - 2)

		ap4 + (blockNum % blocksInACol) * (blockDim().x - 2) + threadIdx().x - 1, # 0-indexed
		ap4 + (blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2) + threadIdx().y - 1, # 0-indexed
		ap4 + (blockNum % blocksInACol) * (blockDim().x - 2) + threadIdx().x - 1 + (ap4 + (blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2) + threadIdx().y - 1) * h + 1 # 1-indexed
	end

	let
		shouldIProcess = (thisY < h - ap4 && thisX % imgWidth < imgWidth - ap4)
		if (0 < thisPX <= h * w)
			# data1[threadNum] = shouldIProcess * (l2[thisPX] - l1[thisPX]) / norm
			# data2[threadNum] = shouldIProcess * (l3[thisPX] - l2[thisPX]) / norm
			# data3[threadNum] = shouldIProcess * (l4[thisPX] - l3[thisPX]) / norm
			data1[threadNum] = @inbounds l1[thisY, thisX]
			# sync_threads()
			sync_warp()
			data2[threadNum] = @inbounds l2[thisY, thisX]
			data1[threadNum] = shouldIProcess * (@inbounds data2[threadNum] - @inbounds data1[threadNum]) / norm
			# sync_threads()
			sync_warp()
			data3[threadNum] = @inbounds l3[thisY, thisX]
			data2[threadNum] = shouldIProcess * (@inbounds data3[threadNum] - @inbounds data2[threadNum]) / norm
			# sync_threads()
			sync_warp()
			data3[threadNum] = shouldIProcess * (@inbounds l4[thisY, thisX] - @inbounds data3[threadNum]) / norm
		end
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && thisY < h - ap4 && thisX % imgWidth < imgWidth - ap4)
		# data 2
		thisO = max3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data2[threadNum])
		thisO = max3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
		thisO = max3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

		# data 3
		thisO = max3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
		thisO = max3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
		thisO = max3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

		# data 1
		thisO = max3(data1[threadNum-1-blockDim().x], data1[threadNum-blockDim().x], data1[threadNum+1-blockDim().x], thisO)
		thisO = max3(data1[threadNum-1], data1[threadNum], data1[threadNum+1], thisO)
		thisO = max3(data1[threadNum-1+blockDim().x], data1[threadNum+blockDim().x], data1[threadNum+1+blockDim().x], thisO)

		if thisO != data2[threadNum]
			# data 2
			thisO = min3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data2[threadNum])
			thisO = min3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
			thisO = min3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

			# data 3
			thisO = min3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
			thisO = min3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
			thisO = min3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

			# data 1
			thisO = min3(data1[threadNum-1-blockDim().x], data1[threadNum-blockDim().x], data1[threadNum+1-blockDim().x], thisO)
			thisO = min3(data1[threadNum-1], data1[threadNum], data1[threadNum+1], thisO)
			thisO = min3(data1[threadNum-1+blockDim().x], data1[threadNum+blockDim().x], data1[threadNum+1+blockDim().x], thisO)
		end
		# @inbounds out1[thisPX] = abs(thisO)
		@inbounds out1[thisY, thisX] = abs(thisO)
		# @inbounds DoG1[thisPX] = data1[threadNum]
		# @inbounds DoG2[thisPX] = data2[threadNum]
		# @inbounds DoG3[thisPX] = data3[threadNum]
	end
	sync_threads()

	shouldIProcess = (ap5 <= thisY < h - ap5 && ap5 <= thisX % imgWidth < imgWidth - ap5)
	if (0 < thisPX <= h * w)
		# data1[threadNum] = shouldIProcess * (l4[thisPX] - l3[thisPX]) / norm
		data1[threadNum] = @inbounds l4[thisY, thisX]
		sync_warp()
		# sync_threads()
		data1[threadNum] = shouldIProcess * (@inbounds l5[thisY, thisX] - data1[threadNum]) / norm
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && ap5 <= thisY < h - ap5 && ap5 <= thisX % imgWidth < imgWidth - ap5)
		# out2
		# Unrolled loop for x = -1, 0, 1 and y = -1, 0, 1
		# data 2
		thisO = max3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data3[threadNum])
		thisO = max3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
		thisO = max3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

		# data 3
		thisO = max3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
		thisO = max3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
		thisO = max3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

		# data 1
		thisO = max3(data1[threadNum-1-blockDim().x], data1[threadNum-blockDim().x], data1[threadNum+1-blockDim().x], thisO)
		thisO = max3(data1[threadNum-1], data1[threadNum], data1[threadNum+1], thisO)
		thisO = max3(data1[threadNum-1+blockDim().x], data1[threadNum+blockDim().x], data1[threadNum+1+blockDim().x], thisO)

		if thisO != data3[threadNum]
			# data 2
			thisO = min3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data3[threadNum])
			thisO = min3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
			thisO = min3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

			# data 3
			thisO = min3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
			thisO = min3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
			thisO = min3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

			# data 1
			thisO = min3(data1[threadNum-1-blockDim().x], data1[threadNum-blockDim().x], data1[threadNum+1-blockDim().x], thisO)
			thisO = min3(data1[threadNum-1], data1[threadNum], data1[threadNum+1], thisO)
			thisO = min3(data1[threadNum-1+blockDim().x], data1[threadNum+blockDim().x], data1[threadNum+1+blockDim().x], thisO)
		end
		# @inbounds out2[thisPX] = abs(thisO)
		@inbounds out2[thisY, thisX] = abs(thisO)
		# @inbounds DoG4[thisPX] = data1[threadNum]
	end
	return
end

function blobs_2(l5, l4, l3, l2, l1, out2, out1, h, w, imgWidth, norm)#, DoG4, DoG3, DoG2, DoG1)
	threadNum::UInt16 = threadIdx().x + (threadIdx().y - 1) * blockDim().x # 1-indexed
	threads = blockDim().x * blockDim().y

	data1 = CuDynamicSharedArray(Float32, threads)
	data2 = CuDynamicSharedArray(Float32, threads, 1 * sizeof(Float32) * threads)
	data3 = CuDynamicSharedArray(Float32, threads, 2 * sizeof(Float32) * threads)
	data4 = CuDynamicSharedArray(Float32, threads, 3 * sizeof(Float32) * threads)

	data1[threadNum] = 0.0
	data2[threadNum] = 0.0
	data3[threadNum] = 0.0
	data4[threadNum] = 0.0
	sync_threads()
	# ground truth
	# this thread has same x and y throughout the kernel. Blocklocal numbering is img - ap4 (top, bottom and verticals)
	# when I process extrema in [data1, data2, data3], I need to check if the thread is outside the ap4 + 1 in all directions
	# when I process extrema in [data2, data3, data4], I need to check if the thread is outside the ap5 + 1 in all directions

	thisY::Int32, thisX::Int32 = let
		blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
		blocksInACol::Int32 = cld(h - 2, blockDim().x - 2)
		blocksInAnImage::Int32 = blocksInACol * cld(imgWidth - 2, blockDim().y - 2)

		(blockNum % blocksInACol) * (blockDim().x - 2) + threadIdx().x, # 0-indexed
		(blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2) + threadIdx().y # 0-indexed
	end

	thisPX::Int32 = thisY + thisX * h + 1

	let
		# shouldIProcess = (thisY < h && thisX % imgWidth < imgWidth)
		# if (0 < thisPX <= h * w)
		if (0 < thisY < h && 0 < thisX % imgWidth < imgWidth)
			data1[threadNum] = @inbounds l1[thisY, thisX]
			data2[threadNum] = @inbounds l2[thisY, thisX]
			data1[threadNum] = (@inbounds data2[threadNum] - @inbounds data1[threadNum]) / norm
			data3[threadNum] = @inbounds l3[thisY, thisX]
			data2[threadNum] = (@inbounds data3[threadNum] - @inbounds data2[threadNum]) / norm
			data3[threadNum] = (@inbounds l4[thisY, thisX] - @inbounds data3[threadNum]) / norm
		end
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && thisY < h && thisX % imgWidth < imgWidth)
		# data 2
		thisO = max3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data2[threadNum])
		thisO = max3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
		thisO = max3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

		# data 3
		thisO = max3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
		thisO = max3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
		thisO = max3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

		# data 1
		thisO = max3(data1[threadNum-1-blockDim().x], data1[threadNum-blockDim().x], data1[threadNum+1-blockDim().x], thisO)
		thisO = max3(data1[threadNum-1], data1[threadNum], data1[threadNum+1], thisO)
		thisO = max3(data1[threadNum-1+blockDim().x], data1[threadNum+blockDim().x], data1[threadNum+1+blockDim().x], thisO)

		if thisO != data2[threadNum]
			# data 2
			thisO = min3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data2[threadNum])
			thisO = min3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
			thisO = min3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

			# data 3
			thisO = min3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
			thisO = min3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
			thisO = min3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

			# data 1
			thisO = min3(data1[threadNum-1-blockDim().x], data1[threadNum-blockDim().x], data1[threadNum+1-blockDim().x], thisO)
			thisO = min3(data1[threadNum-1], data1[threadNum], data1[threadNum+1], thisO)
			thisO = min3(data1[threadNum-1+blockDim().x], data1[threadNum+blockDim().x], data1[threadNum+1+blockDim().x], thisO)
		end
		# @inbounds out1[thisPX] = abs(thisO)
		@inbounds out1[thisY, thisX] = abs(thisO)
		# @inbounds DoG1[thisPX] = data1[threadNum]
		# @inbounds DoG2[thisPX] = data2[threadNum]
		# @inbounds DoG3[thisPX] = data3[threadNum]
	end
	sync_threads()

	if (thisY < h && thisX % imgWidth < imgWidth)
		data4[threadNum] = @inbounds l4[thisY, thisX]
		data4[threadNum] = (@inbounds l5[thisY, thisX] - data4[threadNum]) / norm
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && thisY < h && thisX % imgWidth < imgWidth)
		# out2
		# Unrolled loop for x = -1, 0, 1 and y = -1, 0, 1
		# data 2
		thisO = max3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data3[threadNum])
		thisO = max3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
		thisO = max3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

		# data 3
		thisO = max3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
		thisO = max3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
		thisO = max3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

		# data 1
		thisO = max3(data4[threadNum-1-blockDim().x], data4[threadNum-blockDim().x], data4[threadNum+1-blockDim().x], thisO)
		thisO = max3(data4[threadNum-1], data4[threadNum], data4[threadNum+1], thisO)
		thisO = max3(data4[threadNum-1+blockDim().x], data4[threadNum+blockDim().x], data4[threadNum+1+blockDim().x], thisO)

		if thisO != data3[threadNum]
			# data 2
			thisO = min3(data2[threadNum-1-blockDim().x], data2[threadNum-blockDim().x], data2[threadNum+1-blockDim().x], data3[threadNum])
			thisO = min3(data2[threadNum-1], data2[threadNum], data2[threadNum+1], thisO)
			thisO = min3(data2[threadNum-1+blockDim().x], data2[threadNum+blockDim().x], data2[threadNum+1+blockDim().x], thisO)

			# data 3
			thisO = min3(data3[threadNum-1-blockDim().x], data3[threadNum-blockDim().x], data3[threadNum+1-blockDim().x], thisO)
			thisO = min3(data3[threadNum-1], data3[threadNum], data3[threadNum+1], thisO)
			thisO = min3(data3[threadNum-1+blockDim().x], data3[threadNum+blockDim().x], data3[threadNum+1+blockDim().x], thisO)

			# data 1
			thisO = min3(data4[threadNum-1-blockDim().x], data4[threadNum-blockDim().x], data4[threadNum+1-blockDim().x], thisO)
			thisO = min3(data4[threadNum-1], data4[threadNum], data4[threadNum+1], thisO)
			thisO = min3(data4[threadNum-1+blockDim().x], data4[threadNum+blockDim().x], data4[threadNum+1+blockDim().x], thisO)
		end
		# @inbounds out2[thisPX] = abs(thisO)
		@inbounds out2[thisY, thisX] = abs(thisO)
		# @inbounds DoG4[thisPX] = data1[threadNum]
	end
	return
end

# function blobs_2_rewrite(l5, l4, l3, l2, l1, out2, out1, h, w, imgWidth, norm, DoG4, DoG3, DoG2, DoG1)
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

# Kernel to compact the sparse local-extrema image into a list of potential blobs structs
function stream_compact(d1, xy, h, w, imgWidth, count, oct, lay)
	threadNum = threadIdx().x + blockDim().x * (blockIdx().x - 1) # 1-indexed
	warpNum = (threadIdx().x - 1) ÷ 32 # 0-indexed
	laneNum = (threadIdx().x - 1) % 32 # 0-indexed

	# offset for this block
	shared_count = CuDynamicSharedArray(UInt64, 1)

	if threadIdx().x == 1
		shared_count[1] = 0
	end
	sync_threads()

	# calculate the offset for this warp

	# see if this pixel is a local extrema
	warp_offset::UInt64 = 0
	is_nonzero = false
	if threadNum <= h * w
		is_nonzero = d1[threadNum] >= 0.01
	end
	sync_warp()
	# vote in the warp to generate a mask
	mask = CUDA.vote_ballot_sync(0xffffffff, is_nonzero)
	# count the number of non-zero pixels in this warp
	warp_count::UInt64 = count_ones(mask)

	# leader thread of the warp writes the count to shared memory
	if laneNum == 0
		warp_offset = CUDA.atomic_add!(pointer(shared_count, 1), warp_count)
	end
	# broadcast this offset to all threads in the warp from the leader thread
	warp_offset = CUDA.shfl_sync(0xffffffff, warp_offset, 1)
	sync_threads()

	# find the offset for this block with the total number of non-zero pixels in this block
	if threadIdx().x == 1
		shared_count[1] = CUDA.atomic_add!(CUDA.pointer(count, 1), shared_count[1])
	end
	sync_threads()
	# write into the output array using the offset
	if (ceil(Int, threadNum / 32) * 32 <= h * w) && d1[threadNum] != 0
		index = shared_count[1] + warp_offset + count_ones(mask & ((1 << laneNum) - 1)) # 0-indexed
		thisY = (threadNum - 1) % h + 1
		thisX = ((threadNum - 1) ÷ h) % imgWidth + 1
		thisImg = ((threadNum - 1) ÷ h) ÷ imgWidth + 1
		@inbounds xy[index+1] = potential_blob(thisX, thisY, thisImg, ((threadNum - 1) ÷ h) + 1, oct, lay)
	end
	return
end

# function find_orientations(o3, o2, o1, pointsXY, out, h, w, counts, radii, bins, go3, go2, go1, ago3, ago2, ago1, check_count)
function find_orientations(Os, pointsXY, out, h, w, counts, radii, bins, check_count, printcount)

	subset = 1 + # 1-indexed
			 (blockIdx().x > counts[1]) +
			 (blockIdx().x > counts[2]) +
			 (blockIdx().x > counts[3]) +
			 (blockIdx().x > counts[4]) +
			 (blockIdx().x > counts[5])

	r::Int16 = radii[subset]

	l_threadNum = threadIdx().x + ((2 * r + 1 + 2 * 1)) * (threadIdx().y - 1) # 1-indexed 
	data = CuDynamicSharedArray(Float32, (2 * r + 1 + 2 * 1)^2)
	orientation = CuDynamicSharedArray(Float32, bins, sizeof(Float32) * (2 * r + 1 + 2 * 1)^2)

	if l_threadNum <= bins
		orientation[l_threadNum] = 0.0
	end

	octave = cld(subset, 2)
	o, h, w = Os[octave], Int(h / 2^(octave - 1)), Int(w / 2^(octave - 1))

	X = pointsXY[blockIdx().x].x
	Y = pointsXY[blockIdx().x].y

	x = X + threadIdx().y - r - 2 # 1-indexed
	y = Y + threadIdx().x - r - 2 # 1-indexed
	sync_threads()

	# load elements around XY from the octave
	let
		if 0 < x <= w && 0 < y <= h && threadIdx().x <= 2 * radii[subset] + 1 + 2 && threadIdx().y <= 2 * radii[subset] + 1 + 2
			data[l_threadNum] = o[y, x]
		end
	end
	sync_threads()

	let
		if (1 < x < w && 1 < y < h && 1 < threadIdx().x <= 2 * radii[subset] + 1 + 1 && 1 < threadIdx().y <= 2 * radii[subset] + 1 + 1)# || (-2 < (X - 1231) < 2 && -2 < (Y - 82) < 2)
			if x == X && y == Y
				CUDA.@atomic check_count[1] += 1
			end

			# Calculate the first order derivative through first central difference method
			dy = data[l_threadNum-1] - data[l_threadNum+1]
			dx = data[l_threadNum+(2*r+1+2)] - data[l_threadNum-(2*r+1+2)]
			# Calculate the gaussian weight and the magnitude of the position and the orientation of the gradient.
			weight = exp(-((x - X)^2 + (y - Y)^2) / (2 * (r * 1)^2)) / (2 * pi * (r * 1))
			magnitude = (x - X)^2 + (y - Y)^2 > 0 ? (dx * (X - x) - dy * (Y - y)) / (2 * sqrt((x - X)^2 + (y - Y)^2)) : sqrt(dy^2 + dx^2) / 4
			# Calculate the bin number into which the orientation accumulates
			bin::Int32 = (x - X)^2 + (y - Y)^2 > 0 ? fld((atan((Y - y), (X - x)) + 2 * pi) % (2 * pi), 2 * pi / bins) + 1 : fld((atan(dy, dx) + 2 * pi) % (2 * pi), 2 * pi / bins) + 1 # 1-indexed

			CUDA.@atomic orientation[bin] += weight * magnitude
		end
	end
	sync_threads()
	if l_threadNum <= bins
		@inbounds out[l_threadNum+(blockIdx().x-1)*bins] = orientation[l_threadNum]
	end
	return
end

function filter_blobs(pointXY, orientations, out, count, outCount, bins, threshold = 3)
	# assert bins <= 32
	@assert bins <= 32 "Number of bins should be less than 33"
	l_threadNum = threadIdx().x + blockDim().x * (threadIdx().y - 1) # 1-indexed
	threadNum = l_threadNum + blockDim().x * (blockIdx().x - 1) # 1-indexed

	shared_count = CuDynamicSharedArray(UInt64, 1)
	shared_orientations = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y), sizeof(UInt64))

	if threadIdx().x == 1 && threadIdx().y == 1
		shared_count[1] = 0
	end
	shared_orientations[threadIdx().x, threadIdx().y] = 0.0
	if threadIdx().x <= bins && threadIdx().y + (blockIdx().x - 1) * blockDim().y <= count
		shared_orientations[threadIdx().x, threadIdx().y] = orientations[threadIdx().x, (blockIdx().x-1)*blockDim().y+threadIdx().y]
	end
	sync_threads()

	coeff_of_variation = let
		# for each warp, calculate sum of orientations
		local_mean = shared_orientations[threadIdx().x, threadIdx().y]
		sync_warp()
		for offset in 4:-1:0
			local_mean += CUDA.shfl_down_sync(0xffffffff, local_mean, 1 << offset)
		end
		local_mean = CUDA.shfl_sync(0xffffffff, local_mean, 1)
		local_mean = local_mean / bins

		local_deviation = 0.0
		if threadIdx().x <= bins
			local_deviation = (shared_orientations[threadIdx().x, threadIdx().y] - local_mean)
			local_deviation = local_deviation * local_deviation
		end
		sync_warp()

		# for each warp, calculate sum of squared differences
		for offset in 4:-1:0
			local_deviation += CUDA.shfl_down_sync(0xffffffff, local_deviation, 1 << offset)
		end
		local_deviation = CUDA.shfl_sync(0xffffffff, local_deviation, 1)
		local_deviation = sqrt(local_deviation / bins)

		if local_mean == 0 || abs(local_mean) < 1e-3
			typemax(Float32)
		else
			local_deviation / abs(local_mean)
		end
	end

	sync_warp()

	thisPoint = 0
	if coeff_of_variation < threshold
		if threadIdx().x == 1
			thisPoint = CUDA.@atomic shared_count[1] += 1
		end
		thisPoint = CUDA.shfl_sync(0xffffffff, thisPoint, 1)
	end
	sync_threads()


	if threadIdx().x == 1 && threadIdx().y == 1
		# shared_count[1] = CUDA.atomic_add!(pointer(outCount, 1), shared_count[1])
		shared_count[1] = CUDA.@atomic outCount[1] += shared_count[1]
	end
	sync_threads()
	if coeff_of_variation < threshold
		if threadIdx().x <= bins
			out[(shared_count[1]+thisPoint)*(bins+6)+threadIdx().x] = shared_orientations[threadIdx().x, threadIdx().y]

		end
		sync_threads()
		if threadIdx().x == 1
			out[(shared_count[1]+thisPoint)*(bins+6)+1+bins] = Float32(2^(pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].oct - 1) * pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].thisX)
			out[(shared_count[1]+thisPoint)*(bins+6)+2+bins] = Float32(2^(pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].oct - 1) * pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].y)
			out[(shared_count[1]+thisPoint)*(bins+6)+3+bins] = Float32(pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].oct)
			out[(shared_count[1]+thisPoint)*(bins+6)+4+bins] = Float32(pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].lay)
			out[(shared_count[1]+thisPoint)*(bins+6)+5+bins] = Float32(2^(pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].oct - 1) * pointXY[threadIdx().y+(blockIdx().x-1)*blockDim().y].x)
			out[(shared_count[1]+thisPoint)*(bins+6)+6+bins] = Float32(coeff_of_variation)
		end
	end

	return
end

# Helper Kernels to plot and debug
function plot_blobs_f(points, img, h, w, stride, pType = 1)
	X = points[(blockIdx().x-1)*stride+32+5]
	Y = points[(blockIdx().x-1)*stride+32+2]
	o = points[(blockIdx().x-1)*stride+32+3]
	if 0 < X <= w && 0 < Y <= h
		img[1, Integer(Y), Integer(X)] = 1.0
		img[2, Integer(Y), Integer(X)] = 1.0
		img[3, Integer(Y), Integer(X)] = 1.0
		img[Integer(o), Integer(Y), Integer(X)] = 0.0
		img[4, Integer(Y), Integer(X)] = 1.0
	end
	return
end

function plot_blobs_uf(points, img, h, w, stride, pType = 0)
	o = (points[blockIdx().x].oct - 1) % 3 + 1
	X = points[blockIdx().x].x * 2^(o - 1)
	Y = points[blockIdx().x].y * 2^(o - 1)
	if 0 < X <= w && 0 < Y <= h
		img[Integer(1 + (Y - 1 + (X - 1) * h) * 4)] = 1.0
		img[Integer(2 + (Y - 1 + (X - 1) * h) * 4)] = 1.0
		img[Integer(3 + (Y - 1 + (X - 1) * h) * 4)] = 1.0
		img[Integer(o + (Y - 1 + (X - 1) * h) * 4)] = 0.0
		img[Integer(4 + (Y - 1 + (X - 1) * h) * 4)] = 1.0
	end
	return
end

function check_equal(out, img1, img2, h, w)
	threadNum = threadIdx().x + blockDim().x * (blockIdx().x - 1) # 1-indexed
	if threadNum <= h * w
		if img1[threadNum] != img2[threadNum]
			out[threadNum] = 0.0
			@cuprintln("$(threadNum): $(img1[threadNum]) != $(img2[threadNum])")
		else
			out[threadNum] = 1.0
		end
	end
	return
end