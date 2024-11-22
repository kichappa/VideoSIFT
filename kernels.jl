function col_kernel_strips(inp, conv, buffer, width::Int32, height::Int16, apron::Int8)
	let
		blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
		threadNum::UInt16 = threadIdx().x - 1
		# threads::Int16 = blockDim().x

		# if blockNum == 0 && threadNum == 0
		#     @cuprintln("COL: size of inp: $(size(inp)), size of out/buffer: $(size(buffer))")
		# end
		# there could be more blocks than needed
		# thisX::Int32 = blockNum ÷ Int32(cld((height - 2 * apron), (threads - 2 * apron))) + 1 # 1-indexed
		thisX::Int32 = blockNum ÷ Int32(cld((height - 2 * apron), (blockDim().x - 2 * apron))) + 1 # 1-indexed
		thisY::Int16 = blockNum % cld((height - 2 * apron), (blockDim().x - 2 * apron)) * (blockDim().x - 2 * apron) + (threadIdx().x - 1) + 1 # 1-indexed
		thisPX::Int32 = 0

		data = CuDynamicSharedArray(Float32, blockDim().x)

		# fill the shared memory
		if thisY <= height && thisX <= width
			thisPX = thisY + (thisX - 1) * height
			data[threadNum+1] = inp[thisPX]
			# data[threadIdx().x] = inp[thisPX]
		end
		sync_threads()
		# convolution
		if apron < thisY <= height - apron && thisX <= width && apron <= (threadIdx().x - 1) < (blockDim().x) - apron
			sum::Float32 = 0.0
			for i in -apron:apron
				sum += data[threadNum+1+i] * conv[apron+1+i]
			end
			buffer[thisY, thisX] = sum
		end
	end
	return
end

function col_kernel_strips_2(inp, conv, buffer, width::Int32, height::Int16, imgWidth::Int16, iApron::Int8, apron::Int8)
	let
		blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
		threadNum::UInt16 = threadIdx().x - 1
		# threads::Int16 = blockDim().x

		# if blockNum == 0 && threadNum == 0
		#     @cuprint("COL: size of inp: $(size(inp)), size of out/buffer: $(size(buffer))")
		# end
		# there could be more blocks than needed
		# thisX::Int32 = blockNum ÷ Int32(cld((height - 2 * apron), (threads - 2 * apron))) + 1 # 1-indexed
		thisX::Int32 =
			iApron + imgWidth * (blockNum ÷ UInt32((imgWidth - 2 * iApron) * cld((height - 2 * (iApron + apron)), (blockDim().x - 2 * apron)))) +
			((blockNum % UInt32((imgWidth - 2 * iApron) * cld((height - 2 * (iApron + apron)), (blockDim().x - 2 * apron)))) ÷ UInt32(cld((height - 2 * (iApron + apron)), (blockDim().x - 2 * apron)))) + 1 # 1-indexed
		thisY::Int16 = iApron + (blockNum % cld((height - 2 * (iApron + apron)), (blockDim().x - 2 * apron)) * (blockDim().x - 2 * apron) + threadNum + 1) # 1-indexed
		thisPX::Int32 = thisY + (thisX - 1) * height # 1-indexed

		data = CuDynamicSharedArray(Float32, blockDim().x)

		# fill the shared memory
		if iApron < thisY <= height - iApron && iApron < thisX <= width - iApron && 0< thisPX <= height * width
			data[threadNum+1] = inp[thisPX]
		end
		sync_threads()
		# convolution
		if (apron + iApron) < thisY <= height - (apron + iApron) && iApron < thisX <= width - iApron && apron <= (threadIdx().x - 1) < (blockDim().x) - apron
			sum::Float32 = 0.0
			for i in -apron:apron
				sum += data[threadNum+1+i] * conv[apron+1+i]
			end
			buffer[thisPX] = sum
		end
	end
	return
end

# buffH is the height of the buffer including the black apron at the bottom
# inpH is the height of the image excluding the aprons, after the column kernel
function row_kernel(inp, conv, out, inpH::Int16, buffH::Int16, width::Int32, imgWidth::Int16, apron::Int8)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	# threadNum::UInt16 = threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x
	# threads::Int16 = blockDim().x * blockDim().y


	# if blockNum == 0 && (threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x) == 0
	#     @cuprintln("ROW: size of inp: $(size(inp)), size of out: $(size(out))")
	# end
	if true #threads <= width

		# blocksInACol::Int8 = cld(inpH, blockDim().x)
		blocksInARow::Int16 = cld(imgWidth - 2 * apron, blockDim().y - 2 * apron)
		# blocksInAnImage::Int16 = blocksInACol * blocksInARow
		blocksInAnImage::Int16 = cld(inpH, blockDim().x) * blocksInARow
		# #             |  number of images to the left * imgWidth |   blockNum wrt this image ÷ blocksInAColumn   * thrds in x   | number of threads on the left|
		# thisX::Int32 = fld(blockNum, blocksInAnImage) * imgWidth + fld(blockNum % blocksInAnImage, blocksInACol) * blockDim().y + threadIdx().y # 1-indexed
		# thisY::Int16 = blockNum % blocksInACol * blockDim().x + threadIdx().x # 1-indexed

		# thisImage::Int8 = blockNum ÷ blocksInAnImage # 0-indexed
		# thisBlockNum::Int16 = blockNum % blocksInAnImage # 0-indexed

		thisX::Int32 = (blockNum ÷ blocksInAnImage) * imgWidth + ((blockNum % blocksInAnImage) % blocksInARow) * (blockDim().y - 2 * apron) + threadIdx().y # 1-indexed
		thisY::Int16 = ((blockNum % blocksInAnImage) ÷ blocksInARow) * blockDim().x + threadIdx().x + apron # 1-indexed

		data = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y))

		begin
			# fill the shared memory
			thisPX::Int32 = thisY + (thisX - 1) * buffH
			if thisX <= width && thisY <= inpH + apron
				data[(threadIdx().x-1+(threadIdx().y-1)*blockDim().x)+1] = inp[thisPX]
			end
		end
		sync_threads()

		# if (threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x)==0 && blockNum==0
		#     @cuprintln("Size of inp: $(size(inp)), size of out: $(size(out)), size of data: $(size(data))")
		# end

		thisIsAComputationThread::Bool = thisY <= inpH + apron && apron < thisX <= width - apron && apron < threadIdx().y <= blockDim().y - apron
		if (blockNum % blocksInAnImage) % blocksInARow == blocksInARow - 1
			thisIsAComputationThread = thisIsAComputationThread && (thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth - 2 * apron)
		end
		begin
			# convolution
			# if thisY == 1073 && apron==6 && thisX > 3900
			#     @cuprintln("isThisAComputationThread: $(thisIsAComputationThread), thisX: $thisX)")
			# end
			if thisIsAComputationThread
				sum::Float32 = 0.0
				for i in -apron:apron
					sum += data[(threadIdx().x-1+(threadIdx().y-1)*blockDim().x)+1+i*blockDim().x] * conv[apron+1+i]
				end
				# out[thisY, thisX-apron-fld(blockNum, blocksInAnImage)*2*apron] = sum
				out[thisY, thisX] = sum
				# out[thisY-apron, thisX-apron] = sum
			end
		end
	end
	return
end

function row_kernel_2(inp, conv, out, height::Int16, width::Int32, imgWidth::Int16, iApron::Int8, apron::Int8)
	# FOR CUDA registers, x is vertical and y is horizontal. So, threadIdx().x is vertical and threadIdx().y is horizontal
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x
	threads::Int16 = blockDim().x * blockDim().y

	blocksInACol::Int8 = cld(height - 2 * (iApron + apron), blockDim().x)
	blocksInARow::Int16 = cld(imgWidth - 2 * (iApron + apron), blockDim().y - 2 * apron)
	blocksInAnImage::Int16 = blocksInACol * blocksInARow

	thisY::Int16 = iApron + apron + (blockNum % blocksInACol) * blockDim().x + threadIdx().x # 1-indexed
	thisX::Int32 = iApron + (blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2 * apron) + threadIdx().y # 1-indexed

	data = CuDynamicSharedArray(Float32, threads)

	# fill the shared memory
	begin
		if (iApron + apron) < thisY <= height - (iApron + apron) && iApron < thisX <= width - iApron
			thisPX::Int32 = thisY + (thisX - 1) * height
			data[threadNum+1] = inp[thisPX]
		end
	end
	sync_threads()

	thisIsAComputationThread::Bool =
		((iApron + apron) < thisY <= height - (iApron + apron)) && ((iApron + apron) < thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth - (iApron + apron)) && (apron < threadIdx().y <= blockDim().y - apron) &&
		((iApron + apron) < thisX <= width - (iApron + apron))

	if thisIsAComputationThread
		sum::Float32 = 0.0
		for i in -apron:apron
			sum += data[threadNum+1+i*blockDim().x] * conv[apron+1+i]
		end
		out[thisY, thisX] = sum
	end
	return
end

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

	# convolution
	# if threadNum % 100 == 0
	#     @cuprintln("thisPX: $thisPX, outPX: $outPX, h: $h, w: $w")
	# end
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
		out[outPX] = inp[thisPX]
	end

	return
end

function subtract(l1, l0, out, h, w, imgWidth, iApron, norm)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x - 1
	threads::Int16 = blockDim().x * blockDim().y

	# thisPX::Int32 = blockNum * threads + threadNum + 1 # 1-indexed
	# thisX::Int32 = (thisPX - 1) ÷ h # 0-indexed
	# thisY::Int16 = (thisPX - 1) % h # 0-indexed
	thisAPPX::Int32 = blockNum * threads + threadNum # 0-indexed and indexed in the image without top and bottom aprons
	thisY::Int16 = iApron + thisAPPX % (h - 2 * iApron)  # 0-indexed
	thisX::Int32 = iApron + imgWidth * (thisAPPX ÷ ((imgWidth - 2 * iApron) * (h - 2 * iApron))) + (thisAPPX % ((imgWidth - 2 * iApron) * (h - 2 * iApron))) ÷ (h - 2 * iApron) # 0-indexed
	thisPX::Int32 = thisY + thisX * h + 1 # 1-indexed

	if (0 < thisPX <= h * w)
		# out[thisPX] = (iApron <= thisY < h - iApron && iApron <= thisX % imgWidth < imgWidth - iApron && 0 < thisPX <= h * w) * (l1[thisPX] - l0[thisPX]) / norm
		out[thisPX] = (l1[thisPX] - l0[thisPX]) / norm
	end
	return
end

@inline function max3(a, b, c, val)
	return (max(a, max(b, c)) <= val) * val
end


@inline function min3(a, b, c, val)
	return (min(a, min(b, c)) >= val) * val
end


function blobs(l5, l4, l3, l2, l1, out2, out1, h, w, imgWidth, ap4, ap5, norm)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x + (threadIdx().y - 1) * blockDim().x # 1-indexed
	threads = blockDim().x * blockDim().y

	data = CuDynamicSharedArray(Float32, threads * 3)

	# ground truth
	# this thread has same x and y throughout the kernel. Blocklocal numbering is img - ap4 (top, bottom and verticals)
	# when I process extrema in [data1, data2, data3], I need to check if the thread is outside the ap4 + 1 in all directions
	# when I process extrema in [data2, data3, data4], I need to check if the thread is outside the ap5 + 1 in all directions

	blocksInACol::Int32 = cld(h - 2 * (ap4 + 1), blockDim().x - 2)
	blocksInAnImage::Int32 = blocksInACol * cld(imgWidth - 2 * (ap4 + 1), blockDim().y - 2)

	thisY::Int32 = ap4 + (blockNum % blocksInACol) * (blockDim().x - 2) + threadIdx().x - 1 # 0-indexed
	thisX::Int32 = ap4 + (blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2) + threadIdx().y - 1 # 0-indexed
	thisPX::Int32 = thisY + thisX * h + 1 # 1-indexed

	shouldIProcess = (thisY < h - ap4 && thisX % imgWidth < imgWidth - ap4)

	if (0 < thisPX <= h * w)
		# data[threadNum] = shouldIProcess * (l2[thisPX] - l1[thisPX]) / norm
		# data[blockDim().x * blockDim().y +threadNum] = shouldIProcess * (l3[thisPX] - l2[thisPX]) / norm
		# data[threads*2+threadNum] = shouldIProcess * (l4[thisPX] - l3[thisPX]) / norm

		data[threadNum] = l1[thisPX]
		# sync_threads()
		sync_warp()
		data[threadNum+threads] = l2[thisPX]
		data[threadNum] = shouldIProcess * (l2[thisPX] - data[threadNum]) / norm
		# sync_threads()
		sync_warp()
		data[threadNum+threads*2] = l3[thisPX]
		data[threadNum+threads] = shouldIProcess * (l3[thisPX] - data[threadNum+threads]) / norm
		# sync_threads()
		sync_warp()
		data[threadNum+threads*2] = shouldIProcess * (l4[thisPX] - data[threadNum+threads*2]) / norm
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && thisY < h - ap4 && thisX % imgWidth < imgWidth - ap4)
		# 	# out1
		# 	# Unrolled loop for x = -1, 0, 1 and y = -1, 0, 1
		if (thisPX > h * w)
			@cuprintln("ThreadNum: $threadNum, blockNum: $blockNum, thisX: $thisX, thisY: $thisY, h: $h, w: $w, (thisX mod imgWidth): $(thisX % imgWidth), imgWidth-ap4: $(imgWidth - ap4), blocksInAnImage: $blocksInAnImage, blocksInACol: $blocksInACol")
		end
		thisO = 0.0
		# data 2
		thisO = max3(data[threads+threadNum-1-blockDim().x], data[threads+threadNum-blockDim().x], data[threads+threadNum+1-blockDim().x], data[threads+threadNum])
		thisO = max3(data[threads+threadNum-1], data[threads+threadNum], data[threads+threadNum+1], thisO)
		thisO = max3(data[threads+threadNum-1+blockDim().x], data[threads+threadNum+blockDim().x], data[threads+threadNum+1+blockDim().x], thisO)

		# data 3
		thisO = max3(data[threads*2+threadNum-1-blockDim().x], data[threads*2+threadNum-blockDim().x], data[threads*2+threadNum+1-blockDim().x], thisO)
		thisO = max3(data[threads*2+threadNum-1], data[threads*2+threadNum], data[threads*2+threadNum+1], thisO)
		thisO = max3(data[threads*2+threadNum-1+blockDim().x], data[threads*2+threadNum+blockDim().x], data[threads*2+threadNum+1+blockDim().x], thisO)

		# data 1
		thisO = max3(data[threadNum-1-blockDim().x], data[threadNum-blockDim().x], data[threadNum+1-blockDim().x], thisO)
		thisO = max3(data[threadNum-1], data[threadNum], data[threadNum+1], thisO)
		thisO = max3(data[threadNum-1+blockDim().x], data[threadNum+blockDim().x], data[threadNum+1+blockDim().x], thisO)

		if thisO != data[threads+threadNum]
			# data 2
			thisO = min3(data[threads+threadNum-1-blockDim().x], data[threads+threadNum-blockDim().x], data[threads+threadNum+1-blockDim().x], data[threads+threadNum])
			thisO = min3(data[threads+threadNum-1], data[threads+threadNum], data[threads+threadNum+1], thisO)
			thisO = min3(data[threads+threadNum-1+blockDim().x], data[threads+threadNum+blockDim().x], data[threads+threadNum+1+blockDim().x], thisO)

			# data 3
			thisO = min3(data[threads*2+threadNum-1-blockDim().x], data[threads*2+threadNum-blockDim().x], data[threads*2+threadNum+1-blockDim().x], thisO)
			thisO = min3(data[threads*2+threadNum-1], data[threads*2+threadNum], data[threads*2+threadNum+1], thisO)
			thisO = min3(data[threads*2+threadNum-1+blockDim().x], data[threads*2+threadNum+blockDim().x], data[threads*2+threadNum+1+blockDim().x], thisO)

			# data 1
			thisO = min3(data[threadNum-1-blockDim().x], data[threadNum-blockDim().x], data[threadNum+1-blockDim().x], thisO)
			thisO = min3(data[threadNum-1], data[threadNum], data[threadNum+1], thisO)
			thisO = min3(data[threadNum-1+blockDim().x], data[threadNum+blockDim().x], data[threadNum+1+blockDim().x], thisO)
		end
		out1[thisPX] = abs(thisO)
	end

	shouldIProcess = (ap5 <= thisY < h - ap5 && ap5 <= thisX % imgWidth < imgWidth - ap5)
	if (0 < thisPX <= h * w)
		# data[threadNum] = shouldIProcess * (l2[thisPX] - l1[thisPX]) / norm
		# data[blockDim().x * blockDim().y +threadNum] = shouldIProcess * (l3[thisPX] - l2[thisPX]) / norm
		# data[threads*2+threadNum] = shouldIProcess * (l4[thisPX] - l3[thisPX]) / norm

		data[threadNum] = l4[thisPX]
		sync_warp()
		# sync_threads()
		data[threadNum] = shouldIProcess * (l5[thisPX] - data[threadNum]) / norm
	end
	sync_threads()

	if (1 < threadIdx().x < blockDim().x && 1 < threadIdx().y < blockDim().y && ap5 <= thisY < h - ap5 && ap5 <= thisX % imgWidth < imgWidth - ap5)
		# out2
		# Unrolled loop for x = -1, 0, 1 and y = -1, 0, 1
		# data 2
		thisO = max3(data[threads+threadNum-1-blockDim().x], data[threads+threadNum-blockDim().x], data[threads+threadNum+1-blockDim().x], data[threadNum])
		thisO = max3(data[threads+threadNum-1], data[threads+threadNum], data[threads+threadNum+1], thisO)
		thisO = max3(data[threads+threadNum-1+blockDim().x], data[threads+threadNum+blockDim().x], data[threads+threadNum+1+blockDim().x], thisO)

		# data 3
		thisO = max3(data[threads*2+threadNum-1-blockDim().x], data[threads*2+threadNum-blockDim().x], data[threads*2+threadNum+1-blockDim().x], thisO)
		thisO = max3(data[threads*2+threadNum-1], data[threads*2+threadNum], data[threads*2+threadNum+1], thisO)
		thisO = max3(data[threads*2+threadNum-1+blockDim().x], data[threads*2+threadNum+blockDim().x], data[threads*2+threadNum+1+blockDim().x], thisO)

		# data 1
		thisO = max3(data[threadNum-1-blockDim().x], data[threadNum-blockDim().x], data[threadNum+1-blockDim().x], thisO)
		thisO = max3(data[threadNum-1], data[threadNum], data[threadNum+1], thisO)
		thisO = max3(data[threadNum-1+blockDim().x], data[threadNum+blockDim().x], data[threadNum+1+blockDim().x], thisO)

		if thisO != data[threadNum]
			# data 2
			thisO = min3(data[threads+threadNum-1-blockDim().x], data[threads+threadNum-blockDim().x], data[threads+threadNum+1-blockDim().x], data[threadNum])
			thisO = min3(data[threads+threadNum-1], data[threads+threadNum], data[threads+threadNum+1], thisO)
			thisO = min3(data[threads+threadNum-1+blockDim().x], data[threads+threadNum+blockDim().x], data[threads+threadNum+1+blockDim().x], thisO)

			# data 3
			thisO = min3(data[threads*2+threadNum-1-blockDim().x], data[threads*2+threadNum-blockDim().x], data[threads*2+threadNum+1-blockDim().x], thisO)
			thisO = min3(data[threads*2+threadNum-1], data[threads*2+threadNum], data[threads*2+threadNum+1], thisO)
			thisO = min3(data[threads*2+threadNum-1+blockDim().x], data[threads*2+threadNum+blockDim().x], data[threads*2+threadNum+1+blockDim().x], thisO)

			# data 1
			thisO = min3(data[threadNum-1-blockDim().x], data[threadNum-blockDim().x], data[threadNum+1-blockDim().x], thisO)
			thisO = min3(data[threadNum-1], data[threadNum], data[threadNum+1], thisO)
			thisO = min3(data[threadNum-1+blockDim().x], data[threadNum+blockDim().x], data[threadNum+1+blockDim().x], thisO)
		end
		out2[thisPX] = abs(thisO)
	end
	return
end

function testBlobs(l3, l2, l1, out2, out1, h, w, imgWidth, ap4)
	blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
	threadNum::UInt16 = threadIdx().x + (threadIdx().y - 1) * blockDim().x # 1-indexed
	# threads = blockDim().x * blockDim().y

	data = CuDynamicSharedArray(Float32, blockDim().x * blockDim().y * 2)

	blocksInACol::Int32 = cld(h - 2 * ap4, blockDim().x - 2)
	blocksInAnImage::Int32 = blocksInACol * cld(imgWidth - 2 * ap4, blockDim().y - 2)

	thisY::Int32 = ap4 + (blockNum % blocksInACol) * (blockDim().x - 2) + threadIdx().x - 1 # 0-indexed
	thisX::Int32 = ap4 + (blockNum ÷ blocksInAnImage) * imgWidth + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2) + threadIdx().y - 1 # 0-indexed
	thisPX::Int32 = thisY + thisX * h + 1 # 1-indexed

	shouldIProcess = (thisY < h - ap4 && thisX % imgWidth < imgWidth - ap4)

	if (0 < thisPX <= h * w)
		data[threadNum] = l1[thisPX]
		sync_threads()
		data[threadNum+blockDim().x*blockDim().y] = l2[thisPX]
		data[threadNum] = shouldIProcess * (l2[thisPX] - data[threadNum])
		sync_threads()
		data[threadNum+blockDim().x*blockDim().y] = shouldIProcess * (l3[thisPX] - data[threadNum+blockDim().x*blockDim().y])
		sync_threads()

		out1[thisPX] = data[threadNum]
		out2[thisPX] = data[threadNum+blockDim().x*blockDim().y]
		# out1[thisPX] = shouldIProcess*(l2[thisPX] - l1[thisPX])
		# out2[thisPX] = shouldIProcess*(l3[thisPX] - l2[thisPX])
	end
	return
end
