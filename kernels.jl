function col_kernel_strips(inp, conv, buffer, width::Int32, height::Int16, apron::Int8)
    let
        blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
        threadNum::UInt16 = threadIdx().x - 1
        # threads::Int16 = blockDim().x

        if blockNum == 0 && threadNum == 0
            @cuprintln("COL: size of inp: $(size(inp)), size of out/buffer: $(size(buffer))")
        end
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

        # if threadNum == 0
        #     @cuprintln("Shared memory filled in block $blockNum")
        # end

        # convolution
        if apron < thisY <= height - apron && thisX <= width && apron <= (threadIdx().x - 1) < (blockDim().x) - apron
            sum::Float32 = 0.0
            for i in -apron:apron
                sum += data[threadNum+1+i] * conv[apron+1+i]
                # sum += data[threadIdx().x+i] * conv[apron+1+i]
            end
            buffer[thisY, thisX] = sum
            # buffer[thisY-apron, thisX-apron] = sum
        end
    end
    return
end

function col_kernel_strips_2(inp, conv, buffer, width::Int32, height::Int16, imgWidth::Int16, iApron::Int8, apron::Int8)
    let
        blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
        threadNum::UInt16 = threadIdx().x - 1
        # threads::Int16 = blockDim().x

        if blockNum == 0 && threadNum == 0
            @cuprint("COL: size of inp: $(size(inp)), size of out/buffer: $(size(buffer))")
        end
        # there could be more blocks than needed
        # thisX::Int32 = blockNum ÷ Int32(cld((height - 2 * apron), (threads - 2 * apron))) + 1 # 1-indexed
        thisX::Int32 = iApron + 2 * iApron * (blockNum ÷ UInt32((imgWidth - 2 * iApron) * cld((height - 2 * apron), (blockDim().x - 2 * apron)))) + blockNum ÷ Int32(cld((height - 2 * apron), (blockDim().x - 2 * apron))) + 1 # 1-indexed
        thisY::Int16 = iApron + (blockNum % cld((height - 2 * apron), (blockDim().x - 2 * apron)) * (blockDim().x - 2 * apron) + threadNum + 1) # 1-indexed
        thisPX::Int32 = 0

        data = CuDynamicSharedArray(Float32, blockDim().x)

        # fill the shared memory
        if iApron < thisY <= height - iApron && iApron < thisX <= width - iApron
            thisPX = thisY + (thisX - 1) * height
            data[threadNum+1] = inp[thisPX]
            # data[threadIdx().x] = inp[thisPX]
            # if blockNum < UInt32(cld((height - 2 * apron), (blockDim().x - 2 * apron)))
            #     @cuprintln("thisX: $thisX, thisY: $thisY, thisPX: $thisPX")
            # end
        end
        sync_threads()

        # if threadNum == 0
        #     @cuprintln("Shared memory filled in block $blockNum")
        # end

        # convolution
        if (apron + iApron) < thisY <= height - (apron + iApron) && iApron < thisX <= width - iApron && apron <= (threadIdx().x - 1) < (blockDim().x) - apron
            sum::Float32 = 0.0
            for i in -apron:apron
                sum += data[threadNum+1+i] * conv[apron+1+i]
                # sum += data[threadIdx().x+i] * conv[apron+1+i]
            end
            buffer[thisY, thisX] = sum
            # buffer[thisY-apron, thisX-apron] = sum
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


    if blockNum == 0 && (threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x) == 0
        @cuprintln("ROW: size of inp: $(size(inp)), size of out: $(size(out))")
    end
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
    thisX::Int32 = iApron + (blockNum ÷ blocksInAnImage) * imgWidth  + fld((blockNum % blocksInAnImage), blocksInACol) * (blockDim().y - 2 * apron) + threadIdx().y # 1-indexed

    data = CuDynamicSharedArray(Float32, threads)

    # fill the shared memory
    begin
        if (iApron + apron) < thisY <= height - (iApron + apron) && iApron < thisX <= width - iApron
            thisPX::Int32 = thisY + (thisX - 1) * height
            data[threadNum+1] = inp[thisPX]
        end
    end
    sync_threads()

    # thisIsAComputationThread::Bool = (iApron + apron) < thisY <= height - (iApron + apron) && (iApron + apron) < thisX <= width - (iApron + apron) && apron < threadIdx().y <= blockDim().y - apron
    # if (blockNum % blocksInAnImage) % blocksInACol == blocksInACol - 1
    #     thisIsAComputationThread = thisIsAComputationThread && (thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth - 2 * (iApron + apron))
    # end
    thisIsAComputationThread::Bool = ((iApron + apron) < thisY <= height - (iApron + apron)) && ((iApron + apron) < thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth - (iApron + apron)) && (apron < threadIdx().y <= blockDim().y - apron) && (iApron + apron) < thisX <= width - (iApron + apron)
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

    # data = CuDynamicSharedArray(Float32, threads)

    # h, w = size(inp)
    if blockNum == 0 && threadNum == 0
        @cuprintln("SMPL: h: $h, w: $w, size of inp: $(size(inp)), size of out: $(size(out))")
    end
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
    # sync_threads()

    # if outPX <= ((h ÷ 4) * (w ÷ 4))
    #     out[outPX] = data[threadNum+1]
    # end
    return
end
