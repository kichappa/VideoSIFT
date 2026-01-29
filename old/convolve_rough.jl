# import OpenCV.getGaussianKernel
using OpenCV
using CUDA
# using Images

function gaussian_kernel_unopt(inp, conv, out, width, height, apron)
    x, y = threadIdx().x, threadIdx().y
    X, Y = blockIdx().x, blockIdx().y

    if x <= width - 2 * apron && y <= height - 2 * apron
        out[x, y] = sum(inp[x:x+2*apron, y:y+2*apron] .* conv)
    end
    return
end

function row_kernel(inp, conv, out, width, height, apron, print)
    blockNum::UInt16 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based

    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # if threadNum == 5 && blockNum == 0 && print == 1
    #     @cuprintln("threads: ", threads)
    #     # @cuprintln("threadNum: ", threadNum)
    #     # @cuprintln("blockNum: ", blockNum)
    #     # @cuprintln("sharedMemDim: ", (width, threads ÷ width))
    #     # @cuprintln("TypeOf threads: ", typeof(threads))
    # end

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisX::Int16 = 0 # one-based
    thisY::Int16 = 0 # one-based
    thisPX::Int32 = 0 # zero-based


    # Let's do the row first
    # if thread count is greater than one row
    if threads >= width
        # we'll do (width) * (threads ÷ width) pixels in a block
        data = CuDynamicSharedArray(Float32, (width, threads ÷ width))
        # outDatas = CuDynamicSharedArray(Float32, threads)
        # data = @cuStaticSharedMem(Float32, 5)

        if threadNum < width * (threads ÷ width)
            thisPX = blockNum * width * (threads ÷ width) + threadNum # zero-based
            if true
                thisX = thisPX % width + 1 # one-based
                thisY = (thisPX - thisX + 1) ÷ width + 1 # one-based

                # if threadNum == 5 && blockNum == 0 && print == 1
                #     @cuprintln("thisX: ", thisX)
                #     @cuprintln("thisY: ", thisY)
                #     @cuprintln("dataXY: ", (threadNum % width + 1, (threadNum - threadNum % width) ÷ width + 1))
                #     # @cuprintln("img: ", inp[thisPX])
                # end
                # coalesced memory access
                if (0 <= thisPX < width * height)
                    X::Int16 = threadNum % width + 1
                    Y::Int16 = (threadNum - (X - 1)) ÷ width + 1
                    data[X, Y] = inp[thisPX+1]
                end
                sync_threads()
            end

            # thatX = thisX - apron # one-based
            # thatY = thisY # one-based
            if apron < threadNum % width + 1 <= width - apron && 0 <= thisPX < width * height
                # X = threadNum % width + 1
                # Y = (threadNum - (X - 1)) ÷ width + 1
                outData::Float32 = 0
                # outDatas[threadNum+1] = 0.0
                for i in -apron:apron
                    #     # don't use threadNum here
                    outData += data[threadNum%width+1+i, (threadNum-(X-1))÷width+1] * conv[i+apron+1]
                    # outDatas[threadNum+1] += data[threadNum%width+1+i, (threadNum-(X-1))÷width+1] * conv[i+apron+1]
                    #     # solution to too many resources here would be to allocate a shared memory array that stores thisX and thisY or some variable.
                end
                out[1, threadNum%width+1-apron, blockNum*(threads÷width)+(threadNum-(X-1))÷width+1] = outData
                # out[1, threadNum%width+1-apron, blockNum*(threads÷width)+(threadNum-(X-1))÷width+1] = outDatas[threadNum+1]
            end

        end
    else
        # we'll do threads pixels in a block
        data = CuDynamicSharedArray(Float32, threads)
        # outDatas = CuDynamicSharedArray(Float32, threads)

        # total blocks in a row = width ÷ threads + 1
        # "this" refers to the current pixel in the input image
        thisY = blockNum ÷ (cld(width - 2 * apron, threads - 2 * apron)) + 1 # one-based
        thisX = (blockNum % (cld(width - 2 * apron, threads - 2 * apron))) * (threads - 2 * apron) + threadNum + 1 # one-based

        if 0 < thisX <= width && 0 < thisY <= height
            thisPX = (thisY - 1) * width + (thisX - 1) # zero-based
            data[threadNum+1] = inp[thisPX+1]
        end
        sync_threads()

        if thisX <= width - apron && 0 < thisY <= height && apron <= threadNum < threads - apron
            outData = 0.0
            # outDatas[threadNum+1] = 0.0
            for i in -apron:apron
                outData += data[threadNum+1+i] * conv[i+apron+1]
                # outDatas[threadNum+1] += data[threadNum+1+i] * conv[i+apron+1]
            end
            out[1, thisX-apron, thisY] = outData
            # out[1, thisX-apron, thisY] = outDatas[threadNum+1]
        end
    end
    # if threadNum == 5 && blockNum == 0
    # @cuprintln("row kernel done")
    # end
    return
end

function row_kernel_strips(inp, conv, out, width, height, apron, print)
    blockNum::UInt16 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based

    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # if threadNum == 5 && blockNum == 0 && print == 1
    #     @cuprintln("threads: ", threads)
    #     # @cuprintln("threadNum: ", threadNum)
    #     # @cuprintln("blockNum: ", blockNum)
    #     # @cuprintln("sharedMemDim: ", (width, threads ÷ width))
    #     # @cuprintln("TypeOf threads: ", typeof(threads))
    # end

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisX::Int16 = 0 # one-based
    thisY::Int16 = 0 # one-based
    thisPX::Int32 = 0 # zero-based


    # Let's do the row first
    # if thread count is greater than one row
    # we'll do (width) * (threads ÷ width) pixels in a block
    data = CuDynamicSharedArray(Float32, (width, threads ÷ width))
    # outDatas = CuDynamicSharedArray(Float32, threads)
    # data = @cuStaticSharedMem(Float32, 5)

    if threadNum < width * (threads ÷ width)
        thisPX = blockNum * width * (threads ÷ width) + threadNum # zero-based
        if true
            thisX = thisPX % width + 1 # one-based
            thisY = (thisPX - thisX + 1) ÷ width + 1 # one-based

            # if threadNum == 5 && blockNum == 0 && print == 1
            #     @cuprintln("thisX: ", thisX)
            #     @cuprintln("thisY: ", thisY)
            #     @cuprintln("dataXY: ", (threadNum % width + 1, (threadNum - threadNum % width) ÷ width + 1))
            #     # @cuprintln("img: ", inp[thisPX])
            # end
            # coalesced memory access
            if (0 <= thisPX < width * height)
                X::Int16 = threadNum % width + 1
                Y::Int16 = (threadNum - (X - 1)) ÷ width + 1
                data[X, Y] = inp[thisPX+1]
            end
            sync_threads()
        end

        # thatX = thisX - apron # one-based
        # thatY = thisY # one-based
        if apron < threadNum % width + 1 <= width - apron && 0 <= thisPX < width * height
            # X = threadNum % width + 1
            # Y = (threadNum - (X - 1)) ÷ width + 1
            outData::Float32 = 0
            # outDatas[threadNum+1] = 0.0
            for i in -apron:apron
                #     # don't use threadNum here
                outData += data[threadNum%width+1+i, (threadNum-(X-1))÷width+1] * conv[i+apron+1]
                # outDatas[threadNum+1] += data[threadNum%width+1+i, (threadNum-(X-1))÷width+1] * conv[i+apron+1]
                #     # solution to too many resources here would be to allocate a shared memory array that stores thisX and thisY or some variable.
            end
            out[1, threadNum%width+1-apron, blockNum*(threads÷width)+(threadNum-(X-1))÷width+1] = outData
            # out[1, threadNum%width+1-apron, blockNum*(threads÷width)+(threadNum-(X-1))÷width+1] = outDatas[threadNum+1]
        end

    end
    # if threadNum == 5 && blockNum == 0
    # @cuprintln("row kernel done")
    # end
    return
end

function row_kernel_strip(inp, conv, out, width, height, apron, print)
    blockNum::UInt16 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based

    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisX::Int16 = 0 # one-based
    thisY::Int16 = 0 # one-based
    thisPX::Int32 = 0 # zero-based


    # Let's do the row first
    # we'll do threads pixels in a block
    data = CuDynamicSharedArray(Float32, threads)

    # total blocks in a row = width ÷ threads + 1
    # "this" refers to the current pixel in the input image
    thisY = blockNum ÷ (cld(width - 2 * apron, threads - 2 * apron)) + 1 # one-based
    thisX = (blockNum % (cld(width - 2 * apron, threads - 2 * apron))) * (threads - 2 * apron) + threadNum + 1 # one-based

    if 0 < thisX <= width && 0 < thisY <= height
        thisPX = (thisY - 1) * width + (thisX - 1) # zero-based
        data[threadNum+1] = inp[thisPX+1]
    end
    sync_threads()

    if thisX <= width - apron && 0 < thisY <= height && apron <= threadNum < threads - apron
        outData = 0.0
        for i in -apron:apron
            outData += data[threadNum+1+i] * conv[i+apron+1]
        end
        out[1, thisX-apron, thisY] = outData
    end
    return
end
function col_kernel(inp, conv, out, width, fullHeight, height, apron)
    blockNum = (blockIdx().x - 1) * gridDim().y + (blockIdx().y - 1) # row first block numbering, zero-based
    threadNum = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # row first thread numbering in a block, zero-based
    threads = blockDim().x * blockDim().y # total number of threads in a block
    threadsX = blockDim().x

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    if threads <= fullHeight
        data = CuDynamicSharedArray(Float32, (threadsX, threads ÷ threadsX))

        blocksInARow::Int16 = cld(width - 2 * apron, threadsX) # this is the number of blocks in a row of the image (width)
        blocksInAColumn::Int16 = cld(height - 2 * apron, threads ÷ threadsX - 2 * apron) # this is the number of blocks in a column of the image (height)
        blocksInAnImage::Int32 = blocksInARow * blocksInAColumn

        thisImage::Int8 = blockNum ÷ blocksInAnImage # zero-based
        thisBlockNum::Int16 = blockNum % blocksInAnImage # zero-based

        thisX::Int16 = (thisBlockNum ÷ blocksInAColumn) * threadsX + threadNum % threadsX + 1 # one-based
        thisY::Int32 = thisImage * height + (thisBlockNum % blocksInAColumn) * (threads / threadsX - 2 * apron) + threadNum ÷ threadsX + 1 # one-based

        thisPX::Int32 = (thisY - 1) * (width - 2 * apron) + (thisX - 1) # zero-based

        if 0 <= thisPX < (width - 2 * apron) * fullHeight
            data[threadNum+1] = inp[thisPX+1]
        end
        sync_threads()

        if 0 < thisX <= (width - 2 * apron) && apron < thisY <= fullHeight - apron && apron <= (threadNum ÷ threadsX) < threads / threadsX - apron
            outData = 0.0
            for i in -apron:apron
                outData += data[threadNum+i*threadsX+1] * conv[i+apron+1]
            end
            out[1, thisX, thisY-(thisImage)*2*apron-apron] = outData
        end
    end
    return
end

function convolve(img, schema, imgHeight, print=1)
    # function convolve(schema)
    if schema[:name] == "gaussian1D"
        sigma = convert(Float64, schema[:sigma])
        epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
        apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
        conv = reshape(OpenCV.getGaussianKernel(2 * apron + 1, sigma), 2 * apron + 1)

        if print == 1
            println("Convolve with Gaussian1D")
            println("Sigma: ", sigma)
            println("Epsilon: ", epsilon)
            println("Apron: ", apron)
            # println("Grids: ", grids)
        end
        width::Int32, height::Int32 = (0, 0)
        if length(size(img)) == 2
            width, height = size(img)
        else
            channel, width, height = size(img)
        end

        if print == 1
            println("Width: ", width, ", Height: ", height)
        end

        blocks_row = (32, 32)
        blocks_col = (32, 28)
        while blocks_col[2] - 2 * apron < 0 && blocks_col[1] > 4
            blocks_col = (blocks_col[1] ÷ 2, blocks_col[2] * 2)
        end
        # grids = (cld(width, blocks[1]), cld(height, blocks[2]))
        grids_row = cld((width - 2 * apron), blocks_row[1] * blocks_row[2] - 2 * apron) * height
        # grids_col = cld((height - 2 * apron), blocks[2] - 2 * apron) * cld(width - 2 * apron, blocks[1]) * height ÷ imgHeight
        grids_col = (cld(width - 2 * apron, blocks_col[1]), cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgHeight)
        # grids = 2
        if print == 1
            println("Blocks: ", blocks_row)
            println("Blocks: ", blocks_col)
            println("Grids, col: $grids_col, row: $grids_row")
        end
        # sharedMemSize_col = blocks[1] * blocks[2] * (sizeof(Float32))
        inp_GPU = CuArray(img)
        conv_GPU = CuArray(conv)
        out_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height))
        sharedMemSize = blocks_row[1] * blocks_row[2] * (sizeof(Float32)) #+ 1024 * sizeof(Float32)
        if blocks_row[1] * blocks_row[2] >= width
            # sharedMemSize_row = width * ((blocks[1] * blocks[2]) ÷ width) * (sizeof(Float32))
            if print == 1
                println("more than one row in a block")
                # println("Shared memory size: ", sharedMemSize_row / 1024, " KB")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize row_kernel_strips(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            CUDA.unsafe_free!(inp_GPU)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize col_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgHeight, apron)
        else
            # sharedMemSize_row = blocks[1] * blocks[2] * (sizeof(Float32))
            if print == 1
                println("many blocks in a row")
                # println("Shared memory size: ", sharedMemSize_row / 1024, " KB")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            # @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize row_kernel(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            kernel = @cuda name = "row" launch = false row_kernel_strip(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            # config = launch_configuration(kernel.fun)
            # println("Config: ", config)
            kernel(inp_GPU, conv_GPU, out_GPU, width, height, apron, print, blocks=grids_row, threads=blocks_row, shmem=sharedMemSize)
            CUDA.unsafe_free!(inp_GPU)
            inp_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height - 2 * apron * height ÷ imgHeight))
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize col_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgHeight, apron)
            # kernel = @cuda name = "col" launch = false col_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgHeight, apron)
            # config = launch_configuration(kernel.fun)
            # println("Config: ", config)
            # kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgHeight, apron, blocks=grids_col, threads=blocks_col, shmem=sharedMemSize)
        end
        if print == 1
            println("Done")
        end
        return Array(inp_GPU), Array(out_GPU)
        return 1, 2
    end
end

# img = rand(1500, 512)
# load from assets/lenna.png, use OpenCV
# img = OpenCV.imread("assets/lenna.png", OpenCV.IMREAD_GRAYSCALE)
img1 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_1.png", OpenCV.IMREAD_GRAYSCALE)
img2 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_2.png", OpenCV.IMREAD_GRAYSCALE)
img3 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_3.png", OpenCV.IMREAD_GRAYSCALE)
img4 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_4.png", OpenCV.IMREAD_GRAYSCALE)
# img5 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_5.png", OpenCV.IMREAD_GRAYSCALE)
# img6 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_6.png", OpenCV.IMREAD_GRAYSCALE)
# img7 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_7.png", OpenCV.IMREAD_GRAYSCALE)
# img8 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_8.png", OpenCV.IMREAD_GRAYSCALE)
# img9 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_9.png", OpenCV.IMREAD_GRAYSCALE)
# img10 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_10.png", OpenCV.IMREAD_GRAYSCALE)
# img11 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_11.png", OpenCV.IMREAD_GRAYSCALE)
# img12 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_12.png", OpenCV.IMREAD_GRAYSCALE)
# img13 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_13.png", OpenCV.IMREAD_GRAYSCALE)
# img14 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_14.png", OpenCV.IMREAD_GRAYSCALE)
# img15 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_15.png", OpenCV.IMREAD_GRAYSCALE)
# img16 = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_16.png", OpenCV.IMREAD_GRAYSCALE)
println(size(img1))

imgHeight = size(img1, 3)

start = time()
img1 = convert(Array{Float32}, img1)
img2 = convert(Array{Float32}, img2)
img3 = convert(Array{Float32}, img3)
img4 = convert(Array{Float32}, img4)
# img5 = convert(Array{Float32}, img5)
# img6 = convert(Array{Float32}, img6)
# img7 = convert(Array{Float32}, img7)
# img8 = convert(Array{Float32}, img8)
# img9 = convert(Array{Float32}, img9)
# img10 = convert(Array{Float32}, img10)
# img11 = convert(Array{Float32}, img11)
# img12 = convert(Array{Float32}, img12)
# img13 = convert(Array{Float32}, img13)
# img14 = convert(Array{Float32}, img14)
# img15 = convert(Array{Float32}, img15)
# img16 = convert(Array{Float32}, img16)

println("Time taken to convert to Float32: ", time() - start)
println(size(img1))
# img = img1
start = time()
img = cat(img1, img2, img3, img4, dims=3)#, img5, img6, img7, img8, dims=2)#, img9, img10, img11, img12, img13, img14, img15, img16, dims=2)
println("Time taken to concatenate: ", time() - start)
mat_image = OpenCV.Mat(img)
OpenCV.imwrite("assets/lenna_gaussian_before.png", mat_image)

schema = Dict(:name => "gaussian1D", :sigma => 1.6, :epsilon => 0.1725)
# convolve(reshape(img, size(img)...), schema, convert(Int32, imgHeight), 1)
out2, out1 = convolve(img, schema, imgHeight, 1)

schema1 = Dict(:name => "gaussian1D", :sigma => 1.6, :epsilon => 0.1725)
schema2 = Dict(:name => "gaussian1D", :sigma => 2.2627, :epsilon => 0.1725)
schema3 = Dict(:name => "gaussian1D", :sigma => 3.2, :epsilon => 0.1725)
schema4 = Dict(:name => "gaussian1D", :sigma => 4.5254, :epsilon => 0.1725)
schema5 = Dict(:name => "gaussian1D", :sigma => 6.4, :epsilon => 0.1725)
time_taken = 0
for i in 1:100
    start_t = time()
    out2, out1 = convolve(img, schema1, imgHeight, 0)
    out2, out1 = convolve(img, schema2, imgHeight, 0)
    out2, out1 = convolve(img, schema3, imgHeight, 0)
    out2, out1 = convolve(img, schema4, imgHeight, 0)
    out2, out1 = convolve(img, schema5, imgHeight, 0)
    end_t = time()
    global time_taken += end_t - start_t
    if i == 100
        mat_image1 = OpenCV.Mat(out1)
        mat_image2 = OpenCV.Mat(out2)
        OpenCV.imwrite("assets/gaussian_1.png", mat_image1)
        OpenCV.imwrite("assets/gaussian_2.png", mat_image2)
    end
end
println("NO L2 CACHE: Time taken per iteration: ", time_taken / 100, " seconds")
# # println(out)
# println(size(out))
# # save to assets/lenna_gaussian.png
# # OpenCV.imwrite("assets/lenna_gaussian.png", out)
# # uint8_array = UInt8.(round.(clamp.(out, 0, 1) .* 255))

# # Convert the uint8 array to an OpenCV Mat
# # out = reshape(out, 1, size(out, 1), size(out, 3))
mat_image1 = OpenCV.Mat(out1)
mat_image2 = OpenCV.Mat(out2)

OpenCV.imwrite("assets/lenna_gaussian_1.png", mat_image1)
OpenCV.imwrite("assets/lenna_gaussian_2.png", mat_image2)
# # println(out)