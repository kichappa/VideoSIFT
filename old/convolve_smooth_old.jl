using OpenCV
using CUDA
using FileIO
using Images

function col_kernel_strips(inp, conv, out, height, width, apron, print)
    blockNum::UInt32 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based

    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisY::Int16 = 0 # one-based
    thisX::Int32 = 0 # one-based
    thisPX::Int32 = 0 # zero-based

    # Let's do the row first
    # if thread count is greater than one row
    # we'll do (height) * (threads ÷ height) pixels in a block
    data = CuDynamicSharedArray(Float32, (height, threads ÷ height))

    if threadNum < height * (threads ÷ height)
        thisPX = blockNum * height * (threads ÷ height) + threadNum # zero-based
        if true
            thisY = thisPX % height + 1 # one-based
            thisX = (thisPX - thisY + 1) ÷ height + 1 # one-based
            # coalesced memory access
            if (0 <= thisPX < height * width)
                X::Int16 = threadNum % height + 1
                Y::Int16 = (threadNum - (X - 1)) ÷ height + 1
                data[X, Y] = inp[thisPX+1]
            end
            sync_threads()
        end
        # thatX = thisY - apron # one-based
        # thatY = thisX # one-based
        if apron < threadNum % height + 1 <= height - apron && 0 <= thisPX < height * width
            outData::Float32 = 0
            for i in -apron:apron
                outData += data[threadNum%height+1+i, (threadNum-(X-1))÷height+1] * conv[i+apron+1]
            end
            out[threadNum%height+1-apron, blockNum*(threads÷height)+(threadNum-(X-1))÷height+1] = outData
        end

    end
    return
end

function col_kernel_strip(inp, conv, out, width, height, apron, print)
    blockNum::UInt32 = (blockIdx().x - 1) + (blockIdx().y - 1) * gridDim().x # column major block numbering, zero-based
    threadNum::UInt16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column major thread numbering in a block, zero-based
    # not sure why unsigned uint16 doesn't work
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    thisY::Int16 = 0 # one-based
    thisX::Int32 = 0 # one-based
    thisPX::Int32 = 0 # zero-based

    # Let's do the row first
    # we'll do threads pixels in a block
    data = CuDynamicSharedArray(Float32, threads)

    # total blocks in a row = width ÷ threads + 1
    # "this" refers to the current pixel in the input image
    thisX = blockNum ÷ (cld(width - 2 * apron, threads - 2 * apron)) + 1 # one-based
    thisY = (blockNum % (cld(width - 2 * apron, threads - 2 * apron))) * (threads - 2 * apron) + threadNum + 1 # one-based

    if 0 < thisY <= width && 0 < thisX <= height
        thisPX = (thisX - 1) * width + (thisY - 1) # zero-based
        data[threadNum+1] = inp[thisPX+1]
    end
    sync_threads()

    if thisY <= width - apron && 0 < thisX <= height && apron <= threadNum < threads - apron
        outData = 0.0
        for i in -apron:apron
            outData += data[threadNum+1+i] * conv[i+apron+1]
        end
        out[thisY-apron, thisX] = outData
    end
    return
end

function row_kernel(inp, conv, out, height, fullWidth, width, apron, bufferH)
    blockNum::Int32 = (blockIdx().x - 1) * gridDim().y + (blockIdx().y - 1) # row first block numbering, zero-based
    threadNum::Int16 = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # column first thread numbering in a block, zero-based
    threads::Int16 = blockDim().x * blockDim().y # total number of threads in a block
    # threadsX::Int8 = blockDim().x
    # blockNum = (blockIdx().x - 1) * gridDim().y + (blockIdx().y - 1) # row first block numbering, zero-based
    # threadNum = (threadIdx().x - 1) + (threadIdx().y - 1) * blockDim().x # row first thread numbering in a block, zero-based
    # threads = blockDim().x * blockDim().y # total number of threads in a block
    threadsX = blockDim().x

    # "this" refers to the current pixel in the input image
    # "that" refers to the current pixel in the output image
    if threads <= fullWidth
        data = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y))

        # blocksInACol::Int16 = cld(height - 2 * apron, threadsX) # this is the number of blocks in a row of the image (height)
        # blocksInARow::Int32 = cld(width - 2 * apron, threads ÷ threadsX - 2 * apron) # this is the number of blocks in a column of the image (width)
        # blocksInAnImage::Int32 = blocksInACol * blocksInARow

        # thisImage::Int8 = blockNum ÷ blocksInAnImage # zero-based
        # thisBlockNum::Int32 = blockNum % blocksInAnImage # zero-based

        # thisY::Int16 = (thisBlockNum ÷ blocksInARow) * threadsX + threadNum % threadsX + 1 # one-based
        # thisX::Int32 = thisImage * width + (thisBlockNum % blocksInARow) * (threads ÷ threadsX - 2 * apron) + threadNum ÷ threadsX + 1 # one-based

        # thisPX::Int32 = (thisX - 1) * (height - 2 * apron) + (thisY - 1) # zero-based

        blocksInACol::Int16 = cld(height - 2 * apron, blockDim().x) # this is the number of blocks in a row of the image (height)
        blocksInARow::Int32 = cld(width - 2 * apron, blockDim().y - 2 * apron) # this is the number of blocks in a column of the image (width)
        blocksInAnImage::Int32 = blocksInACol * blocksInARow

        thisImage::Int8 = blockNum ÷ blocksInAnImage # zero-based
        thisBlockNum::Int32 = blockNum % blocksInAnImage # zero-based

        thisX::Int32 = thisImage * width + (thisBlockNum % blocksInARow) * (threads ÷ blockDim().x - 2 * apron) + threadNum ÷ blockDim().x + 1 # one-based
        thisY::Int16 = (thisBlockNum ÷ blocksInARow) * blockDim().x + threadNum % blockDim().x + 1 # one-based

        thisPX::Int32 = (thisX - 1) * (height - 2 * apron) + (thisY - 1) # zero-based
        bufferthisPx::Int32 = (thisX - 1) * (bufferH) + (thisY - 1) # zero-based
        # if threadNum == 0 && blockNum == 0
        #     @cuprintln("type of fullWidth: $(typeof(fullWidth))")
        # end
        # if thisBlockNum == 0
        #     @cuprintln("thisPx: $thisPX, thisX: $thisX, thisY: $thisY, thisBlockNum: $thisBlockNum, thisImage: $thisImage, blockNum: $blockNum, threadNum: $threadNum, ")
        # end
        # blocksInACol = cld(height - 2 * apron, threadsX) # this is the number of blocks in a row of the image (height)
        # blocksInARow = cld(width - 2 * apron, threads ÷ threadsX - 2 * apron) # this is the number of blocks in a column of the image (width)
        # blocksInAnImage = blocksInACol * blocksInARow

        # thisImage = blockNum ÷ blocksInAnImage # zero-based
        # thisBlockNum = blockNum % blocksInAnImage # zero-based

        # thisY = (thisBlockNum ÷ blocksInARow) * threadsX + threadNum % threadsX + 1 # one-based
        # thisX = thisImage * width + (thisBlockNum % blocksInARow) * (threads ÷ threadsX - 2 * apron) + threadNum ÷ threadsX + 1 # one-based

        # thisPX = (thisX - 1) * (height - 2 * apron) + (thisY - 1) # zero-based

        if 0 <= thisPX < (height - 2 * apron) * fullWidth
            # data[threadNum+1] = 0
            data[threadNum+1] = inp[bufferthisPx+1]
        end
        sync_threads()

        if threadNum == 1 && blockNum == 1
            @cuprintln("blockDim: ($(blockDim().x), $(blockDim().y)), gridDim: ($(gridDim().x), $(gridDim().y))")
            @cuprintln("threads: $threads, threadsX: $(blockDim().x), blocksInACol: $blocksInACol, blocksInARow: $blocksInARow, blocksInAnImage: $blocksInAnImage")
        end
        if 0 < thisY <= (height - 2 * apron) && apron < thisX <= fullWidth - apron && apron <= (threadNum ÷ threadsX) < threads ÷ threadsX - apron
            outData = 0.0
            for i in -apron:apron
                outData += data[threadNum+i*threadsX+1] * conv[i+apron+1]
            end
            out[thisY, thisX-(thisImage)*2*apron-apron] = outData
        end
        # out[1, 1] = 0
    end
    # if threadNum == 1 && blockNum == 1
    #     @cuprintln("Hello from netherspace")
    # end
    return
end

function convolve(img, schema, imgWidth, print=1)
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
        end
        width::Int32, height::Int32 = (0, 0)
        if length(size(img)) == 2
            width, height = size(img)
        else
            _, width, height = size(img)
        end

        if print == 1
            println("Width: ", width, ", Height: ", height)
        end

        blocks_row = (32, 32)
        blocks_col = (32, 32)
        # blocks_col = (32, 28)
        while blocks_col[2] - 2 * apron < 0 && blocks_col[1] > 4
            blocks_col = (blocks_col[1] ÷ 2, blocks_col[2] * 2)
        end
        if print == 1
            println("$(height - 2 * apron), $(blocks_col[2]), $(blocks_col[2] - 2 * apron), $(height ÷ imgWidth)")
        end
        grids_row = cld((width - 2 * apron), blocks_row[1] * blocks_row[2] - 2 * apron) * height
        # println("total blocks needed for row: $(cld(width - 2 * apron, blocks_col[1])* (cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth))")
        # println("sqrt of that is: $(sqrt(cld(width - 2 * apron, blocks_col[1])* (cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth)))")
        grids_in_col = floor(Int32, sqrt(cld(width - 2 * apron, blocks_col[1]) * (cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth)))
        while ((cld(width - 2 * apron, blocks_col[1]) * (cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth)) % grids_in_col) != 0
            grids_in_col -= 1
        end
        # grids_col = (cld(width - 2 * apron, blocks_col[1]), cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth)
        grids_col = (cld(cld(width - 2 * apron, blocks_col[1]) * (cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth), grids_in_col), grids_in_col)
        if print == 1
            println("Row Threads: ", blocks_row)
            println("Col Threads: ", blocks_col)
            println("Grids, row: $grids_row, col: $grids_col")
            println("Convolution kernel: ", conv)
        end
        inp_GPU = CuArray(img)
        conv_GPU = CuArray(conv)
        out_GPU = CuArray(zeros(Float32, width - 2 * apron, height))
        sharedMemSize = blocks_row[1] * blocks_row[2] * (sizeof(Float32)) # shared memory size in bytes
        if blocks_row[1] * blocks_row[2] >= width
            if print == 1
                println("more than one row in a block")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize col_kernel_strips(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            # CUDA.unsafe_free!(inp_GPU)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize row_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgWidth, apron)
        else
            if print == 1
                println("many blocks in a row")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize col_kernel_strip(inp_GPU, conv_GPU, out_GPU, width, height, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize row_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgWidth, apron)
            # kernel = @cuda launch = false row_kernel(out_GPU, conv_GPU, inp_GPU, width, height, imgWidth, apron)
            # println(launch_configuration(kernel.fun))
        end
        if print == 1
            println("Done")
        end
        return collect(inp_GPU), collect(out_GPU)
        # return 1, 2
    end
end

function convolves(inp_GPU, out_GPU, buffer, schema, imgWidth, bufferH, print=1)
    if schema[:name] == "gaussian1D"
        sigma = convert(Float64, schema[:sigma])
        apron = getApron(schema)
        conv = reshape(OpenCV.getGaussianKernel(2 * apron + 1, sigma), 2 * apron + 1)

        if print == 1
            println("Convolve with Gaussian1D")
            println("Sigma: ", sigma)
            # println("Epsilon: ", epsilon)
            println("Apron: ", apron)
        end
        height::Int32, width::Int32 = (0, 0)
        if length(size(img)) == 2
            height, width = size(img)
        else
            _, height, width = size(img)
        end

        if print == 1
            println("height: ", height, ", width: ", width)
        end

        blocks_row = (32, 32)
        # blocks_col = (32, 32)
        blocks_col = (32, 16)
        while blocks_col[2] - 2 * apron < 0 && blocks_col[1] > 4
            blocks_col = (blocks_col[1] ÷ 2, blocks_col[2] * 2)
        end
        if print == 1
            println("$(width - 2 * apron), $(blocks_col[2]), $(blocks_col[2] - 2 * apron), $(width ÷ imgWidth)")
        end
        grids_row = cld((height - 2 * apron), blocks_row[1] * blocks_row[2] - 2 * apron) * width
        # println("total blocks needed for row: $(cld(height - 2 * apron, blocks_col[1])* (cld((width - 2 * apron), blocks_col[2] - 2 * apron) * width ÷ imgWidth))")
        # println("sqrt of that is: $(sqrt(cld(height - 2 * apron, blocks_col[1])* (cld((width - 2 * apron), blocks_col[2] - 2 * apron) * width ÷ imgWidth)))")
        grids_in_col = floor(Int32, sqrt(cld(height - 2 * apron, blocks_col[1]) * (cld((width - 2 * apron), blocks_col[2] - 2 * apron) * width ÷ imgWidth)))
        while ((cld(height - 2 * apron, blocks_col[1]) * (cld((width - 2 * apron), blocks_col[2] - 2 * apron) * width ÷ imgWidth)) % grids_in_col) != 0
            grids_in_col -= 1
        end
        # grids_col = (cld(height - 2 * apron, blocks_col[1]), cld((width - 2 * apron), blocks_col[2] - 2 * apron) * width ÷ imgWidth)
        grids_col = (cld(cld(height - 2 * apron, blocks_col[1]) * (cld((width - 2 * apron), blocks_col[2] - 2 * apron) * width ÷ imgWidth), grids_in_col), grids_in_col)
        if print == 1
            println("Row Threads: ", blocks_row)
            println("Col Threads: ", blocks_col)
            println("Grids, row: $grids_row, col: $grids_col")
            println("Convolution kernel: ", conv)
        end
        # inp_GPU = CuArray(img)
        conv_GPU = CuArray(conv)
        # out_GPU = CuArray(zeros(Float32, height - 2 * apron, width))
        sharedMemSize = blocks_row[1] * blocks_row[2] * (sizeof(Float32)) # shared memory size in bytes
        if blocks_row[1] * blocks_row[2] >= height
            if print == 1
                println("more than one row in a block")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize col_kernel_strips(inp_GPU, conv_GPU, out_GPU, height, width, apron, print)
            # CUDA.unsafe_free!(inp_GPU)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize row_kernel(out_GPU, conv_GPU, inp_GPU, height, width, imgWidth, apron, bufferH)
        else
            if print == 1
                println("many blocks in a row")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize col_kernel_strip(inp_GPU, conv_GPU, buffer, height, width, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize row_kernel(buffer, conv_GPU, out_GPU, height, width, imgWidth, apron, bufferH)
            # kernel = @cuda launch = false row_kernel(out_GPU, conv_GPU, inp_GPU, height, width, imgWidth, apron, bufferH)
            # println(launch_configuration(kernel.fun))
        end
        if print == 1
            println("Done")
        end
        return collect(inp_GPU), collect(out_GPU)
        # return 1, 2
    end
end

function convolve1(inp_GPU, schema, imgWidth, width, height, print=1)
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
        end

        if print == 1
            println("Width: ", width, ", Height: ", height)
        end

        blocks_row = (32, 32)
        blocks_col = (32, 28)
        while blocks_col[2] - 2 * apron < 0 && blocks_col[1] > 4
            blocks_col = (blocks_col[1] ÷ 2, blocks_col[2] * 2)
        end
        grids_row = cld((width - 2 * apron), blocks_row[1] * blocks_row[2] - 2 * apron) * height
        grids_col = (cld(width - 2 * apron, blocks_col[1]), cld((height - 2 * apron), blocks_col[2] - 2 * apron) * height ÷ imgWidth)
        if print == 1
            println("Blocks: ", blocks_row)
            println("Blocks: ", blocks_col)
            println("Grids, col: $grids_col, row: $grids_row")
        end
        conv_GPU = CuArray(conv)
        out1_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height))
        out2_GPU = CuArray(zeros(Float32, 1, width - 2 * apron, height - 2 * apron * height ÷ imgWidth))
        sharedMemSize = blocks_row[1] * blocks_row[2] * (sizeof(Float32)) # shared memory size bytes
        if blocks_row[1] * blocks_row[2] >= width
            if print == 1
                println("more than one row in a block")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize col_kernel_strips(inp_GPU, conv_GPU, out1_GPU, width, height, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize row_kernel(out1_GPU, conv_GPU, out2_GPU, width, height, imgWidth, apron)
        else
            if print == 1
                println("many blocks in a row")
                println("Shared memory size: ", sharedMemSize / 1024, " KB")
                println("Convolution kernel: ", conv)
            end
            @cuda blocks = grids_row threads = blocks_row shmem = sharedMemSize col_kernel_strip(inp_GPU, conv_GPU, out1_GPU, width, height, apron, print)
            @cuda blocks = grids_col threads = blocks_col shmem = sharedMemSize row_kernel(out1_GPU, conv_GPU, out2_GPU, width, height, imgWidth, apron)
        end
        if print == 1
            println("Done")
        end
        return Array(out2_GPU), Array(out1_GPU)
        return 1, 2
    end
end

function getApron(schema)
    sigma = convert(Float64, schema[:sigma])
    epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
    apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
    return apron
end

# imgNum = 16
nImages = 12
println("Here we go!")
# for nImages in 1:imgNum
img = []
imgWidth = 0
time_taken = 0
let
    for i in 1:nImages
        # img_temp = OpenCV.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_$i.png", OpenCV.IMREAD_GRAYSCALE)
        # img_temp = convert(Array{Float32}, img_temp)

        img_temp = FileIO.load("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_$i.png")
        img_temp = Float32.(Gray.(img_temp))

        if i == 1
            global img = img_temp
            if length(size(img_temp)) == 2
                global imgWidth = size(img_temp, 2)
            else
                global imgWidth = size(img_temp, 3)
            end
        else
            # global img = cat(img, img_temp, dims=3)
            global img = cat(img, img_temp, dims=2)
        end
    end

    start = time()
    println(typeof(img))
    println("Image size: ", size(img))
    # println(img[1:2, :])
    save("assets/gaussian_before.png", colorview(Gray, img))
    # mat_image = OpenCV.Mat(img)
    # OpenCV.imwrite("assets/gaussian_before.png", mat_image)

    schema = Dict(:name => "gaussian1D", :sigma => 1.6, :epsilon => 0.1725)


    schema1 = Dict(:name => "gaussian1D", :sigma => 1.6, :epsilon => 0.1725)
    schema2 = Dict(:name => "gaussian1D", :sigma => 2.2627, :epsilon => 0.1725)
    schema3 = Dict(:name => "gaussian1D", :sigma => 3.2, :epsilon => 0.1725)
    schema4 = Dict(:name => "gaussian1D", :sigma => 4.5254, :epsilon => 0.1725)
    schema5 = Dict(:name => "gaussian1D", :sigma => 6.4, :epsilon => 0.1725)

    inp_GPU = CuArray(img)
    buffer_GPU = CuArray(zeros(Float32, size(img, 1) - 2 * getApron(schema1), size(img, 2)))
    out_GPU1 = CuArray(zeros(Float32, size(img, 1) - 2 * getApron(schema1), size(img, 2) - 2 * getApron(schema1)))
    out_GPU2 = CuArray(zeros(Float32, size(img, 1) - 2 * getApron(schema2), size(img, 2) - 2 * getApron(schema2)))
    out_GPU3 = CuArray(zeros(Float32, size(img, 1) - 2 * getApron(schema3), size(img, 2) - 2 * getApron(schema3)))
    out_GPU4 = CuArray(zeros(Float32, size(img, 1) - 2 * getApron(schema4), size(img, 2) - 2 * getApron(schema4)))
    out_GPU5 = CuArray(zeros(Float32, size(img, 1) - 2 * getApron(schema5), size(img, 2) - 2 * getApron(schema5)))
    bufferH = size(img, 2)-2*getApron(schema1)
    out2, out1 = convolves(inp_GPU, out_GPU5, buffer_GPU, schema5, imgWidth,  bufferH, 1)

    iterations = 5
    for i in 1:iterations
        # try
        start_t = time()
        out2, out1 = convolves(inp_GPU, out_GPU1, buffer_GPU, schema1, imgWidth, bufferH,  0)
        # end_t = time()
        # println("\tTime taken $i: ", end_t - start_t, " seconds")
        # global time_taken += end_t - start_t
        # start_t = time()
        out2, out1 = convolves(inp_GPU, out_GPU2, buffer_GPU, schema2, imgWidth, bufferH,  0)
        # end_t = time()
        # global time_taken += end_t - start_t
        # println("\tTime taken $i: ", end_t - start_t, " seconds")
        # start_t = time()
        out2, out1 = convolves(inp_GPU, out_GPU3, buffer_GPU, schema3, imgWidth, bufferH,  0)
        # end_t = time()
        # global time_taken += end_t - start_t
        # println("\tTime taken $i: ", end_t - start_t, " seconds")
        # start_t = time()
        out2, out1 = convolves(inp_GPU, out_GPU4, buffer_GPU, schema4, imgWidth, bufferH,  0)
        # end_t = time()
        # global time_taken += end_t - start_t
        # println("\tTime taken $i: ", end_t - start_t, " seconds")
        # start_t = time()
        out2, out1 = convolves(inp_GPU, out_GPU5, buffer_GPU, schema5, imgWidth, bufferH,  0)
        end_t = time()
        global time_taken += end_t - start_t
        # println("\tTime taken $i: ", end_t - start_t, " seconds")
        if i == iterations
            # mat_image1 = OpenCV.Mat(out1)
            # mat_image2 = OpenCV.Mat(out2)
            # OpenCV.imwrite("assets/gaussian_1.png", mat_image1)
            # OpenCV.imwrite("assets/gaussian_2.png", mat_image2)
            save("assets/gaussian_1.png", colorview(Gray, out1))
            save("assets/gaussian_2.png", colorview(Gray, out2))

        end
        # if i % (iterations ÷ 20) == 0
        #     println("Iteration: ", i)
        #     CUDA.memory_status()
        # end
        # if i > 75
        #     println("Iteration: ", i)
        #     CUDA.memory_status()
        # end
        # catch e
        #     println("Error: ", e)
        #     continue
        # end
    end
    println("NO L2 CACHE: Time taken per iteration: ", time_taken / (iterations * nImages), " seconds per image when $nImages images are processed at once")
end