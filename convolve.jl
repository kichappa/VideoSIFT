using CUDA, Images, FileIO
import OpenCV.getGaussianKernel
using DelimitedFiles

function getApron(schema)
    if typeof(schema) == Dict{Symbol,Any}
        sigma = convert(Float64, schema[:sigma])
        epsilon = haskey(schema, :epsilon) ? schema[:epsilon] : 0.0001
        apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
        return apron
    else
        aprons = Int8[]
        for i in eachindex(schema)
            sigma = convert(Float64, schema[i][:sigma])
            epsilon = haskey(schema[i], :epsilon) ? schema[i][:epsilon] : 0.0001
            apron = ceil(Int, sigma * sqrt(-2 * log(epsilon)))
            push!(aprons, apron)
        end
        return aprons
    end
end

function getSchemas(schemaBase, sigma, s, layers)
    schemas = []
    for i in 1:layers
        newSchema = copy(schemaBase)
        newSchema[:sigma] = Float64(round(sigma * s^(i - 1), digits=4))
        push!(schemas, newSchema)
    end
    return schemas
end

function col_kernel_strips(inp, conv, buffer, width::Int32, height::Int16, apron::Int8)
    let
        blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
        threadNum::UInt16 = threadIdx().x - 1
        threads::Int16 = blockDim().x

        # there could be more blocks than needed
        thisX::Int32 = blockNum ÷ cld((height - 2 * apron), (threads - 2 * apron)) + 1 # 1-indexed
        thisY::Int16 = blockNum % cld((height - 2 * apron), (threads - 2 * apron)) * (threads - 2 * apron) + threadNum + 1 # 1-indexed
        thisPX::Int32 = 0

        data = CuDynamicSharedArray(Float32, threads)

        # fill the shared memory
        if thisY <= height && thisX <= width
            thisPX = thisY + (thisX - 1) * height
            data[threadNum+1] = inp[thisPX]
        end
        sync_threads()

        # convolution
        if apron < thisY <= height - apron && thisX <= width && apron <= threadNum < threads - apron
            sum::Float32 = 0.0
            for i in -apron:apron
                sum += data[threadNum+1+i] * conv[apron+1+i]
            end
            buffer[thisY, thisX] = sum
        end
    end
    return
end

# buffH is the height of the buffer including the black apron at the bottom
# inpH is the height of the image excluding the aprons, after the column kernel
function row_kernel(inp, conv, out, inpH::Int16, buffH::Int16, width::Int32, imgWidth::Int16, apron::Int8)
    blockNum::UInt32 = blockIdx().x - 1 + (blockIdx().y - 1) * gridDim().x # block number, column major, 0-indexed
    threadNum::UInt16 = threadIdx().x - 1 + (threadIdx().y - 1) * blockDim().x
    threads::Int16 = blockDim().x * blockDim().y

    if threads <= width

        blocksInACol::Int8 = cld(inpH, blockDim().x)
        blocksInARow::Int16 = cld(imgWidth - 2 * apron, blockDim().y - 2 * apron)
        blocksInAnImage::Int16 = blocksInACol * blocksInARow
        # #             |  number of images to the left * imgWidth |   blockNum wrt this image ÷ blocksInAColumn   * thrds in x   | number of threads on the left|
        # thisX::Int32 = fld(blockNum, blocksInAnImage) * imgWidth + fld(blockNum % blocksInAnImage, blocksInACol) * blockDim().y + threadIdx().y # 1-indexed
        # thisY::Int16 = blockNum % blocksInACol * blockDim().x + threadIdx().x # 1-indexed

        # thisImage::Int8 = blockNum ÷ blocksInAnImage # 0-indexed
        # thisBlockNum::Int16 = blockNum % blocksInAnImage # 0-indexed

        thisX::Int32 = (blockNum ÷ blocksInAnImage) * imgWidth + ((blockNum % blocksInAnImage) % blocksInARow) * (blockDim().y - 2 * apron) + threadIdx().y # 1-indexed
        thisY::Int16 = ((blockNum % blocksInAnImage) ÷ blocksInARow) * blockDim().x + threadIdx().x + apron # 1-indexed

        data = CuDynamicSharedArray(Float32, (blockDim().x, blockDim().y))

        # fill the shared memory
        thisPX::Int32 = thisY + (thisX - 1) * buffH
        if thisX <= width && thisY <= inpH + apron
            data[threadNum+1] = inp[thisPX]
        end
        sync_threads()

        # if threadNum==0 && blockNum==0
        #     @cuprintln("Size of inp: $(size(inp)), size of out: $(size(out)), size of data: $(size(data))")
        # end

        # convolution
        thisIsAComputationThread::Bool = thisY <= inpH + apron && apron < thisX <= width - apron && apron < threadIdx().y <= blockDim().y - apron
        # if thisY == 1073 && apron==6 && thisX > 3900
        #     @cuprintln("isThisAComputationThread: $(thisIsAComputationThread), thisX: $thisX)")
        # end
        if (blockNum % blocksInAnImage) % blocksInARow == blocksInARow - 1
            thisIsAComputationThread = thisIsAComputationThread && (thisX - (blockNum ÷ blocksInAnImage) * imgWidth <= imgWidth - 2 * apron)
        end
        if thisIsAComputationThread
            sum::Float32 = 0.0
            for i in -apron:apron
                sum += data[threadNum+1+i*blockDim().x] * conv[apron+1+i]
            end
            # out[thisY, thisX-apron-fld(blockNum, blocksInAnImage)*2*apron] = sum
            out[thisY, thisX] = sum
        end
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
    if outPX <= (h * w) ÷ 4
        out[outPX] = data[threadNum+1]
    end
    return
end

function doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    time_taken = 0
    for j in 1:octaves
        # println("performing octave $j")
        for i in 1:layers
            # assuming height <= 1024
            threads_column = 1024 #32 * 32
            threads_row = (16, 768 ÷ 16)
            while threads_row[2] - 2 * aprons[i] <= 0 && threads_row[1] > 4
                threads_row = (threads_row[1] ÷ 2, threads_row[2] * 2)
            end
            # println("threads_column: $threads_column, threads_row: $threads_row")
            # println(cld(height, prod(threads_column)))
            if cld(height, prod(threads_column)) >= 1
                blocks_column = makeThisNearlySquare((cld(height - 2 * aprons[i], threads_column - 2 * aprons[i]), width))
                # println("org_blocks_column: $((cld(height-2*aprons[i], threads_column-2*aprons[i]), width))")
                # println("blocks_column: $blocks_column")
                blocks_row = makeThisNearlySquare((cld(height - 2 * aprons[i], threads_row[1]) * cld(width - 2 * aprons[i], threads_row[2] - 2 * aprons[i]) + cld(height - 2 * aprons[i], threads_row[1]) / 2 * cld(imgWidth - 2 * aprons[i], threads_row[2] - 2 * aprons[i]), 1))
                # println("blocks_row: $blocks_row")  
                shmem_column = threads_column * sizeof(Float32)
                shmem_row = threads_row[1] * threads_row[2] * sizeof(Float32)


                time_taken += CUDA.@elapsed buffer .= 0
                time_taken += CUDA.@elapsed @cuda threads = threads_column blocks = blocks_column shmem = shmem_column col_kernel_strips(img_gpu, conv_gpus[i], buffer, Int32(width), Int16(height), Int8(aprons[i]))
                # kernel = @cuda name = "col" launch = false col_kernel_strips(img_gpu, conv_gpus[1], buffer, Int32(width), Int16(height), Int8(aprons[i]))
                # println(launch_configuration(kernel.fun))
                # kernel = @cuda name = "row" launch = false row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
                # println(launch_configuration(kernel.fun))
                # println("h-2ap:$(Int16(height - 2 * aprons[i])), h: $(Int16(height)), w: $(Int32(width)), imW: $(Int16(imgWidth)), apron: $(Int8(aprons[i]))")
                time_taken += CUDA.@elapsed @cuda threads = threads_row blocks = blocks_row shmem = shmem_row row_kernel(buffer, conv_gpus[i], out_gpus[j][i], Int16(height - 2 * aprons[i]), Int16(height), Int32(width), Int16(imgWidth), Int8(aprons[i]))
                # save("assets/gaussian_new_o$(j)_l$(i)_r.png", colorview(Gray, collect(buffer)))
                # save("assets/gaussian_new_o$(j)_l$(i)_rc.png", colorview(Gray, collect(out_gpus[j][i])))
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

function makeThisNearlySquare(blocks)
    product = blocks[1] * blocks[2]
    X = floor(Int32, sqrt(product))
    Y = X
    while product % X != 0 && X / Y > 0.75
        X -= 1
    end

    if product % X == 0
        return Int32.((X, product ÷ X))
    else
        return Int32.((Y, cld(product, Y)))
    end
end

let
    println("Here we go!")
    nImages = 1
    img = []
    imgWidth = 0
    time_taken = 0
    # load the images
    for i in 1:nImages
        img_temp = Float32.(Gray.(FileIO.load("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_$i.png")))
        if i == 1
            img = img_temp
            imgWidth = size(img, 2)
        else
            img = cat(img, img_temp, dims=2)
        end
    end

    height, width = size(img)
    println(size(img))
    save("assets/gaussian_new_0.png", colorview(Gray, collect(img)))

    schemaBase = Dict(:name => "gaussian1D", :epsilon => 0.1725)

    layers = 3
    octaves = 4
    schemas = getSchemas(schemaBase, 1.6, sqrt(2), layers)
    aprons = getApron(schemas)

    # create GPU elements
    img_gpu = CuArray(img)
    # buffer_resample = CUDA.zeros(Float32, height ÷ 2, width ÷ 2)
    # @cuda threads = 1024 blocks = makeThisNearlySquare((cld(height * width ÷ 4, 1024), 1)) shmem=1024*sizeof(Float32) resample_kernel(img_gpu, buffer_resample)
    # save("assets/resample.png", colorview(Gray, Array(buffer_resample)))

    buffer = CUDA.zeros(Float32, height, width)
    conv_gpus = []
    out_gpus = []
    for j in 1:octaves
        out_gpus_octave = []
        for i in 1:layers
            # out_gpu = CUDA.zeros(Float32, height - 2 * aprons[i], width - 2 * nImages * aprons[i])
            out_gpu = CUDA.zeros(Float32, cld(height, (2^(j - 1))), cld(width, (2^(j - 1))))
            push!(out_gpus_octave, out_gpu)
            if j == 1
                kernel = reshape(getGaussianKernel(2 * aprons[i] + 1, schemas[i][:sigma]), 2 * aprons[i] + 1)
                push!(conv_gpus, CuArray(kernel))
            end
        end
        push!(out_gpus, out_gpus_octave)
    end


    # i = 1
    # warmup_inp = CUDA.rand(Float32, 1080, 1920)
    # warmupout_gpus = []
    # for i in 1:layers
    #     warmupout_gpu = CUDA.zeros(Float32, 1080 - 2 * aprons[i], 1920 - 2 * aprons[i])
    #     push!(warmupout_gpus, warmupout_gpu)
    # end
    doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    println("Warmup done!")
    iterations = 10
    for i in 1:iterations
        time_taken += doLayersConvolvesAndDoGAndOctave(img_gpu, out_gpus, buffer, conv_gpus, aprons, height, width, imgWidth, layers, octaves)
    end
    println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time")
    for j in 1:octaves
        for i in 1:(layers-1)
            # save("assets/gaussian_new_$([i])_1.png", colorview(Gray, collect(buffer)))
            # save("assets/gaussian_new_$([i]).png", colorview(Gray, collect(out_gpus[j][i])))
            save("assets/DoG_o$(j)l$(i).png", colorview(Gray, Array(out_gpus[j][i])))
            # out = collect(out_gpus[j][i])
            # save("assets/DoG_$([i]).txt", collect(out_gpus[j][i]))
            # writedlm("assets/DoG_$([i]).csv", Array(out_gpus[j][i]), ',')
        end
    end
    # println(aprons)
end