using CUDA, Printf

function getGaussianKernel(ksize, sigma)
    kernel = CUDA.zeros(Float32, ksize)
    kernel = exp.(-0.5 * ((0:ksize-1) .- (ksize - 1) / 2) .^ 2 / sigma^2)
    kernel = kernel ./ sum(kernel)
    return kernel
end

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

function format_number(x)
    if abs(x) == 0 || (abs(x) >= 0.0001 && abs(x) < 100000 && (round(x, digits=5) - x )<= 0.00001)
        return round(x, digits=5)
    else
        return @sprintf("%.4e", x)
    end
end

import Base.println
println(s::String) = begin
    println(stdout, s)
    flush(stdout)
end