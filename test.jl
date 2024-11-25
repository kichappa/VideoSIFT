using CUDA

function kernel(a)
    CUDA.atomic_add!(pointer(a, 1), 1)
    return
end

a = 1
b = CuArray([a])
@cuda threads = 5 blocks = 1 kernel(b)

println(collect(b))
println(a)