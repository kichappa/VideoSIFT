using CUDA

function kernel(a, b, c)
    point1, point2 = let
        if c == 1
            a, b
        else
            b, a
        end
    end
    point1[2, 2] = 5
    point2[2, 2] = 6
    return
end

function kernel2(a, b)
    c = CuDynamicSharedArray(Float32, 2)
    d = CuDynamicSharedArray(Float32, 2, sizeof(Float32)*2)
    c[1] = a
    c[2] = b
    d[1] = b
    d[2] = a
    @cuprintln(c[1], c[2])
    @cuprintln(d[1], d[2])
    return
end

# define a 2x2 matrix {{1, 2}, {3, 4}}
# a = CuArray([1 2; 3 4])
# b = CuArray([1 2; 3 4])
# @cuda threads = 5 blocks = 1 kernel(a, b, 1)

# println(collect(a))
# println(collect(b))

a = 1
b = 2
@cuda threads = 1 blocks = 1 shmem=sizeof(Float32)*4 kernel2(a, b)