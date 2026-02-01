using CUDA

# Force CUDA initialization before ncu starts profiling
CUDA.device()
temp = CUDA.zeros(Float32, 1, 1)  # Initialize CUDA context
CUDA.synchronize()

CUDA.@profile external=true begin
    # Now include your main script
    include("main.jl")
end