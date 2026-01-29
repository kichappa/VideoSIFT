using Pkg
using InteractiveUtils

# Check versions of the main packages
println("Package versions:")
for pkg in ["CUDA", "GPUArrays", "GPUCompiler", "LLVM"]
	try
		Pkg.status([pkg])
	catch
		println("$pkg not installed")
	end
end

# Get detailed CUDA information
using CUDA
println("\nCUDA details:")
println("CUDA runtime version: ", CUDA.runtime_version())
println("CUDA driver version: ", CUDA.driver_version())
println("CUDA capability: ", CUDA.capability(CUDA.device()))