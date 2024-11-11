import OpenCV.getGaussianKernel


function getGaussianKernelalt(ksize, sigma)
    kernel = zeros(Float32, ksize)
    kernel = exp.(-0.5 * ((0:ksize-1) .- (ksize - 1) / 2) .^ 2 / sigma^2)
    kernel = kernel ./ sum(kernel)
    return kernel
end

k = 11
sigma = 5.4

println(reshape(getGaussianKernel(k, sigma), k))
println(getGaussianKernelalt(k, sigma))
println(round.(reshape(getGaussianKernel(k, sigma), k) .- getGaussianKernelalt(k, sigma), digits=5))