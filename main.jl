using Images, FileIO, DelimitedFiles, CSV, DataFrames, Format, VideoIO, Glob
include("helper.jl")
include("kernels.jl")
include("blobs.jl")
include("header.jl")
include("calibration_helper.jl")
# include("kernels_inbounds.jl")

fmt = "%.8f"

let
	println("Here we go!")
	cameras = 2
	nImages = 1
	img = []
	imgWidth = 0
	time_taken = 0
	iterations = 1

	# multiple ways to process images. Could be a video file, video stream, or images that will be concatenated.

	# v = Vector{VideoIO.VideoReader}(undef, cameras)
	# for views in 1:cameras
	# 	v[views] = VideoIO.openvideo("assets/videos/view$(views).mp4")
	# end
	
	v = []
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam1/DSC_0048.JPG"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam2/DSC_0083.JPG"))))

	
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam3/MVI_1171_11.png"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam3/MVI_1171_69.png"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam4/DSC_0264_493.png"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam4/DSC_0264_822.png"))))

	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam13/DSCF7567.JPG"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam13/DSCF7568.JPG"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam13/DSCF7569.JPG"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/videos/cam13/DSCF7570.JPG"))))
	# push!(v, Float32.(Gray.(FileIO.load("assets/images/DSCF7591.JPG"))))

	path = "assets/videos/cam22"
	f = vcat(glob("*.mp4", path), glob("*.mov", path), glob("*.MP4", path), glob("*.MOV", path))[1]
	push!(v, Float32.(Gray.(getFrame(f, 339))))
	push!(v, Float32.(Gray.(getFrame(f, 202))))

	# load the images
	for i in 1:cameras
		img_temp = v[i]	
		if i == 1
			img = img_temp
			imgWidth = size(img, 2)
		else
			img = cat(img, img_temp, dims = 2)
		end
	end

	height, width = size(img)
	println(size(img))

	octaves = 5 # not hard coded in the blobs kernel
	layers = 5 # hard coded in the extractBlobXYs function and orientation kernel

	sigma0 = 1.6
	k = 2^(1 / (layers - 3))

    time_taken, count, orientations, blobs, XY_gpu = getBlobs(img, height, width, imgWidth, octaves, layers, nImages, sigma0, k, iterations, time_taken)
	
	println("Got the blobs...")

	println("count: $count, size of XY_gpu: $(size(XY_gpu))")
	XY = [[b.thisX, b.y, b.thisImg, b.x, b.oct, b.lay][i] for b in collect(XY_gpu), i in 1:6]
	println("Total potential blobs: $count, utilization: $(round(count / size(XY_gpu)[1]*100, digits=2))%")
	CSV.write("assets/blobs.csv", DataFrame(XY, :auto))
	df = DataFrame(collect(transpose(collect(orientations))), :auto)
	CSV.write("assets/orientations.csv", df,
		transform = (col, val) -> typeof(val) <: AbstractFloat ? cfmt(fmt, val) : val)
	# CSV.write("assets/filtered_blobs.csv", DataFrame(collect(transpose(blobs)), :auto))
	df = DataFrame(collect(transpose(collect(blobs))), :auto)
	CSV.write("assets/filtered_blobs.csv", df,
		transform = (col, val) -> typeof(val) <: AbstractFloat ? cfmt(fmt, val) : val)

	println("Time taken: $(round(time_taken / (iterations * nImages), digits=5))s for $layers layers and $octaves octaves per image @ $nImages images at a time. Total time taken: $(round(time_taken/iterations, digits=5))s per iteration.")
end
