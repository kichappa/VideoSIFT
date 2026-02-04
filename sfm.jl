using Images, FileIO, DelimitedFiles, CSV, DataFrames, Format, VideoIO, Glob
include("main.helper.jl")
include("kernels.jl")
include("blobs.jl")
include("header.jl")
include("calibration.helper.jl")


ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
ENV["JULIA_PYTHONCALL_EXE"] = expanduser("~/.conda/envs/cv/bin/python")
using PythonCall, CondaPkg
using OffsetArrays, SparseArrays

fmt = "%.8f"
blobs = nothing
let
	println("Here we go!")

	path = "assets/videos/cam22"
	f = vcat(glob("*.mp4", path), glob("*.mov", path), glob("*.MP4", path), glob("*.MOV", path))[1]
	v = VideoIO.openvideo(f)
	fps = VideoIO.framerate(v)
	println("Let's generate SFM using video $f at $(Int32(fps))hz")
	sfm_start = 11.5 # seconds
	sfm_end = 13.5   # seconds

	sfm_start = Int(round(sfm_start * fps))
	sfm_end = Int(round(sfm_end * fps))
	if sfm_end > VideoIO.get_number_frames(f)
		sfm_end = VideoIO.get_number_frames(f)
	end
	@assert sfm_start < sfm_end "Start time must be less than end time!"
	println("Processing frames $sfm_start to $sfm_end")

	nImages = 16

	# Start calibration asynchronously
	calibrate_task = @async begin
		calibrate(
			"assets/videos/cam",
			(22,),
			(11, 8, 0),
			20,
			(-1, 1, 0);
			save = false,
			debug = false,
			verbosity = 1,
			from_video = true,
			start_frame = [0*25],
			end_frame = [10*25],
			start_frame_stage2 = [sfm_start],
			end_frame_stage2 = [sfm_end],
			num_images = [30, Inf],
			win_size = (26, 26),
			# RO = true,
			PnP = true,
			ransac = true,
			refineLM = true,
		)
	end

	# Run blob detection asynchronously
	blob_task = begin
		all_blobs = Vector{Matrix{Float32}}()
		# process batches of nImages frames at a time
		for frame in sfm_start:nImages:sfm_end-1
			time_taken = 0
			imgs = nothing
			imgWidth = 0
			# load min(nImages, remaining frames) frames
			for i in frame:min(frame+nImages-1, sfm_end-1)
				if isnothing(imgs)
					imgs = Float32.(Gray.(getFrame(v, i)))
					imgWidth = size(imgs, 2)
				else
					imgs = cat(imgs, Float32.(Gray.(getFrame(v, i))), dims = 2)
				end
			end
			height, width = size(imgs)
			print("\tProcessing frames $frame to $(min(frame + nImages - 1, sfm_end-1)) : size = ($height, $(Int32(width//imgWidth)) * $imgWidth)")

			octaves = 5
			layers = 5
			sigma0 = 1.6
			k = 2^(1 / (layers - 3))
			time_taken, count, _, blobs, _, counts, _ = getBlobs(
				imgs,
				height,
				width,
				imgWidth,
				octaves,
				layers,
				Int32(width/imgWidth),
				sigma0,
				k,
				time_taken,
			)
			blank_slate = CUDA.zeros(Float32, 4, size(imgs)...)
			blank_slate[1:3, :, :] .= reshape(CuArray(imgs), 1, size(imgs)...) .* 0.2
			blank_slate[4, :, :] .= 1.0
			@cuda threads = 1 blocks = count plot_blobs_f(CuArray(blobs), blank_slate, height, width, size(blobs, 1))
			CUDA.synchronize()
			save("assets/sfm/filtered_blobs_$(frame)_$(min(frame + nImages - 1, sfm_end-1)).png", colorview(RGBA, Array(blank_slate)))
			blobs = vcat(zeros((1, size(blobs, 2))), blobs)
			blobs[1, :] .= floor.(Int32, (blobs[32+6, :] .- 1) ./ 3840 .+ 1)
			# We can now extract xys blobs from image i using blobs[[32+2, 32+3], blobs[1,:] .== i]
			# The x coordinate will be image local, not group local. ie x ∈ [1, imgWidth]
			
			# Push blobs for each frame in the batch to all_blobs
			for i in 1:Int32(width//imgWidth)
				frame_blobs = blobs[[32+2, 32+3], blobs[1,:] .== i]
				push!(all_blobs, frame_blobs)
			end
			print("\e[2K\e[1G")
		end
		all_blobs
	end
	println("Blob detection completed!")
	println(size(blob_task), size(blob_task[1]))
	
	print("Waiting for calibration to complete...")
	ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, ret, properties = fetch(calibrate_task)
	# print("\e[2K\e[1G")
	println("✅!")

	# DEBUG: What exactly are "filenames"?
	println("Filenames returned from calibration:")
	for fn in filenames
		println(" - $fn")
	end

	# Wait for both tasks to complete
	# print("Waiting for blob detection to complete...")
	# all_blobs = fetch(blob_task)
	# print("\e[2K\e[1G")
	# println("✅!")
end
