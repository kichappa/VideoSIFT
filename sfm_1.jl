using Images, FileIO, DelimitedFiles, CSV, DataFrames, Format, VideoIO, Glob, UnPack, JLD2
include("main.helper.jl")
include("kernels.jl")
include("blobs.jl")
include("header.jl")
include("calibration.helper.jl")
include("triangulate.jl")
include("plot.helper.jl")

ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
ENV["JULIA_PYTHONCALL_EXE"] = expanduser("~/.conda/envs/cv/bin/python")
using PythonCall, CondaPkg
using OffsetArrays, SparseArrays

using LinearAlgebra, Hungarian

fmt = "%.8f"
blobs = nothing
begin
	println("Here we go!")

	cam = 22
	path = "assets/videos/cam$cam"
	file = vcat(glob("*.mp4", path), glob("*.MP4", path), glob("*.mov", path), glob("*.MOV", path))[1]
	v = VideoIO.openvideo(file)
	fps = VideoIO.framerate(v)
	println("Let's generate SFM using video $file at $(Float32(fps))hz")
	calibration_start_time = 0 # seconds
	calibration_end_time = 7.5  # seconds
	sfm_start = 0 # seconds
	sfm_end = 13.5    # seconds

	sfm_start = Int(round(sfm_start * fps))
	sfm_end = Int(round(sfm_end * fps))
	if sfm_end > VideoIO.get_number_frames(file)
		sfm_end = VideoIO.get_number_frames(file)
	end
	@assert sfm_start < sfm_end "Start time must be less than end time!"
	println("Processing frames $sfm_start to $(sfm_end-1)")

	nImages = 16

	# Start calibration asynchronously
	# @unpack ret_arr, mtx_arr, dist_arr = jldopen("assets/camera_intrinsics/fujifilm_x-s20.jld2")
	mtx_arr, dist_arr = nothing, nothing
	calibrate_task = @async let
		start_time = Int(round(calibration_start_time * fps))
		end_time = Int(round(calibration_end_time * fps))
		calibrate(
			"assets/videos/cam",
			(cam,),
			(11, 8, 0),
			20,
			(-1, 1, 0);
			save = false,
			debug = false,
			verbosity = 1,
			from_video = true,
			start_stage1 = [start_time],
			end_stage1 = [end_time],
			start_stage2 = [sfm_start],
			end_stage2 = [sfm_end],
			# start_stage2 = [10],
			# end_stage2 = [11],
			num_images = [30, Inf],
			win_size = (26, 26),
			return_py_intrinsic = false,
			guess_mtx = mtx_arr,
			guess_dist = dist_arr,
			# RO = true,
			PnP = true,
			ransac = true,
			refineLM = true,
		)
	end

	# Run blob detection asynchronously
	all_blobs = let
		all_blobs = Vector{Matrix{Float32}}()
		# process batches of nImages frames at a time
		gpu_elements = nothing
		for frame in sfm_start:nImages:(sfm_end-1)
			time_taken = 0
			imgs = nothing
			imgWidth = 0
			# load min(nImages, remaining frames) frames
			for i in frame:min(frame+nImages-1, sfm_end-1)
				print("Loading frame $i")
				if isnothing(imgs)
					imgs = Float32.(Gray.(getFrame(v, i; fps = fps)))
					imgWidth = size(imgs, 2)
				else
					imgs = cat(imgs, Float32.(Gray.(getFrame(v, i; fps = fps))), dims = 2)
				end
				print("\e[2K\e[1G")
			end
			height, width = size(imgs)
			print("\tProcessing frames $frame to $(min(frame + nImages - 1, sfm_end-1)) : size = ($height, $(Int32(width//imgWidth)) * $imgWidth)")

			octaves = 5
			layers = 5
			sigma0 = 1.6
			k = 2^(1 / (layers - 3))
			# if width/imgWidth != nImages
			# 	gpu_elements = nothing
			# end
			gpu_elements = nothing
			# time_taken, count, _, blobs, _, counts, _, gpu_elements = getBlobs(
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
				time_taken;
				debug = false,
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
				frame_blobs = blobs[[32+2, 32+3], blobs[1, :] .== i]
				push!(all_blobs, frame_blobs)
			end
			print("\e[2K\e[1G")
		end
		all_blobs
	end
	GC.gc()
	all_blobs = OffsetArray(all_blobs, sfm_start:(sfm_end-1))

	println("Blob detection completed!")
	# println(size(all_blobs), size(all_blobs[begin]))

	ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, ret, properties = fetch(calibrate_task)

	# n_rvecs_arr = Vector{Vector}(undef, 0)
	# push!(n_rvecs_arr, [view(parent(rvecs_arr[1]), :)[1] for i in sfm_start:sfm_end])
	# n_tvecs_arr = Vector{Matrix{Float64}}(undef, 1)
	# n_tvecs_arr[1] = view(tvecs_arr[1], :, fill(1, sfm_end-sfm_start+1))
	n_rvecs_arr = [parent(rvecs_arr[i]) for i in eachindex(rvecs_arr)]
	n_tvecs_arr = tvecs_arr
	filenames[1] = ["$i" for i in sfm_start:sfm_end]

	println("Calibration completed!")
	# print("\e[2K\e[1G")
	println("✅!")

	# Now triangulate the points across frames
	# Create camera identifiers - one for each frame
	cams = fill(1, length(all_blobs))
	cams = OffsetArray(cams, sfm_start:(sfm_end-1))

	# # Package calibration data as a tuple
	# calibration_data = (ret_arr, mtx_arr, dist_arr, parent.(rvecs_arr), tvecs_arr, filenames)
	calibration_data = (ret_arr, mtx_arr, dist_arr, n_rvecs_arr, n_tvecs_arr, filenames)

	jldsave("assets/sfm/blobs+calibration_data_$(1)_$(sfm_start)_$(sfm_end).jld2"; calibration_data, all_blobs)

	# Call triangulateSubsetPoints
	# println("Triangulating $(length(all_blobs)) frames...")
	distances, assignments, reconstructed_points = triangulateSubsetPoints(
		all_blobs,
		nothing,
		cams,
		calibration_data;
		# cam_mtx_override = true,
	)

	# # save reconstructed points as jld2
	jldsave("assets/sfm/reconstructed_points_$(1)_$(sfm_start)_$(sfm_end).jld2";
		distances,
		assignments,
		reconstructed_points,
	)

	println("Triangulation complete!")
	# println("Number of views: $(length(reconstructed_points))")

	# camera_pos = [reshape(-rvecs_arr[1][i+begin-1] * tvecs_arr[1][:, i], (3, 1)) for i in 1:length(rvecs_arr[1])]
	# visualize_3d_points(camera_pos; filename = "camera_pos")

	# Visualize blob correspondences for every consecutive frame pair
	plot_consecutive_blob_correspondences(
		file,
		all_blobs,
		assignments,
		fps;
		out_path = "assets/LoFTR/cam$(cam)_blobs_consecutive_$(sfm_start)-$(sfm_end-1).mp4",
		filter_invalid = true,
	)

end
