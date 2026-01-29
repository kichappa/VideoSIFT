using Statistics, UnPack, JLD2, Glob
# This file contains functions for camera calibration using OpenCV and Julia.

# Function to undistort a point using the camera matrix and distortion coefficients.
# This function uses an iterative approach to undistort the point.
function undistort_point(mtx, dist_coeffs, xy_distort; num_iters = 5, debug = false)
	fx = mtx[1, 1]
	fy = mtx[2, 2]
	cx = mtx[1, 3]
	cy = mtx[2, 3]

	k1, k2, p1, p2, k3 = dist_coeffs
	u_dist, v_dist = xy_distort

	# Convert to normalized distorted coordinates
	x = (u_dist - cx) / fx
	y = (v_dist - cy) / fy

	# Iterative distortion correction
	for _ in 1:num_iters
		r2 = x^2 + y^2
		r_factor = 1 + k1 * r2 + k2 * r2^2 + k3 * r2^3
		delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x^2)
		delta_y = p1 * (r2 + 2 * y^2) + 2 * p2 * x * y

		x = (u_dist / fx - cx / fx - delta_x) / r_factor
		y = (v_dist / fy - cy / fy - delta_y) / r_factor
		if debug
			println("(x, y): $(x*fx + cx), $(y*fy + cy)")
		end
	end

	# Convert back to pixel coordinates
	u_undist = x * fx + cx
	v_undist = y * fy + cy

	return [u_undist, v_undist]
end

# Function to undistort a set of points using the camera matrix and distortion coefficients.
function undistortPoints(points, mtx, dist_coeffs, num_iters = 5)
	undistorted_points = zeros(Float32, 2, length(points))
	for (i, point) in enumerate(points)
		undistorted_points[:, i] = undistort_point(mtx, dist_coeffs, point, num_iters = num_iters)
	end
	return undistorted_points
end

#-----------------------------------------------------------------------------------------------------------------------
# Function to extract the camera matrix, distortion coefficients, rotation matrix, and translation vector for the camera
# Overloaded to accept either a frame number or a frame name and direct data matrices or a JLD data file.
function extract_matrices(cam, frame::Integer, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr)
	# Extract the camera matrix, distortion coefficients, rotation matrix, and translation vector for the camera
	K = mtx_arr[cam]
	dist = dist_arr[:, cam]
	R = rvecs_arr[cam][frame]
	t = tvecs_arr[cam][:, frame]
	return K, dist, R, t
end

function extract_matrices(cam, frame_name::String, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
	# Extract the camera matrix, distortion coefficients, rotation matrix, and translation vector for the camera
	frame = findfirst(x -> x == frame_name, filenames[cam])
	if isnothing(frame)
		println("Frame not found")
		return nothing
	end
	frame = frame[1]
	return extract_matrices(cam, frame, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr)
end

function extract_matrices(cam, frame::Integer, jld_file)
	# Extract the camera matrix, distortion coefficients, rotation matrix, and translation vector for the camera
	ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames = begin
		try
			@unpack ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames = jldopen(jld_file)
			ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames
		catch
			@unpack ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = jldopen(jld_file)
			ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, nothing, filenames
		end
	end
	return extract_matrices(cam, frame, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
end

function extract_matrices(cam, frame_name::String, jld_file)
	# Extract the camera matrix, distortion coefficients, rotation matrix, and translation vector for the camera
	ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames = begin
		try
			@unpack ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames = jldopen(jld_file)
			ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames
		catch
			@unpack ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = jldopen(jld_file)
			ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, nothing, filenames
		end
	end
	return extract_matrices(cam, frame_name, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
end
# -----------------------------------------------------------------------------------------------------------------------

# low level function to calculate reprojection error
# uses cv2.projectPoints to project the 3D points to the image plane
function __reprojection_error(cv, np, objpoints, imgpoints, mtx, dist, rvecs, tvecs)
	reproj_err = Vector{Float64}(undef, length(objpoints))
	for i in 1:length(objpoints)
		imgPoints2, jac = cv.projectPoints(objpoints[i], rvecs[i-1], tvecs[i-1], mtx, dist)
		err = cv.norm(imgpoints[i], imgPoints2, cv.NORM_L2) / length(imgPoints2)
		reproj_err[i] = pyconvert(Float64, err)
	end
	return reproj_err
end

# low level function to find chessboard corners
# uses cv2.findChessboardCorners to find the corners of the chessboard pattern
function __findChessboardCorners(jl_img, cb_grid, cv, np; criteria = nothing, invert = false)
	if isnothing(criteria)
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	end

	# Convert the image to grayscale
	jl_img = Gray.(jl_img) .* 255.0

	if invert
		jl_img = 255 .- jl_img
	end
	gray = cv.cvtColor(np.array([UInt8(x.val) for x in jl_img]), cv.COLOR_GRAY2BGR)

	# Find the chessboard corners
	ret, corners = cv.findChessboardCorners(gray, cb_grid[[1, 2]], nothing)

	ret = pyconvert(Bool, ret)

	if ret
		corners = cv.cornerSubPix(cv.cvtColor(gray, cv.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)
	end
	if ret
		println("Chessboard corners found")
		return ret, reduce(vcat, [pyconvert(Array{Float64}, x) for x in corners])
	else
		println("Chessboard corners not found")
		return ret, nothing
	end
end

# This function is a wrapper around the __findChessboardCorners function.
# Overloaded to accept either a file name or an image array.
function findChessboardCorners(image::String, cb_grid; criteria = nothing, cv = cv, np = np, invert = false)
	return __findChessboardCorners(FileIO.load(image), cb_grid, cv, np; criteria = criteria, invert = invert)
end

function findChessboardCorners(image, cb_grid; criteria = nothing, cv = cv, np = np, invert = false)
	return __findChessboardCorners(image, cb_grid, cv, np; criteria = criteria, invert = invert)
end

function findChessboardCorners(image::Vector{String}, cb_grid; criteria = nothing, cv = cv, np = np, invert = false)
	rets = Vector{Bool}(undef, length(image))
	corners = Array{Float64}(undef, 2, prod(cb_grid), length(image))
	for i in eachindex(image)
		ret, corner = __findChessboardCorners(FileIO.load(image[i]), cb_grid, cv, np; criteria = criteria, invert = invert)
		rets[i] = ret
		corners[:, :, i] = permutedims(corner)
	end
	return rets, corners
end

function findChessboardCorners(image::Vector, cb_grid; criteria = nothing, cv = cv, np = np, invert = false)
	rets = Vector{Bool}(undef, length(image))
	corners = Array{Float64}(undef, 2, prod(cb_grid), length(image))
	for i in eachindex(image)
		ret, corner = __findChessboardCorners(image[i], cb_grid, cv, np; criteria = criteria, invert = invert)
		rets[i] = ret
		corners[:, :, i] = permutedims(corner)
	end
	return rets, corners
end

# Function to convert a video file to frames and save them to a directory.
function videoToFrames(file; path = nothing, dir_name = nothing, filename = nothing, ext = "png", num_images = nothing, stride = 1)
	if isnothing(dir_name)
		dir_name = "calibration"
	end
	if isnothing(path)
		path = split(file, ".")[1]
	end
	if isnothing(filename)
		filename = splitdir(path)[2]
		path = splitdir(path)[1]
	end
	if dir_name != ""
		path = path * "/" * dir_name
	end
	if !isdir(path)
		mkdir(path)
	end

	println("Saving frames of $(filename) to $(path)")

	f = VideoIO.openvideo(file)
	frame_number = 1
	while !eof(f) && (isnothing(num_images) || (frame_number - 1) // stride <= num_images)
		if (frame_number - 1) % stride == 0
			frame = read(f)
			FileIO.save(path * "/" * filename * "_$(frame_number).$(ext)", frame)
		end
		frame_number += 1
	end
end

# Function to get a frame from a video file.
# Overloaded to accept either a VideoIO.VideoReader object or a file name.
function getFrame(v::VideoIO.VideoReader, frame, fps=25)
	seek(v, frame/fps)
	return read(v)
end

function getFrame(v::String, frame, fps=25)
	v = VideoIO.openvideo(v)
	seek(v, frame/fps)
	return read(v)
end

# low level function to find chessboard corners
function __calibrateCamera(py_img, filename, count, camera, i, cb_grid, objp, objpoints, imgpoints, filenames; criteria = nothing, save = false, ret_py = false, debug = false, cv = nothing, np = nothing, target_dir = nothing, win_size = (11, 11))
	if isnothing(cv)
		cv = pyimport("cv2")
	end
	if isnothing(np)
		np = pyimport("numpy")
	end

	ret, corners = cv.findChessboardCorners(py_img, cb_grid, cv.CALIB_CB_ADAPTIVE_THRESH)

	if pyconvert(Bool, ret)
		if debug
			println("✅")
		end
		count += 1
		push!(objpoints, np.array(objp'))
		corners = cv.cornerSubPix(cv.cvtColor(py_img, cv.COLOR_BGR2GRAY), corners, win_size, (-1, -1), criteria)
		push!(imgpoints, corners)
		# filename without extension
		push!(filenames[camera], filename)
		if save
			cv.drawChessboardCorners(py_img, cb_grid, corners, ret)
			if !isdir(target_dir * string(i) * "/corners")
				mkdir(target_dir * string(i) * "/corners")
			end
			println("\t... saving to $(target_dir * string(i) * "/corners/$(splitext(filename)[1])_corners$(splitext(filename)[2])")")
			try
				cv.imwrite(target_dir * string(i) * "/corners/$(splitext(filename)[1])_corners$(splitext(filename)[2])", py_img)
			catch e
				println("Error saving image: ", e)
			end
		end
	else
		if debug
			println("❌")
		end
	end
	return count
end

# This function calibrates the camera using the chessboard pattern.
# It is a wrapper around the OpenCV function cv2.calibrateCamera, and cv2.findChessboardCorners through __calibrateCamera().
# This wrapper is extremely flexible, and can be used to calibrate cameras or solvePnP if given the correct intrinsic matrix and distortion coefficients.
function calibrateCamera(
	target_dir,
	num_cameras,
	cb_grid,
	cb_size,
	cb_plane;
	criteria = nothing,
	save = false,
	ret_py = false,
	return_py_intrinsic = false,
	include_list = [],
	debug = false,
	num_images = Inf,
	from_video = false,
	stride = 1,
	win_size = (11, 11),
	invert = false,
	vfile_name = nothing,
	guess_mtx = nothing,
	guess_dist = nothing,
	verbosity = 2,
	RO = false,
	iFixedPoint = nothing,
	PnP = false,
	ransac = false,
	refineLM = false,
	fps = 25
)

	cv = pyimport("cv2")
	np = pyimport("numpy")

	println("Calibrating $(num_cameras) camera(s) in $(target_dir), keyword criteria: $(criteria), save: $(save), ret_py: $(ret_py)")


	# Basic sanity checks
	if typeof(num_cameras) == Int
		@assert num_cameras > 0 "Number of cameras must be greater than 0"
	end
	@assert length(cb_grid) == 3 "cb_grid must be a tuple of 3 elements, each representing the number of internal corners in the x, y, and z directions"
	@assert all(cb_grid .>= 0) "cb_grid must have all non-zero elements, with zero representing cb not found in that direction"
	@assert sum(abs.(cb_plane)) == 2 "cb_plane must have only 2 non-zero elements"
	@assert !any((cb_grid .== 0) .⊻ (cb_plane .== 0)) "cb_grid and cb_plane must have zeros in the same direction"
	@assert !all([RO, PnP]) "Only one of RO or PnP can be true"

	if debug
		println("cb_grid: ", cb_grid)
		println("cb_size: ", cb_size)
		println("cb_plane: ", cb_plane)
		println("calibration technique: ", RO ? "RO" : (PnP ? "PnP" * (ransac ? "+RANSAC" : "") : "Standard"))
	end
	if isnothing(criteria)
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	end

	objp = zeros(Float32, 3, prod(cb_grid .| (cb_grid .== 0)))
	println("size of objp: ", size(objp))

	if cb_plane[3] == 0
		objp[1, :] = reshape([sign(cb_plane[1]) * x * cb_size for x in 0:(cb_grid[1]-1), y in 0:(cb_grid[2]-1)], 1, :)
		objp[2, :] = reshape([sign(cb_plane[2]) * y * cb_size for x in 0:(cb_grid[1]-1), y in 0:(cb_grid[2]-1)], 1, :)
		cb_grid = cb_grid[[1, 2]]
	end
	if cb_plane[2] == 0
		objp[1, :] = reshape([sign(cb_plane[1]) * x * cb_size for x in 0:(cb_grid[1]-1), z in 0:(cb_grid[3]-1)], 1, :)
		objp[3, :] = reshape([sign(cb_plane[3]) * z * cb_size for x in 0:(cb_grid[1]-1), z in 0:(cb_grid[3]-1)], 1, :)
		cb_grid = cb_grid[[1, 3]]
	end
	if cb_plane[1] == 0
		objp[2, :] = reshape([sign(cb_plane[2]) * y * cb_size for y in 0:(cb_grid[2]-1), z in 0:(cb_grid[3]-1)], 1, :)
		objp[3, :] = reshape([sign(cb_plane[3]) * z * cb_size for y in 0:(cb_grid[2]-1), z in 0:(cb_grid[3]-1)], 1, :)
		cb_grid = cb_grid[[2, 3]]
	end
    println("py_cb_grid: $cb_grid")

	ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_err_arr = [], [], [], [], [], []


	if typeof(num_cameras) == Int
		num_cameras = 1:num_cameras
	end
	filenames = Vector{Vector{String}}(undef, length(num_cameras))
	if debug
		println("Camera array: ", num_cameras)
	end
	
	# Iterate over each camera to find the chessboard corners in each image of a camera
	for i in num_cameras
		camera = findfirst(x -> x == i, num_cameras)
		filenames[camera] = Vector{String}()
		objpoints = []
		imgpoints = []
		jl_img = nothing
		if debug
			if from_video
				f = vcat(glob("*.mp4", target_dir * string(i)), glob("*.mov", target_dir * string(i)), glob("*.MP4", target_dir * string(i)), glob("*.MOV", target_dir * string(i)))[1]
				println("Searching for frames in $f for camera $(i)")
			else
				println("Searching for images in $(target_dir * string(i)) for camera $(i)")
			end
		end
		count = 0
		jl_img = nothing
		if from_video
			# check if target_dir is a video file or a directory
			f = nothing
			if isnothing(vfile_name) && isdir(target_dir * string(i))
				f = vcat(glob("*.mp4", target_dir * string(i)), glob("*.mov", target_dir * string(i)), glob("*.MP4", target_dir * string(i)), glob("*.MOV", target_dir * string(i)))[1]
				f = VideoIO.openvideo(f)
			else
				f = VideoIO.openvideo(vfile_name)
			end
			frame_number = 1
			try
				while !eof(f) && count < num_images
					jl_img = read(f)
					if (frame_number - 1) % stride == 0
						if debug && verbosity >= 1
							print("Processing: Frame $(frame_number)($count)...")
						end
						jl_img = Gray.(jl_img) .* 255.0
						if invert
							jl_img = 255 .- jl_img
						end
						py_img = cv.cvtColor(np.array([UInt8(x.val) for x in jl_img]), cv.COLOR_GRAY2BGR)

						try
							count = __calibrateCamera(
								py_img,
								"$(frame_number)",
								count,
								camera,
								i,
								cb_grid,
								objp,
								objpoints,
								imgpoints,
								filenames;
								criteria = criteria,
								save = save,
								ret_py = ret_py,
								debug = debug,
								cv = cv,
								np = np,
								target_dir = target_dir,
								win_size = win_size,
							)
						catch
							continue
						end
					end
					frame_number += 1
				end
			catch e
				println("Error reading video: ", e)
			end
			jl_img = Gray.(getFrame(f, 25, fps))
			for file in vcat(
				glob("*.JPG", target_dir * string(i)),
				glob("*.jpg", target_dir * string(i)),
				glob("*.png", target_dir * string(i)),
				glob("*.JPEG", target_dir * string(i)),
				glob("*.jpeg", target_dir * string(i)),
			)
				# extract the filename 
				filename = split(file, "/")[end]

				if count < num_images || filename in include_list
					if debug && verbosity >= 1
						print("Processing: $file...")
					end
					jl_img = Gray.(FileIO.load(file)) .* 255.0
					if invert
						jl_img = 255.0 .- jl_img
					end
					# save this inverted image to a file
					py_img = cv.cvtColor(np.array([UInt8(x.val) for x in jl_img]), cv.COLOR_GRAY2BGR)

					try
						count =
							__calibrateCamera(
								py_img,
								filename,
								count,
								camera,
								i,
								cb_grid,
								objp,
								objpoints,
								imgpoints,
								filenames;
								criteria = criteria,
								save = save,
								ret_py = ret_py,
								debug = debug,
								cv = cv,
								np = np,
								target_dir = target_dir,
								win_size = win_size,
							)
					catch
						continue
					end
				end
			end
		end

		if jl_img == nothing
			println("No images found for camera $(i)")
			continue
		end

		if PnP
			print("Camera $(i): positioning...")
		else
			print("Camera $(i): calibration...")
		end

		# Calibrate the camera using the chessboard corners if we are calibrating
		# Otherwise, use the guess_mtx and guess_dist to solvePnP
		ret, mtx, dist, rvecs, tvecs, inliers = nothing, nothing, nothing, nothing, nothing, nothing
		if RO
			if isnothing(guess_mtx)
				ret, mtx, dist, rvecs, tvecs = cv.calibrateCameraRO(objpoints, imgpoints, reverse(jl_img.size), iFixedPoint, nothing, nothing)
			else
				flags = cv.CALIB_USE_INTRINSIC_GUESS |
						cv.CALIB_FIX_ASPECT_RATIO |
						cv.CALIB_FIX_PRINCIPAL_POINT

				# Set termination criteria (100 iterations, 1e-6 epsilon)
				criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
				ret, mtx, dist, rvecs, tvecs = cv.calibrateCameraRO(
					objpoints,
					imgpoints,
					reverse(jl_img.size), iFixedPoint,
					guess_mtx[findfirst(x -> x == i, num_cameras)],
					guess_dist[findfirst(x -> x == i, num_cameras)],
					flags = flags,
					criteria = criteria)
			end
		elseif PnP
			rvecs, tvecs, inliers = [], [], []
			mtx = guess_mtx[findfirst(x -> x == i, num_cameras)]
			dist = guess_dist[findfirst(x -> x == i, num_cameras)]
			ret = 0
			for j in axes(objpoints, 1)
				if ransac
					_, rvec, tvec, inlier = cv.solvePnPRansac(objpoints[j], imgpoints[j], guess_mtx[findfirst(x -> x == i, num_cameras)], guess_dist[findfirst(x -> x == i, num_cameras)], useExtrinsicGuess = false)
					push!(inliers, inlier)
				else
					_, rvec, tvec = cv.solvePnP(objpoints[j], imgpoints[j], guess_mtx[findfirst(x -> x == i, num_cameras)], guess_dist[findfirst(x -> x == i, num_cameras)], useExtrinsicGuess = false)
				end
				push!(rvecs, rvec)
				push!(tvecs, tvec)
			end
			rvecs = OffsetArray(rvecs, 0:(length(rvecs)-1))
			tvecs = OffsetArray(tvecs, 0:(length(tvecs)-1))
			inliers = OffsetArray(inliers, 0:(length(inliers)-1))
		else
			if isnothing(guess_mtx)
				ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, reverse(jl_img.size), nothing, nothing)
			else
				flags = cv.CALIB_USE_INTRINSIC_GUESS |
						cv.CALIB_FIX_ASPECT_RATIO |
						cv.CALIB_FIX_PRINCIPAL_POINT

				# Set termination criteria (100 iterations, 1e-6 epsilon)
				criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
				ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
					objpoints,
					imgpoints,
					reverse(jl_img.size),
					guess_mtx[findfirst(x -> x == i, num_cameras)],
					guess_dist[findfirst(x -> x == i, num_cameras)],
					flags = flags,
					criteria = criteria)
			end
		end
		println(" complete!")
		
		# Refine the camera matrix and distortion coefficients using the RANSAC or LM algorithm if requested
		if refineLM
			reproj_err = __reprojection_error(cv, np, objpoints, imgpoints, mtx, dist, rvecs, tvecs)
			println("\tError before RefineLM: mean = $(sum(reproj_err)/length(reproj_err)), std = $(std(reproj_err))")
			for j in axes(objpoints, 1)
				if ransac
					try
						if length(inliers[j-1]) >= 3
							obj_subset = objpoints[j][inliers[j-1]]
							obj_subset = np.reshape(obj_subset, (-1, 3)) 
							obj_subset = np.asarray(obj_subset, dtype=np.float64)

							img_subset = imgpoints[j][inliers[j-1]]
							img_subset = np.reshape(img_subset, (-1, 2))
							img_subset = np.asarray(img_subset, dtype=np.float64)

							
							obj_subset = np.ascontiguousarray(obj_subset)
							img_subset = np.ascontiguousarray(img_subset)
							
							rvec, tvec = cv.solvePnPRefineLM(obj_subset, img_subset, guess_mtx[findfirst(x -> x == i, num_cameras)], guess_dist[findfirst(x -> x == i, num_cameras)], rvecs[j-1], tvecs[j-1])
						end
					catch e
						println("\tRefining failed in image $j after RANSAC. Skipping.")
						rvec = rvecs[j-1]
						tvec = tvecs[j-1]
					end
				else
					rvec, tvec = cv.solvePnPRefineLM(objpoints[j], imgpoints[j], guess_mtx[findfirst(x -> x == i, num_cameras)], guess_dist[findfirst(x -> x == i, num_cameras)], rvecs[j-1], tvecs[j-1])
				end
				rvecs[j-1] = rvec
				tvecs[j-1] = tvec
			end
		end

		reproj_err = __reprojection_error(cv, np, objpoints, imgpoints, mtx, dist, rvecs, tvecs)
		if debug 
			println("Camera $(i) calibration error: $ret: mean = $(sum(reproj_err)/length(reproj_err)), std = $(std(reproj_err))")
			if verbosity >= 2
				println("Camera $(i) matrix: ", mtx)
				println("Camera $(i) distortion: ", dist)
				println("Camera $(i) rvecs: ", rvecs)
				println("Camera $(i) tvecs: ", tvecs)
			end
		end
		# save calibration data
		push!(ret_arr, ret)
		push!(mtx_arr, mtx)
		push!(dist_arr, dist)
		push!(rvecs_arr, rvecs)
		push!(tvecs_arr, tvecs)
		push!(reproj_err_arr, reproj_err)
	end

	# Convert the Python objects to Julia objects for the remaining pipeline

	# ret_arr::Vector{Float64}.
	# Usage: ret_arr[i]::Float64 = calibration error for camera i 
	jl_ret_arr = [pyconvert(Float64, x) for x in ret_arr]

	# mtx_arr::Vector{Matrix{Float64}}.
	# Usage: mtx_arr[i]::Matrix{Float64} = camera matrix for camera i
	jl_mtx_arr = [pyconvert(Matrix{Float64}, x) for x in mtx_arr]

	# dist_arr::Matrix{Float64}.
	# Usage:  i-th column  dist_arr[:,i]::Vector{Float64} = distortion coefficients for camera i
	jl_dist_arr = reduce(hcat, [pyconvert(Vector{Float64}, x[0]) for x in dist_arr])

	# rvecs_arr::Vector{Vector{Matrix{Float64},1},1}.
	# Usage: rvecs_arr[i][j]::Matrix{Float64} = rotation matrix for j-th image in camera i
	jl_rvecs_arr = [[pyconvert(Matrix{Float64}, cv.Rodrigues(x)[0]) for x in rvecs] for rvecs in rvecs_arr]

	# tvecs_arr::Vector{Matrix{Float64}}.
	# Usage: tvecs_arr[i][:,j]::Vector{Float64} = translation vector for j-th image in camera i
	jl_tvecs_arr = [reduce(hcat, [pyconvert(Matrix, x) for x in collect(y)]) for y in tvecs_arr]

	if ret_py
		return jl_ret_arr, jl_mtx_arr, jl_dist_arr, jl_rvecs_arr, jl_tvecs_arr, reproj_err_arr, filenames, [ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr]
	elseif return_py_intrinsic
		return jl_ret_arr, jl_mtx_arr, jl_dist_arr, jl_rvecs_arr, jl_tvecs_arr, reproj_err_arr, filenames, mtx_arr, dist_arr
	else
		return jl_ret_arr, jl_mtx_arr, jl_dist_arr, jl_rvecs_arr, jl_tvecs_arr, reproj_err_arr, filenames
	end
end


# Function to calibrate the camera using the chessboard pattern.
# Uses num_images[1] images in a strided manner to find the chessboard corners, and calibrates the camera using the found corners and get the intrinsic data.
# Uses this data to solvePnP for the remaining num_images[2] images in the video.
function calibrate(
	target_dir,
	num_cameras,
	cb_grid,
	cb_size,
	cb_plane;
	criteria = nothing,
	save = false,
	ret_py = false,
	return_py_intrinsic = false,
	include_list = [],
	debug = false,
	num_images = [30, Inf],
	from_video = false,
	stride = 1,
	win_size = (11, 11),
	invert = false,
	vfile_name = nothing,
	guess_mtx = nothing,
	guess_dist = nothing,
	verbosity = 2,
	RO = false,
	iFixedPoint = nothing,
	PnP = false,
	ransac = false,
	refineLM = false,
	fps = 25,
	
)
	properties = Dict([
		("target_dir", target_dir),
		("num_cameras", num_cameras),
		("cb_grid", cb_grid),
		("cb_size", cb_size),
		("cb_plane", cb_plane)
		])
	_, _, _, _, _, _, _, py_mtx_arr, py_dist_arr =
	calibrateCamera(
	target_dir,
	num_cameras,
	cb_grid,
	cb_size,
	cb_plane;
		save = false,
		debug = true,
		num_images = num_images[1],
		stride = floor(Int, 15*25/30),
		return_py_intrinsic = true,
		invert = invert,
		from_video = from_video,
		win_size = win_size,
		# guess_mtx = py_mtx_arr,
		# guess_dist = py_dist_arr,
		verbosity = verbosity,
		RO = RO,
		# refineLM = false,
		iFixedPoint = 20 * 3 + 16,
		)
	if ret_py
		ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, py =
		calibrateCamera(
				target_dir,
				num_cameras,
				cb_grid,
				cb_size,
				cb_plane;
				save = save,
				debug = debug,
				num_images = num_images[2],
				# stride = 10,
				ret_py = ret_py,
				invert = invert,
				from_video = from_video,
				win_size = win_size,
				guess_mtx = py_mtx_arr,
				guess_dist = py_dist_arr,
				verbosity = verbosity,
				PnP = true,
				ransac = ransac,
				refineLM = refineLM,
			)
		return ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, properties, py
	else
		ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames =
		calibrateCamera(
				target_dir,
				num_cameras,
				cb_grid,
				cb_size,
				cb_plane;
				save = save,
				debug = debug,
				# num_images = 20,
				# stride = 10,
				ret_py = ret_py,
				invert = invert,
				from_video = from_video,
				win_size = win_size,
				guess_mtx = py_mtx_arr,
				guess_dist = py_dist_arr,
				verbosity = verbosity,
				PnP = true,
				ransac = true,
				refineLM = true,
			)
		if return_py_intrinsic
			return ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, properties, py_mtx_arr, py_dist_arr
		else
			return ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, properties
		end
	end
end