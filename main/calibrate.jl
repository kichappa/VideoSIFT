
# ENV["JULIA_PYTHONCALL_EXE"] = "~/.conda/envs/cv/bin/python"  # optional
# ENV["JULIA_PYTHONCALL_EXE"] = "@PyCall"  # optional

ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
ENV["JULIA_PYTHONCALL_EXE"] = expanduser("~/.conda/envs/cv/bin/python")
using PythonCall, CondaPkg
using VideoIO, Images, FileIO, Glob, JLD2, UnPack, CSV, DataFrames, LinearAlgebra, Hungarian, OffsetArrays, SparseArrays
using Plots
include("triangulate.jl")
include("calibration_helper.jl")

# f = VideoIO.openvideo("assets/videos/2view1.mp4")
# jl_img = read(f)

# ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr = calibrateCamera("assets/videos/cam", 2, (11, 9, 0), 25, (1, -1, 0))
# jldsave(""assets/videos/cam/calibration/data.jld2"; ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr)

# ---------------------------------------------------------
# ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = calibrateCamera("assets/videos/cam", (3, 4), (9, 7, 0), 25, (1, 1, 0); save = false, include_list = ["MVI_1171_11.png", "MVI_1171_69.png", "DSC_0264_493.png", "DSC_0264_822.png"])
# ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = calibrateCamera("assets/videos/cam", (7,), (8, 7, 0), 25, (1, -1, 0); save = false, debug=true, num_images = 100)
# _, _, _, _, _, _, py_mtx_arr, py_dist_arr =  calibrateCamera("assets/videos/cam", (7,), (8, 7, 0), 12.5, (1, -1, 0); save = false, debug=true, num_images = 1000, return_py_intrinsic=true, win_size=(6, 6), invert=true);
# _, _, _, _, _, _, _, py_mtx_arr, py_dist_arr =
# 	calibrateCamera("assets/videos/cam", (8,), (8, 7, 0), 12.5, (1, -1, 0); save = false, debug = true, num_images = 20, stride = 5, return_py_intrinsic = true, invert = true, win_size = (26, 26), verbosity = 1)
# _, _, _, _, _, _, _, py_mtx_arr, py_dist_arr =
# ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames =
# 	calibrateCamera(
# 		"assets/videos/cam",
# 		(14,),
# 		(11, 8, 0),
# 		16,
# 		(-1, 1, 0);
# 		save = false,
# 		debug = true,
# 		from_video = false,
# 		num_images = 20,
# 		stride = 5,
# 		return_py_intrinsic = true,
# 		invert = false,
# 		win_size = (52, 52),
# 		verbosity = 1,
# 		RO = false,
# 		iFixedPoint = 20 * 3 + 16,
# 	)
# ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames =
# 	calibrateCamera(
	# 		"assets/videos/cam",
	# 		(15,),
	# 		(11, 8, 0),
	# 		16,
	# 		(-1, 1, 0);
# 		save = false,
# 		debug = true,
# 		num_images = 30,
# 		stride = 10,
# 		invert = false,
# 		from_video = true,
# 		win_size = (11, 11),
# 		guess_mtx = py_mtx_arr,
# 		guess_dist = py_dist_arr,
# 		verbosity = 1,
# 		PnP = false,
# 		refineLM = false,
# 		iFixedPoint = 20 * 3 + 16,
# 	)
# _, _, _, _, _, _, _, py_mtx_arr, py_dist_arr =
# calibrateCamera(
# 	"assets/videos/cam",
# 	(18,),
# 	(19, 14, 0),
# 	12.5,
# 	(-1, 1, 0);
# 	save = false,
# 	debug = true,
# 	num_images = 20,
# 	stride = floor(Int, 15*25/30),
# 	return_py_intrinsic = true,
# 	invert = false,
# 	from_video = true,
# 	win_size = (11, 11),
# 	# guess_mtx = py_mtx_arr,
# 	# guess_dist = py_dist_arr,
# 	verbosity = 1,
# 	RO = true,
# 	# refineLM = false,
# 	iFixedPoint = 20 * 3 + 16,
# 	)
	
# ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames =
# calibrateCamera(
# 		"assets/videos/cam",
# 		(18,),
# 		(19, 14, 0),
# 		12.5,
# 		(-1, 1, 0);
# 		save = false,
# 		debug = true,
# 		# num_images = 20,
# 		# stride = 10,
# 		save = false,
# 		debug = true,
# 		guess_mtx = py_mtx_arr,
# 		guess_dist = py_dist_arr,
# 		verbosity = 1,
# 		PnP = true,
# 		ransac = true,
# 		refineLM = true,
# 	)

cam = 23

ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, properties = 
	calibrate(
		"assets/videos/cam",
		(cam,),
		(11, 8, 0),
		16,
		(-1, 1, 0);
		save = false,
		debug = true,
		verbosity = 1,
		from_video = true,
		num_images = [30, Inf],
		win_size = (26, 26),
		# RO = true,
		PnP = true,
		ransac = true,
		refineLM = true,
	)

jldsave("assets/videos/cam/calibration/data_$(cam)_vid.jld2"; ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, reproj_arr, filenames, properties)
# write filenames to a csv file
max_len = maximum(length.(filenames))
# Build a dictionary mapping each new column to a vector, padding with missing values if needed.
cols = Dict{Symbol, Vector{Union{String, Missing}}}()
for (i, vec) in enumerate(filenames)
	col = [j <= length(vec) ? vec[j] : missing for j in 1:max_len]
	cols[Symbol("col$(i)")] = col
end
df = DataFrame(cols)
CSV.write("assets/videos/cam/calibration/data_$(cam)_vid.csv", df)
# ---------------------------------------------------------

# @unpack ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr = jldopen("assets/videos/cam/calibration/data.jld2")


print()

