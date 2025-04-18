# use CondaPkg's scratch environment
# ENV["JULIA_PYTHONCALL_EXE"] = "~/.conda/envs/cv/bin/python"  # optional
# ENV["JULIA_PYTHONCALL_EXE"] = "@PyCall"  # optional

# use an existing conda environment 
ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
ENV["JULIA_PYTHONCALL_EXE"] = expanduser("~/.conda/envs/cv/bin/python")
using PythonCall, CondaPkg
using VideoIO, Images, FileIO, Glob, JLD2, UnPack, CSV, DataFrames, LinearAlgebra, Hungarian, OffsetArrays, SparseArrays
using Plots
include("triangulate.jl")
include("calibration_helper.jl")


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


print()

