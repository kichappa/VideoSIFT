include("kernels.jl")
using FileIO, Images
# get all png files in root directory
png_files = readdir(".") |> filter(f -> endswith(f, ".png"))
imgs = [FileIO.load(f) for f in png_files]
