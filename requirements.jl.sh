#!/bin/bash
# Install Julia packages required for growth-rl project

julia -e '
using Pkg
Pkg.add([
    "CSV", 
    "CUDA",
    "DataFrames", 
    "DelimitedFiles", 
    "FileIO", 
    "Format", 
    "Images", 
    "Glob",
    "JLD2",
    "Printf",
    "Statistics",
    "UnPack",
    "VideoIO", 
    ])
'