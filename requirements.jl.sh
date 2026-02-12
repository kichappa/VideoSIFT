#!/bin/bash
# Install Julia packages required for growth-rl project

julia -e '
using Pkg
Pkg.add([
    "CondaPkg",
    "CSV", 
    "CUDA",
    "DataFrames", 
    "DelimitedFiles", 
    "FileIO", 
    "Format", 
    "GLMakie",
    "Glob",
    "Hungarian",
    "Images", 
    "JLD2",
    "LinearAlgebra",
    "NNlib"
    "OffsetArrays",
    "Printf",
    "PythonCall",
    "SparseArrays",
    "Statistics",
    "UnPack",
    "VideoIO", 
    ])
'