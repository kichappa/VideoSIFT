julia> include("timeit2.jl")
Gaussian time: 0.0003201204057808758 seconds per image @ 2 images at once
Pyramid time: [0.0019370039, 0.00026341513] seconds per image @ 2 images at once
Blobs time: 0.0023665904019538296 seconds per image @ 2 images at once
Gaussian time: 0.0006259566061539726 seconds per image @ 4 images at once
Pyramid time: [0.0017562606, 0.00018308974] seconds per image @ 4 images at once
Blobs time: 0.0021117710034513487 seconds per image @ 4 images at once
Gaussian time: 0.001273952847928152 seconds per image @ 8 images at once
Pyramid time: [0.0016576775, 0.00014485503] seconds per image @ 8 images at once
Blobs time: 0.00197905912934191 seconds per image @ 8 images at once
Gaussian time: 0.0024344295774371308 seconds per image @ 16 images at once
Pyramid time: [0.0016086687, 0.00012484327] seconds per image @ 16 images at once
Blobs time: 0.0019106239999404572 seconds per image @ 16 images at once
Gaussian time: 0.0048754918707253625 seconds per image @ 32 images at once
Pyramid time: [0.0016107309, 0.0001150353] seconds per image @ 32 images at once
Blobs time: 0.0022772109862699203 seconds per image @ 32 images at once
Gaussian time: 0.011180978209442327 seconds per image @ 64 images at once
Pyramid time: [0.0017496685, 0.000106993546] seconds per image @ 64 images at once

julia> include("timeit2.jl")
Pyramid time: [0.0019093702, 0.00025861966] seconds per image @ 2 images at once
Pyramid time: [0.0017431302, 0.00018241288] seconds per image @ 4 images at once
Pyramid time: [0.001657192, 0.00014422629] seconds per image @ 8 images at once
Pyramid time: [0.0016083723, 0.00012323937] seconds per image @ 16 images at once
Pyramid time: [0.0015868745, 0.00011336459] seconds per image @ 32 images at once
Pyramid time: [0.001587973, 0.00010847765] seconds per image @ 64 images at once
Pyramid time: [0.0015708173, 0.00010573144] seconds per image @ 128 images at once
Pyramid time: [0.0017684081, 0.000104657396] seconds per image @ 256 images at once

(cv-gpu) [kshenoy8@atl1-1-01-006-19-0 VideoSIFT]$ python timeit_pyramid.py 
Pytorch pyramid time: [0.001034, 0.000118] seconds per image @ 1 images at once
Pyramid time: [0.005131, 0.001577] seconds per image @ 1 images at once
Pytorch pyramid time: [0.001039, 0.000117] seconds per image @ 2 images at once
Pyramid time: [0.005112, 0.001615] seconds per image @ 2 images at once
Pytorch pyramid time: [0.001024, 0.000113] seconds per image @ 4 images at once
Pyramid time: [0.005154, 0.001607] seconds per image @ 4 images at once
Pytorch pyramid time: [0.001049, 0.000118] seconds per image @ 8 images at once
Pyramid time: [0.005147, 0.001602] seconds per image @ 8 images at once
Pytorch pyramid time: [0.001085, 0.000125] seconds per image @ 16 images at once
Pyramid time: [0.005142, 0.001588] seconds per image @ 16 images at once
Pytorch pyramid time: [0.001042, 0.000117] seconds per image @ 32 images at once
Pyramid time: [0.005154, 0.001605] seconds per image @ 32 images at once
Pytorch pyramid time: [0.001231, 0.000124] seconds per image @ 64 images at once
Pyramid time: [0.005178, 0.001620] seconds per image @ 64 images at once
Pytorch pyramid time: [0.001093, 0.000124] seconds per image @ 128 images at once
Pyramid time: [0.005419, 0.001739] seconds per image @ 128 images at once
Pytorch pyramid time: [0.001188, 0.000128] seconds per image @ 256 images at once
Pyramid time: [0.028925, 0.010632] seconds per image @ 256 images at once
(cv-gpu) [kshenoy8@atl1-1-01-006-19-0 VideoSIFT]$ 




