using FilePathsBase  # optional, if you prefer file path utilities

function delete_frames(folder1, corners_folder=nothing)
    # folder1 = "assets/videos/cam4 copy"
    if isnothing(corners_folder)
        corners_folder = joinpath(folder1, "corners")
    end

    # Get all png files from folder1 (only files at root of folder1)
    png_files = filter(f -> endswith(f, ".png"), readdir(folder1, join=true))

    for file in png_files
        # Extract the base name without extension (e.g., "filename" from "filename.png")
        base = splitext(basename(file))[1]
        # Build the expected corners file name (e.g., "filename_corners.png")
        corners_file = joinpath(corners_folder, base * "_corners.png")
        
        if !isfile(corners_file)
            println("Deleting $(file) because corresponding corners file not found: $(corners_file)")
            rm(file; force=true)
        end
    end
end