using CairoMakie, Statistics

# Assuming you have:
# points = [rand(3, 100) for _ in 1:t]  # t frames of 3×n_t points
# camera_pos = [10.0, 10.0, 10.0]
# lookat = [0.0, 0.0, 0.0]

# function visualize_3d_points(points; camera_pos=nothing, lookat=nothing, fps=30)
#     fig = Figure(resolution=(1920, 1080))
#     ax = Axis3(fig[1, 1])
    
#     # Observable for animation
#     frame_idx = Observable(1)
#     current_points = @lift begin
#         frame = points[$frame_idx+firstindex(points)-1]
#         frame[:, .!isnan.(frame[1, :])]  # Filter out columns where [1, j] is NaN
#     end  

#     # Scatter plot of points
#     scatter!(ax, 
#         @lift($(current_points)[1, :]),  # x coordinates
#         @lift($(current_points)[2, :]),  # y coordinates
#         @lift($(current_points)[3, :]),  # z coordinates
#         markersize=5, color=:blue)

#     if isnothing(lookat)
#         # Center the lookat point based on the mean of first frame's points (after filtering NaNs)
#         first_frame = points[begin]
#         first_frame_filtered = first_frame[:, .!isnan.(first_frame[1, :])]
#         lookat = Float64.(vec(mean(first_frame_filtered, dims=2)))
#         println("Calculated lookat point: ", lookat)  # Debugging: print the calculated lookat point
#     end

#     if isnothing(camera_pos)
#         # Position the camera at a distance from the lookat point
#         camera_pos = lookat .+ [1000.0, 1000.0, 1000.0]  
#     end

#     # Set camera view - for Camera3D, set eyeposition and lookat directly
#     ax.scene.camera_controls.eyeposition[] = camera_pos
#     ax.scene.camera_controls.lookat[] = lookat
#     ax.scene.camera_controls.upvector[] = [0.0, 0.0, 1.0]
    
#     println(typeof(points))

#     # Saves directly to "output.mp4" and returns "output.mp4"
#     # filename = record(fig, "assets/sfm/output.mp4", 1:length(points); framerate=30) do i
#     #     frame_idx[] = i
#     # end

#     # # You can specify different formats:
#     # record(fig, "animation.mp4", frames)     # MP4 (H.264)
#     # record(fig, "animation.mkv", frames)     # MKV
#     # record(fig, "animation.gif", frames)     # GIF (slower, larger)
# end

function visualize_3d_points(points; camera_pos=nothing, lookat=nothing, fps=30, filename=nothing)
    fig = Figure(resolution=(1920, 1080))
    ax = Axis3(fig[1, 1])
    
    cam3d!(ax.scene)

    # Observable for animation
    frame_idx = Observable(1)
    
    # Extract current frame points - returns a 3×N matrix (columns are points)
    current_points = @lift begin
        idx = $frame_idx
        frame = points[begin + idx - 1]
        valid_cols = .!isnan.(view(frame, 1, :))
        frame[:, valid_cols]
    end
    
    # Extract coordinates - view rows as vectors for scatter
    x_coords = @lift(view($current_points, 1, :))
    y_coords = @lift(view($current_points, 2, :))
    z_coords = @lift(view($current_points, 3, :))

    # Scatter plot of points
    scatter!(ax, x_coords, y_coords, z_coords, markersize=5, color=:blue)

    # Calculate lookat point if not provided
    if isnothing(lookat)
        first_frame = points[begin]
        valid_cols = .!isnan.(view(first_frame, 1, :))
        first_frame_filtered = first_frame[:, valid_cols]
        lookat = Float64.(vec(mean(first_frame_filtered, dims=2)))
    end

    # Calculate camera position if not provided
    if isnothing(camera_pos)
        camera_pos = lookat .+ [1000.0, 1000.0, 1000.0]
    end

    # Set camera: eye position, lookat center, up vector
    update_cam!(ax.scene, Vec3f(camera_pos...), Vec3f(lookat...), Vec3f(0, 0, 1))
    
    display(fig)
    # return fig, ax, frame_idx
    filename = isnothing(filename) ? "output" : filename
    CairoMakie.record(fig, "assets/sfm/$filename.mp4", 1:length(points); framerate=30) do i
        frame_idx[] = i  # Update the observable to show frame i
    end
end