using CairoMakie, Statistics
include("calibration.helper.jl")

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

"""
	draw_blob_correspondences(img0, frame0_idx, img1, frame1_idx,
							  blobs0, blobs1, assignment; label0_suffix="") -> Figure

Render a side-by-side CairoMakie Figure for a single consecutive frame pair.

- `img0`, `img1`     : color images (Matrix of RGB).
- `blobs0`, `blobs1` : 2×N matrices of (x, y) blob coordinates (row 1 = x, row 2 = y).
- `assignment`       : Vector{Int} of length N; `assignment[j]` is the index in `blobs1`
					   that matches blob `j` in `blobs0`. Values < 1 are invalid.
- `label0_suffix`    : appended to the left-frame label (e.g. `" (ref)"`).
"""
function draw_blob_correspondences(
	img0::AbstractMatrix,
	frame0_idx::Int,
	img1::AbstractMatrix,
	frame1_idx::Int,
	blobs0::AbstractMatrix{<:Real},
	blobs1::AbstractMatrix{<:Real},
	assignment::AbstractVector{<:Integer};
	label0_suffix::String = "",
)
	h0, w0 = size(img0)
	h1, w1 = size(img1)
	H = max(h0, h1)
	W = w0 + w1

	canvas                       = fill(RGB{N0f8}(0, 0, 0), H, W)
	canvas[1:h0, 1:w0]           .= img0
	canvas[1:h1, (w0+1):(w0+w1)] .= img1

	# build NaN-separated line segments and scatter lists
	segs_x = Float64[]
	segs_y = Float64[]
	xs0    = Float64[]
	ys0    = Float64[]
	xs1    = Float64[]
	ys1    = Float64[]

	for j in eachindex(assignment)
		k = assignment[j]
		k < 1 && continue
		x0f = Float64(blobs0[1, j]);
		y0f = Float64(H - blobs0[2, j])   # flip y
		x1f = Float64(blobs1[1, k]) + w0;
		y1f = Float64(H - blobs1[2, k])
		append!(segs_x, [x0f, x1f, NaN])
		append!(segs_y, [y0f, y1f, NaN])
		push!(xs0, x0f);
		push!(ys0, y0f)
		push!(xs1, x1f);
		push!(ys1, y1f)
	end

	fig = Figure(resolution = (W, H))
	ax  = Axis(fig[1, 1];
				title  = "frame $frame0_idx$label0_suffix  →  frame $frame1_idx",
				aspect = DataAspect()
			)
	hidedecorations!(ax)
	hidespines!(ax)
	image!(ax, 0 .. W, 0 .. H, rotr90(canvas))
	lines!(ax, segs_x, segs_y; color = :lime, linewidth = 1.5)
	scatter!(ax, xs0, ys0; color = :red, markersize = 10)
	scatter!(ax, xs1, ys1; color = :cyan, markersize = 10)

	return fig
end


"""
	plot_consecutive_blob_correspondences(video_file, all_blobs, assignments, fps;
										  out_path, filter_invalid)

For every consecutive pair (i, i+1) in `all_blobs`, write a side-by-side frame
with blob correspondence lines to `out_path` (MP4).

- `all_blobs`   : OffsetArray of `Matrix{Float32}(2, n_blobs)`.
				  Row 1 = x (column), row 2 = y (row) in image coordinates.
- `assignments` : OffsetArray of `Vector{Int}` with the same index range.
				  `assignments[i][j]` is the 1-based index of the matching blob in
				  frame `i+1`, or < 1 when no valid match exists.
- `fps`         : output framerate.
- `filter_invalid` : skip assignments < 1 when `true` (default).

The last index in `all_blobs` is used only as the *target* of the previous frame's
assignments; no pair is drawn starting from it.
"""
function plot_consecutive_blob_correspondences(
	video_file::String,
	all_blobs,
	assignments,
	fps::Real;
	out_path::String     = "assets/LoFTR/blob_correspondences.mp4",
	filter_invalid::Bool = true,
)
	v         = VideoIO.openvideo(video_file)
	idx_first = firstindex(all_blobs)
	idx_last  = lastindex(all_blobs)

	# --- helpers ---
	function load_color(i)
		RGB.(getFrame(v, i; fps = fps))
	end

	function make_canvas(fr0, fr1)
		lh0, lw0 = size(fr0)
		lh1, lw1 = size(fr1)
		lH = max(lh0, lh1)
		cv = fill(RGB{N0f8}(0, 0, 0), lH, lw0 + lw1)
		cv[1:lh0, 1:lw0] .= fr0
		cv[1:lh1, (lw0+1):(lw0+lw1)] .= fr1
		return cv, lw0, lh0, lh1
	end

	# --- infer canvas dimensions from first pair ---
	fr0_init = load_color(idx_first)
	fr1_init = load_color(idx_first + 1)
	cv_init, w0_init, h0_init, _ = make_canvas(fr0_init, fr1_init)
	H_init, W_init = size(cv_init)

	# --- Observables ---
	canvas_obs = Observable(cv_init)
	segs_xs    = Observable(Float64[])
	segs_ys    = Observable(Float64[])
	pts0_xs    = Observable(Float64[])
	pts0_ys    = Observable(Float64[])
	pts1_xs    = Observable(Float64[])
	pts1_ys    = Observable(Float64[])

	fig = Figure(resolution = (W_init, H_init))
	ax  = CairoMakie.Axis(fig[1, 1]; aspect = DataAspect())
	hidedecorations!(ax)
	hidespines!(ax)
	image!(ax, 0 .. W_init, 0 .. H_init, @lift(rotr90($canvas_obs)))
	lines!(ax, segs_xs, segs_ys; color = :lime, linewidth = 1.5)
	scatter!(ax, pts0_xs, pts0_ys; color = :red, markersize = 10)
	scatter!(ax, pts1_xs, pts1_ys; color = :cyan, markersize = 10)

	# --- per-frame update helper ---
	function update_frame!(i)
		j = i + 1
		fr0 = load_color(i)
		fr1 = load_color(j)
		cv, lw0, lh0, lh1 = make_canvas(fr0, fr1)
		lH = size(cv, 1)
		canvas_obs[] = cv

		blobs0 = all_blobs[i]
		blobs1 = all_blobs[j]
		assign = assignments[i]

		_sx = Float64[];
		_sy = Float64[]
		_x0 = Float64[];
		_y0 = Float64[]
		_x1 = Float64[];
		_y1 = Float64[]

		for jj in eachindex(assign)
			k = assign[jj]
			(filter_invalid && k < 1) && continue
			x0f = Float64(blobs0[1, jj])
			y0f = Float64(lH - blobs0[2, jj])      # flip y (image → Makie)
			x1f = Float64(blobs1[1, k]) + lw0
			y1f = Float64(lH - blobs1[2, k])
			append!(_sx, [x0f, x1f, NaN])
			append!(_sy, [y0f, y1f, NaN])
			push!(_x0, x0f);
			push!(_y0, y0f)
			push!(_x1, x1f);
			push!(_y1, y1f)
		end

		segs_xs[] = _sx;
		segs_ys[] = _sy
		pts0_xs[] = _x0;
		pts0_ys[] = _y0
		pts1_xs[] = _x1;
		pts1_ys[] = _y1

		if (i - idx_first + 1) % 100 == 0
			println("  pair ($i, $j) done  ($(length(_x0)) valid matches)")
		end
	end

	# --- record ---
	mkpath(dirname(out_path))
	CairoMakie.record(fig, out_path, idx_first:(idx_last-1);
		framerate = round(Int, fps)) do i
		update_frame!(i)
	end

	VideoIO.close(v)
	println("Saved: $out_path")
end


function visualize_3d_points(points; camera_pos = nothing, lookat = nothing, fps = 30, filename = nothing)
	fig = Figure(resolution = (1920, 1080))
	ax = Axis3(fig[1, 1])

	cam3d!(ax.scene)

	# Observable for animation
	frame_idx = Observable(1)

	# Extract current frame points - returns a 3×N matrix (columns are points)
	current_points = @lift begin
		idx = $frame_idx
		frame = points[begin+idx-1]
		valid_cols = .!isnan.(view(frame, 1, :))
		frame[:, valid_cols]
	end

	# Extract coordinates - view rows as vectors for scatter
	x_coords = @lift(view($current_points, 1, :))
	y_coords = @lift(view($current_points, 2, :))
	z_coords = @lift(view($current_points, 3, :))

	# Scatter plot of points
	scatter!(ax, x_coords, y_coords, z_coords, markersize = 10, color = :blue)

	# Calculate lookat point if not provided
	if isnothing(lookat)
		first_frame = points[begin]
		valid_cols = .!isnan.(view(first_frame, 1, :))
		first_frame_filtered = first_frame[:, valid_cols]
		lookat = Float64.(vec(mean(first_frame_filtered, dims = 2)))
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
	CairoMakie.record(fig, "assets/sfm/$filename.mp4", 1:length(points); framerate = 30) do i
		frame_idx[] = i  # Update the observable to show frame i
	end
end
