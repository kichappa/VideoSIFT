using Statistics, CUDA, NNlib, CUDA.CUSOLVER

# function to calculate the distance between two points in 3D space,
# given the camera matrices, rotation matrices, translation vectors, and the distorted pixel coordinates.
function distance3D(K1, dist1, R1, t1, K2, dist2, R2, t2, p1, p2; num_iters = 5)

	# Undistort the input pixel coordinates (p1 and p2 are 2-vectors, e.g. [u, v])
	p1_undist = undistort_point(K1, dist1, p1; num_iters = num_iters)
	p2_undist = undistort_point(K2, dist2, p2; num_iters = num_iters)

	# Convert the pixel coordinates to homogeneous coordinates
	p1_h = [p1_undist[1], p1_undist[2], 1.0]
	p2_h = [p2_undist[1], p2_undist[2], 1.0]

	# Compute the camera centers
	c1 = -R1' * t1
	c2 = -R2' * t2

	# Compute normalized rays in camera coordinates
	l1 = inv(K1) * p1_h
	l1 = l1 / norm(l1)
	l2 = inv(K2) * p2_h
	l2 = l2 / norm(l2)

	# Transform the rays into world coordinates
	l1 = R1' * l1
	l2 = R2' * l2

	# Compute the distance between the two skew rays:
	l1xl2 = cross(l1, l2)
	d = abs(dot(c2 - c1, l1xl2)) / norm(l1xl2)
	return d
end

# function to calculate the intersection point of two rays in 3D space
# this method is not statiscally stable, but it is fast and works decently well in practice
function intersection3D(K1, dist1, R1, t1, K2, dist2, R2, t2, p1, p2; num_iters = 5)
	# Undistort the input pixel coordinates
	p1_undist = undistort_point(K1, dist1, p1; num_iters = num_iters)
	p2_undist = undistort_point(K2, dist2, p2; num_iters = num_iters)

	# Convert the pixel coordinates to homogeneous coordinates
	p1_h = [p1_undist[1], p1_undist[2], 1.0]
	p2_h = [p2_undist[1], p2_undist[2], 1.0]

	# Compute the camera centers
	c1 = -R1' * t1
	c2 = -R2' * t2

	# Compute normalized rays in camera coordinates
	l1 = inv(K1) * p1_h
	l1 = l1 / norm(l1)
	l2 = inv(K2) * p2_h
	l2 = l2 / norm(l2)

	# Transform the rays into world coordinates
	l1 = R1' * l1
	l2 = R2' * l2

	# Compute the dot product between the rays
	d_dot = dot(l1, l2)
	denom = 1 - d_dot^2

	# Handle nearly parallel rays: return the midpoint of the camera centers
	if abs(denom) < eps()
		return (c1 + c2) / 2
	end

	# Solve for the parameters s and t that give the closest points on the rays
	A = dot(l1, c2 - c1)
	B = dot(l2, c2 - c1)
	s = (A + d_dot * B) / denom
	t_param = (B + d_dot * A) / denom

	# Closest points on the rays
	q1 = c1 + s * l1
	q2 = c2 + t_param * l2

	# Return the midpoint as the estimated intersection point
	return (q1 + q2) / 2
end

# ------------------------------------------------------------------------------------------------------------------------
# functions to find the intersection of two rays in 3D space in a statistically stable way
# solves x1 cross P1 X = 0 and x2 cross P2 X = 0
# where P1 and P2 are the camera matrices
# x is the pixel coordinates in each view
# X is the 3D point in space to be found
# uses CUDA to speed up the process

# function to calculate the normization matrix for a view 
# (3D array version with fixed number of points across views)
function compute_T(points::AbstractArray{T, 3}) where T
	μ = Statistics.mean!(ones(2, 1, size(points, 3)), points)

	scale = sqrt(2) ./
			dropdims(
		Statistics.mean!(ones(1, 1, size(points, 3)),
			sqrt.(sum((points .- μ) .^ 2, dims = 1)),
		),
		dims = (1, 2),
	)
	μ = dropdims(μ, dims = 2)

	μ = CuArray(μ)
	scale = CuArray(scale)
	Ts = CUDA.zeros(eltype(T), 3, 3, size(points, 3))
	Ts[1, 1, :] .= scale
	Ts[1, 3, :] .= -scale .* μ[1, :]
	Ts[2, 2, :] .= scale
	Ts[2, 3, :] .= -scale .* μ[2, :]
	Ts[3, 3, :] .= 1.0f0
	return Ts
end

# Overload for variable-sized points (vector of matrices)
function compute_T(points_arr::Vector{<:Matrix})
	n_views = length(points_arr)
	T = eltype(points_arr[1])
	μ = zeros(T, 2, n_views)
	scale = zeros(T, n_views)

	for (idx, pts) in enumerate(points_arr)
		# Compute mean and scale on GPU
		μ[:, idx] .= vec(mean(pts, dims = 2))  # 2-element vector
		centered = pts .- view(μ, :, idx) # using view to avoid copying
		scale[idx] = sqrt(2.0f0) / mean(sqrt.(sum(centered .^ 2, dims = 1)))
	end

	μ = CuArray(μ)
	scale = CuArray(scale)

	Ts = CUDA.zeros(T, 3, 3, n_views)
	Ts[1, 1, :] .= scale
	Ts[1, 3, :] .= -scale .* μ[1, :]
	Ts[2, 2, :] .= scale
	Ts[2, 3, :] .= -scale .* μ[2, :]
	Ts[3, 3, :] .= 1.0f0

	return Ts
end

# function to compute the triangulation of points given camera matrices and points
# points is a matched 3D array of points, where the first dimension is the x, y coordinates,
# 												the second dimension is the number of points, 
# 												the third dimension same point in different views
# points[:, i, j] = [x, y] is the i-th point to be triangulated in the j-th view
# Ks is a vector of camera matrices, Rs is a vector of rotation matrices, ts is a vector of translation vectors
function triangulate_batched(Ks, Rs, ts, points)
	N = size(points, 2) # number of points
	M = size(points, 3) # number of views

	Ts = compute_T(points)
	points_homog = CUDA.ones(3, N, M)
	points_homog[1:2, :, :] .= CuArray(points)
	Ps = CUDA.zeros(Float32, 3, 4, M)
	println(Ks)
	println(Rs)
	println(ts)
	for i in axes(Ks, 1)
		Ps[:, :, i] .= CuArray(Ks[i]) * CuArray(hcat(Rs[i][:, :], ts[i]))
	end

	# Normalize the points and camera matrices
	points_norm = batched_mul(Ts, points_homog)
	Ps_norm = batched_mul(Ts, Ps)

	# Preallocate output [3×N×(M-1)]
	triangulated_points = CUDA.zeros(Float32, 3, N, M - 1)
	for pair in 1:(M-1)
		triangulated_points[:, :, pair] = triangulate_pair(
			Ps_norm[:, :, pair],
			Ps_norm[:, :, pair+1],
			points_norm[:, :, pair],
			points_norm[:, :, pair+1],
		)
	end

	return collect(triangulated_points)
end

# Overload for variable-sized points (vector of matrices) with Hungarian assignments
function triangulate_batched(Ks, Rs, ts, points::Vector{<:Matrix}, assignments::Vector{<:AbstractVector})
	n_views = length(points)
	@assert length(assignments) == n_views "Need n-1 assignments for n views"

	Ts = compute_T(points) # CUDA array of normalization matrices

	# Compute P on CPU, transfer once, multiply with GPU Ts
	Ps_norm_vec = [view(Ts,:,:,i) * CuArray(Ks[i] * hcat(Rs[i][:, :], ts[i])) for i in 1:n_views]

	# Pre-compute normalized points
	points_norm_vec = Vector{CuArray{Float32, 2}}(undef, n_views)
	for i in 1:n_views
		n_pts = size(points[i], 2)
		pts_h = CUDA.ones(Float32, 3, n_pts)
		pts_h[1:2, :] .= CuArray(points[i])
		points_norm_vec[i] = view(Ts,:,:,i) * pts_h
	end

	# Store triangulated results for each pair
	reconstructed_points = Vector{Matrix{Float32}}(undef, n_views)

	for view1_idx in 1:n_views
		view2_idx = view1_idx % n_views + 1

		# Get assignments for this pair
		# mask invalid assignments
		assign_masked = map(x -> x < 1 ? 1 : x, assignments[view1_idx])

		println("$view1_idx to $view2_idx:")
		# Triangulate this pair using pre-computed normalized data
		triangulated = triangulate_pair(
			Ps_norm_vec[view1_idx],
			Ps_norm_vec[view2_idx],
			view(points_norm_vec[view1_idx], :, :),
			view(points_norm_vec[view2_idx], :, assign_masked),
		)

		triangulated = collect(triangulated)
		triangulated[:, assignments[view1_idx] .< 1] .= NaN
		# println("View $view1_idx to $view2_idx: Triangulated size: ", size(triangulated))
		# println(collect(triangulated))

		reconstructed_points[view1_idx] = triangulated
	end

	return reconstructed_points
end

# function to triangulate a pair of points given the camera matrices and points
function triangulate_pair(P1, P2, pts1, pts2)
	N = size(pts1, 2)
	A_batch = CUDA.zeros(Float32, 4, 4, N)

	# kernel to fill the A matrix
	function kernel_fill_A!(P1, P2, pts1, pts2, A_batch, N)
		i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		if i <= N
			x1 = pts1[1, i]
			y1 = pts1[2, i]
			x2 = pts2[1, i]
			y2 = pts2[2, i]

			@inbounds begin
				# First camera equations
				A_batch[1, 1, i] = x1 * P1[3, 1] - P1[1, 1]
				A_batch[1, 2, i] = x1 * P1[3, 2] - P1[1, 2]
				A_batch[1, 3, i] = x1 * P1[3, 3] - P1[1, 3]
				A_batch[1, 4, i] = x1 * P1[3, 4] - P1[1, 4]

				A_batch[2, 1, i] = y1 * P1[3, 1] - P1[2, 1]
				A_batch[2, 2, i] = y1 * P1[3, 2] - P1[2, 2]
				A_batch[2, 3, i] = y1 * P1[3, 3] - P1[2, 3]
				A_batch[2, 4, i] = y1 * P1[3, 4] - P1[2, 4]

				# Second camera equations
				A_batch[3, 1, i] = x2 * P2[3, 1] - P2[1, 1]
				A_batch[3, 2, i] = x2 * P2[3, 2] - P2[1, 2]
				A_batch[3, 3, i] = x2 * P2[3, 3] - P2[1, 3]
				A_batch[3, 4, i] = x2 * P2[3, 4] - P2[1, 4]

				A_batch[4, 1, i] = y2 * P2[3, 1] - P2[2, 1]
				A_batch[4, 2, i] = y2 * P2[3, 2] - P2[2, 2]
				A_batch[4, 3, i] = y2 * P2[3, 3] - P2[2, 3]
				A_batch[4, 4, i] = y2 * P2[3, 4] - P2[2, 4]
			end
		end
		return
	end

	# Launch kernel
	threads = 256
	blocks = cld(N, threads)
	@cuda threads = threads blocks = blocks kernel_fill_A!(P1, P2, pts1, pts2, A_batch, N)

	# Compute SVD in a batched manner on the GPU
	F = CUDA.svd(A_batch)

	# Extract the last column of V (the right singular vectors)
	V = F.V
	triangulated_points = V[begin:(end-1), 4, :] ./ V[4, 4, :]'

	# Rescale the triangulated points
	return triangulated_points
end


function triangulatePoints(points, image_names, cams, datafile)
	K = []
	d = []
	R = []
	t = []
	for view in axes(points, 3)
		# Handle both JLD file path and direct calibration data
		if datafile isa AbstractString
			# datafile is a path to JLD file
			K1, d1, R1, t1 = extract_matrices(cams[view], image_names[view], datafile)
		else
			# datafile is a tuple/named tuple: (ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
			ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = datafile
			K1, d1, R1, t1 = extract_matrices(cams[view], image_names[view], ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
		end
		push!(K, K1)
		push!(d, d1)
		push!(R, R1)
		push!(t, t1)
	end

	# distances are stored in a 3D array,
	# each layer (third dimension) i contains the 3D distances between points in view[i] (row) and view[i+1] (column)
	distances = zeros(size(points)[[2, 2, 3]])
	for view1 in axes(points, 3)
		view2 = (view1) % size(points, 3) + 1
		for i in axes(points, 2)
			for j in axes(points, 2)
				distances[i, j, view1] = distance3D(K[view1], d[view1], R[view1], t[view1], K[view2], d[view2], R[view2], t[view2], points[:, i, view1], points[:, j, view2])
			end
		end
	end

	assignments = []
	for view in axes(points, 3)
		assignment, _ = hungarian(distances[:, :, view])
		push!(assignments, assignment)
	end

	cumulative_assignments = zeros(Int, size(points, 2), size(points, 3) + 1)
	cumulative_assignments[:, 1] = 1:size(points, 2)
	for view in axes(points, 3)
		cumulative_assignments[:, view+1] = assignments[view][cumulative_assignments[:, view]]
	end
	cumulative_assignments = cumulative_assignments[:, (begin+1):end]

	matched_points = copy(points)
	for view in axes(points, 3)
		matched_points[:, :, view] = points[:, cumulative_assignments[:, view], view]
	end

	reconstructed_points = zeros(Float32, 3, size(points, 2), size(points, 3))

	reconstructed_points[:, :, begin:(end-1)] = triangulate_batched(K, R, t, matched_points)

	return distances[begin:(end-1)], cumulative_assignments[:, begin:(end-1)], reconstructed_points[:, :, begin:(end-1)]
end

function triangulateSubsetPoints(points, image_names, cams, datafile; cam_mtx_override = false)
	# points are Vector{Matrix{Float32, 2, n}, views} where n is the number of points in the respective view. n is not the same for all views. 
	K = []
	d = []
	R = []
	t = []
	# iterate over views to get calibration data for each view
	for view in eachindex(points)
		if cam_mtx_override
			ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = datafile
			K1 = mtx_arr[begin]
			d1 = dist_arr[:, begin]
			R1 = rvecs_arr[begin][begin]
			t1 = tvecs_arr[begin][:, begin]
		else
			# Handle both JLD file path and direct calibration data
			if datafile isa AbstractString
				# datafile is a path to JLD file
				K1, d1, R1, t1 = extract_matrices(cams[view], image_names[view], datafile)
			else
				# datafile is a tuple/named tuple: (ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
				ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames = datafile
				# println("$(typeof(cams)), $(typeof(cams[view])), $(typeof(view)), $(typeof(ret_arr)), $(typeof(mtx_arr)), $(typeof(dist_arr)), $(typeof(rvecs_arr)), $(typeof(tvecs_arr)), $(typeof(filenames))")
				# println(extract_matrices(cams[view], view, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames))
				K1, d1, R1, t1 = extract_matrices(cams[view], view, ret_arr, mtx_arr, dist_arr, rvecs_arr, tvecs_arr, filenames)
			end
		end
		push!(K, K1)
		push!(d, d1)
		push!(R, R1)
		push!(t, t1)
	end

	K = OffsetArray(K, firstindex(points):lastindex(points))
	d = OffsetArray(d, firstindex(points):lastindex(points))
	R = OffsetArray(R, firstindex(points):lastindex(points))
	t = OffsetArray(t, firstindex(points):lastindex(points))

	# distances are stored in a 3D array,
	# each layer (third dimension) i contains the 3D distances between points in view[i] (row) and view[i+1] (column)
	println("Computing pairwise distances between points in different views...")
	sizes = size.(points, 2) # also potentially offset if points are offset in indexing
	distances = similar(points, Matrix{Float32})
	pwise_max_sizes = [maximum([sizes[i], sizes[(i-firstindex(sizes)+1)%length(sizes)+firstindex(sizes)]]) for i in eachindex(sizes)]
	for i in eachindex(points)
		next_i = (i - firstindex(points) + 1) % length(points) + firstindex(points)
		# println("View-pairs ($i, $next_i) with sizes ($(sizes[i]), $(sizes[next_i]): $(pwise_max_sizes[i]))")
	end
	for view1 in eachindex(points)
		view2 = (view1 - firstindex(points) + 1) % length(points) + firstindex(points)
		local_distances = zeros(Float32, size(points[view1], 2), size(points[view2], 2))
		for i in axes(points[view1], 2)
			for j in axes(points[view2], 2)
				local_distances[i, j] = distance3D(K[view1], d[view1], R[view1], t[view1], K[view2], d[view2], R[view2], t[view2], points[view1][:, i], points[view2][:, j])
			end
		end
		distances[view1] = fill(2*maximum(local_distances), pwise_max_sizes[view1], pwise_max_sizes[view1])
		distances[view1][begin:(begin+size(points[view1], 2)-1), begin:(begin+size(points[view2], 2)-1)] = local_distances
	end
	println("Distance computation complete! Now solving assignment problem for each view pair...")
	assignments = Vector{Vector{Int}}()
	jldsave("assets/sfm/distances_$(1)_$(sfm_start)_$(sfm_end).jld2"; distances)
	for view in eachindex(points)
		assignment, _ = hungarian(distances[view])
		push!(assignments, assignment)
	end

	assignments .= [
		# begin
		# println("$i -- $((i+1) % length(points)): $(i+firstindex(points)-1) --- $(i%length(points)+firstindex(points))")
		map(x ->
				x <= size(
					points[i%length(points)+firstindex(points)], 2) ?
				x : -1,
			a[1:size(points[i+firstindex(points)-1], 2)],
		)
		# end
		for (i, a) in enumerate(assignments)
	]
	println("Assignments computed! Now triangulating points for each view pair...")
	for i in eachindex(assignments)
		println("$i: $(assignments[i])")
	end
	# cumulative_assignments = zeros(Int, size(points, 2), size(points, 3) + 1)
	# cumulative_assignments[:, 1] = 1:size(points, 2)
	# for view in axes(points, 3)
	# 	cumulative_assignments[:, view+1] = assignments[view][cumulative_assignments[:, view]]
	# end
	# cumulative_assignments = cumulative_assignments[:, (begin+1):end]

	# matched_points = copy(points)
	# for view in axes(points, 3)
	# 	matched_points[:, :, view] = points[:, cumulative_assignments[:, view], view]
	# end

	reconstructed_points = triangulate_batched(parent(K), parent(R), parent(t), parent(points), assignments)
	println("Triangulation complete! Reshaping results...")
	reconstructed_points = OffsetArray(reconstructed_points, firstindex(points):lastindex(points))

	return distances, OffsetArray(assignments, firstindex(points):lastindex(points)), reconstructed_points
end

function plotReconstructedPoints(points::AbstractMatrix)
	@assert size(points, 2) == 3 "points must be an n×3 matrix"

	plot = Plots.scatter3d(points[:, 1], points[:, 2], points[:, 3],
		markercolor = :blue, markersize = 5,
		xlabel = "X", ylabel = "Y", zlabel = "Z",
		title = "Reconstructed 3D Points",
		label = "Reconstructed Points")

	savefig("assets/reconstructed_points.png")
	println("3D plot saved to assets/reconstructed_points.png")
	return plot
end
