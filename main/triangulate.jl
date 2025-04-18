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
function compute_T(points)
	μ = mean!(ones(2, 1, size(points, 3)), points)
	scale = sqrt(2) ./ dropdims(mean!(ones(1, 1, size(points, 3)), sqrt.(sum((points .- μ) .^ 2, dims = 1))), dims = (1, 2))
	μ = dropdims(μ, dims = 2)

	μ = CuArray(μ)
	scale = CuArray(scale)
	Ts = CUDA.zeros(eltype(scale), 3, 3, size(points, 3))
	Ts[1, 1, :] .= scale
	Ts[1, 3, :] .= -scale .* μ[1, :]
	Ts[2, 2, :] .= scale
	Ts[2, 3, :] .= -scale .* μ[2, :]
	Ts[3, 3, :] .= 1.0f0
	return Ts
end

# function to compute the triangulation of points given camera matrices and points
# points is a matched 3D array of points, where the first dimension is the x, y, z coordinates,
# 												the second dimension is the number of points, 
# 												the third dimension same point in different views
# points[:, i, j] = [x, y, z] is the i-th point to be triangulated in the j-th view
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
	for pair in 1:M-1
		triangulated_points[:, :, pair] = triangulate_pair(
			Ps_norm[:, :, pair],
			Ps_norm[:, :, pair+1],
			points_norm[:, :, pair],
			points_norm[:, :, pair+1],
			Ts[:, :, pair],
			Ts[:, :, pair+1],
		)
	end

	return collect(triangulated_points)
end

# function to triangulate a pair of points given the camera matrices and points
function triangulate_pair(P1, P2, pts1, pts2, T1, T2)
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
	triangulated_points = V[begin:end-1, 4, :] ./ V[4, 4, :]'

	# Rescale the triangulated points
	return triangulated_points
end


function triangulatePoints(points, image_names, cams, datafile)
	K = []
	d = []
	R = []
	t = []
	for view in axes(points, 3)
		K1, d1, R1, t1 = extract_matrices(cams[view], image_names[view], datafile)
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
				distances[i, j, view1] = distance3D(K[view1], d[view1], R[view1], t[view1], K[view2], d[view2], R[view2], t[view2], points[:, 1, view1], points[:, j, view2])
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
	cumulative_assignments = cumulative_assignments[:, begin+1:end]

	matched_points = copy(points)
	for view in axes(points, 3)
		matched_points[:, :, view] = points[:, cumulative_assignments[:, view], view]
	end

	reconstructed_points = zeros(Float32, 3, size(points, 2), size(points, 3))

	reconstructed_points[:, :, begin:end-1] = triangulate_batched(K, R, t, matched_points)

	return distances[begin:end-1], cumulative_assignments[:, begin:end-1], reconstructed_points[:, begin:end-1, :]
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
