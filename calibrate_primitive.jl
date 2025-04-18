using LinearAlgebra


function getOmegaMatrix(p)
    return [  0       -p[3]   p[2]; 
                p[3]     0      -p[1];
                -p[2]    p[1]   0
                ]
end


points = zeros(4, 19)

points[begin:3, 1] = [0, 10, 0]
points[begin:3, 2] = [0, 5, 0]
points[begin:3, 3] = [0, 0, 0]
points[begin:3, 4] = [5, 0, 0]
points[begin:3, 5] = [10, 0, 0]
points[begin:3, 6] = [10, 10, 0]
points[begin:3, 7] = [5, 10, 0]
points[begin:3, 8] = [1, 9, 0]
points[begin:3, 9] = [1, 6, 0]
points[begin:3, 10] = [4, 6, 0]
points[begin:3, 11] = [4, 9, 0]
points[begin:3, 12] = [1, 4, 0]
points[begin:3, 13] = [1, 1, 0]
points[begin:3, 14] = [4, 1, 0]
points[begin:3, 15] = [4, 4, 0]
points[begin:3, 16] = [5, 5, 0]
points[begin:3, 17] = [6, 9, 5]
points[begin:3, 18] = [6, 6, 5]
points[begin:3, 19] = [6, 5, 4]

points[4, :] .= 1

points[1:3,:] = points[1:3,:] .* 0.01

aspect_ratio = [1920 1080; 1920 1080]

K =  [[1.532 0 0; 0 2.723 0; 0 0 1],  [1.532 0 0; 0 2.723 0; 0 0 1]]

# xy1 = [685 975 1 ; 697 995 1 ; 726 984 1 ; 713 964 1 ; 698 1016 1 ; 727 1005 1 ; 687 999 1; 714 988 1]
# p1 = [1, 2, 3, 4, 5, 6, 7, 8]

# xy2 = [1212 1013 1 ; 1224 975 1 ; 1174 961 1 ; 1160 998 1 ; 1221 1001 1 ; 1171 987 1; 1209 1046 1 ; 1158 1031 1]
# p2 = [1, 2, 3, 4, 5, 6, 7, 8]

xy1 = [ 582 935     1;
        597 978     1;
        613 1022    1;
        670 1001    1;
        726 978     1;
        634 914     1;
        595 939     1;
        605 963     1;
        637 951     1;
        627 926     1;
        609 982     1;
        620 1010    1;
        656 996     1;
        644 968     1;
        651 956     1;
        651 906     1;
        656 922     1
        ]
p1 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]

xy2 = [ 1457 890    1;
        1477 837    1;
        1495 787    1;
        1423 766    1;
        1300 842    1;
        1379 865    1;
        1445 874    1;
        1457 844    1;
        1413 830    1;
        1402 859    1;
        1465 822    1;
        1479 792    1;
        1433 777    1;
        1420 808    1;
        1403 813    1;
        1374 791    1;
        1387 762    1
        ]
p2 = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

pXY1 = points[:, p1]
pXY2 = points[:, p2]

xy1 = ((xy1 ./ [aspect_ratio[1,:]...,1]') .- 0.5) .* [2, -2, 2]'
xy2 = ((xy2 ./ [aspect_ratio[2,:]...,1]') .- 0.5) .* [2, -2, 2]'

A1 = zeros(3 * size(p1, 1), 12)
A2 = zeros(3 * size(p2, 1), 12)

function getSubAMatrix(p, K, X)
    omegaP = [  0       -p[3]   p[2]; 
                p[3]     0      -p[1];
                -p[2]    p[1]   0
                ]
    omegaP = omegaP * K
    return kron(omegaP', X)'
end 

for i = 1:size(p1, 1)
    A1[3*i-2:3*i, :] = getSubAMatrix(xy1[i, :], K[1], pXY1[:, i])
end

for i = 1:size(p2, 1)
    A2[3*i-2:3*i, :] = getSubAMatrix(xy2[i, :], K[2], pXY2[:, i])
end

# for i = 0:size(p1, 1)-1
#     A1[3*i+1, :] = [
#         0, 0, 0, 0, 
#         -pXY1[1, i+1], -pXY1[2, i+1], -pXY1[3, i+1], -pXY1[4, i+1], 
#         xy1[i+1, 2] * pXY1[1, i+1], xy1[i+1, 2] * pXY1[2, i+1], xy1[i+1, 2] * pXY1[3, i+1], xy1[i+1, 2] * pXY1[4, i+1]
#     ]
#     # A1[3*i+1,:] = [[0, 0, 0, 0, -pXY2[:,i+1]..., (xy1[i+1, 2] .* pXY1[:, i+1])...]]
#     A1[3*i+2, :] = [
#         pXY1[1, i+1], pXY1[2, i+1], pXY1[3, i+1], pXY1[4, i+1],
#         0, 0, 0, 0,
#         -xy1[i+1, 1] * pXY1[1, i+1], -xy1[i+1, 1] * pXY1[2, i+1], -xy1[i+1, 1] * pXY1[3, i+1], -xy1[i+1, 1] * pXY1[4, i+1]
#         ]
#     A1[3*i+3, :] = [
#         -xy1[i+1, 2] * pXY1[1, i+1], -xy1[i+1, 2] * pXY1[2, i+1], -xy1[i+1, 2] * pXY1[3, i+1], -xy1[i+1, 2] * pXY1[4, i+1],
#         xy1[i+1, 1] * pXY1[1, i+1], xy1[i+1, 1] * pXY1[2, i+1], xy1[i+1, 1] * pXY1[3, i+1], xy1[i+1, 1] * pXY1[4, i+1],
#         0, 0, 0, 0
#     ] 
#     end

# for i = 0:size(p2, 1)-1
#     A2[3*i+1, :] = [
#         0, 0, 0, 0, 
#         -pXY2[1, i+1], -pXY2[2, i+1], -pXY2[3, i+1], -pXY2[4, i+1], 
#         xy2[i+1, 2] * pXY2[1, i+1], xy2[i+1, 2] * pXY2[2, i+1], xy2[i+1, 2] * pXY2[3, i+1], xy2[i+1, 2] * pXY2[4, i+1]
#     ]
#     A2[3*i+2, :] = [
#         pXY2[1, i+1], pXY2[2, i+1], pXY2[3, i+1], pXY2[4, i+1],
#         0, 0, 0, 0,
#         -xy2[i+1, 1] * pXY2[1, i+1], -xy2[i+1, 1] * pXY2[2, i+1], -xy2[i+1, 1] * pXY2[3, i+1], -xy2[i+1, 1] * pXY2[4, i+1]
#         ]
#     A2[3*i+3, :] = [
#         -xy2[i+1, 2] * pXY2[1, i+1], -xy2[i+1, 2] * pXY2[2, i+1], -xy2[i+1, 2] * pXY2[3, i+1], -xy2[i+1, 2] * pXY2[4, i+1],
#         xy2[i+1, 1] * pXY2[1, i+1], xy2[i+1, 1] * pXY2[2, i+1], xy2[i+1, 1] * pXY2[3, i+1], xy2[i+1, 1] * pXY2[4, i+1],
#         0, 0, 0, 0
#     ]
# end


# print("A1: \n[")
# for i = eachrow(A1)
#     for j in eachindex(i)
#         print(i[j], " ")
#     end
#     println()
# end
# println("]")
# print("A2: \n[")
# for i = eachrow(A2)
#     for j in eachindex(i)
#         print(i[j], " ")
#     end
#     println()
# end
# println("]")

# solve A1 * m1 = 0
U1, S1, V1 = svd(A1)
println("Singular values of A1: ", S1)

m1 = V1[:, end]

# solve A2 * m2 = 0
U2, S2, V2 = svd(A2)
println("Singular values of A2: ", S2)

m2 = V2[:, end]

# reshape m1 and m2 to 3x3 matrices
M1 = reshape(m1, 4, 3)'
M2 = reshape(m2, 4, 3)'

println("M1: ", M1)
println("M2: ", M2)

function decompose_projection(M)
    A = M[:, 1:3]
    t = M[:, 4]
    
    # RQ decomposition of A'
    Q, R = qr(A')
    K = UpperTriangular(R')  # Intrinsic matrix (upper triangular)
    R_rot = Matrix(Q')       # Rotation matrix
    
    # Ensure positive diagonal elements in K
    D = Diagonal(sign.(diag(K)))
    K = K * D
    R_rot = D * R_rot
    
    # Enforce rotation matrix properties (det = 1)
    if det(R_rot) < 0
        R_rot *= -1
        K *= -1
    end
    
    # Normalize by scaling factor (optional)
    scale = K[3,3]
    K ./= scale
    t ./= scale
    
    return K, R_rot, t
end

# K1, R1, t1 = decompose_projection(M1)
# K2, R2, t2 = decompose_projection(M2)

K1 = K[1]
R1 = M1[:, 1:3]
t1 = M1[:, 4]

K2 = K[2]
R2 = M2[:, 1:3]
t2 = M2[:, 4]

# println("K1: ", K1)
# println("R1: ", R1)
# println("t1: ", t1)

# println("K2: ", K2)
# println("R2: ", R2)
# println("t2: ", t2)

function distance(K1, R1, t1, K2, R2, t2, p1, p2)
    C1 = - R1' * t1
    C2 = - R2' * t2

    L1 = (R1' * inv(K1) * p1)
    L1 = L1 ./ norm(L1)

    L2 = (R2' * inv(K2) * p2)
    L2 = L2 ./ norm(L2)

    L1xL2 = cross(L1, L2)
    d = abs(dot(C2-C1, L1xL2)) / norm(L1xL2)
    return d
end

d = zeros(size(p1, 1), size(p2, 1))
for i = 1:size(p1, 1)
    for j = 1:size(p2, 1)
        d[i, j] = distance(K1, R1, t1, K2, R2, t2, xy1[i, :], xy2[j, :])
    end
end

# # column minimums
# d = minimum(d, dims=1)

d = hcat([0,p2...], vcat(p1', d))

# println("Distance matrix:\n", d)