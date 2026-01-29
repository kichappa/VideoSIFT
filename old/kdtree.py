import numpy as np
import scipy.spatial as spatial
import time

length_of_video = 1*60*60 # seconds
interest_points = np.float32(np.random.rand(50 * length_of_video * 30, 128))

# start time = current time
start_time = time.time()
# perform kdtree
tree = spatial.cKDTree(interest_points, leafsize=10)
# end time 
end_time = time.time()
print("Time taken to build KDTree: ", end_time - start_time)