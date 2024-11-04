import cv2, numpy as np
from scipy.ndimage import maximum_filter

def peak_detection_scipy(dog1_r, dog2_r, dog3_r):
    rows, cols = dog2_r.shape
    
    # dog1_r, dog1_g, dog1_b = cv2.split(dog1)
    # dog2_r, dog2_g, dog2_b = cv2.split(dog2)
    # dog3_r, dog3_g, dog3_b = cv2.split(dog3)
    
    DoG_stack_r = np.stack([dog1_r, dog2_r, dog3_r], axis=-1)
    print("DoG_stack_r", DoG_stack_r)
    # DoG_stack_g = np.stack([dog1_g, dog2_g, dog3_g], axis=-1)
    # DoG_stack_b = np.stack([dog1_b, dog2_b, dog3_b], axis=-1)
    
    local_max_r = maximum_filter(DoG_stack_r, size=3)
    print("local_max_r", local_max_r)
    # local_max_g = maximum_filter(DoG_stack_g, size=3)
    # local_max_b = maximum_filter(DoG_stack_b, size=3)
    
    peak_mask_r = (dog2_r == local_max_r[:,:,1])
    print("peak_mask_r", peak_mask_r)
    # peak_mask_g = (dog2_g == local_max_g[:,:,1])
    # peak_mask_b = (dog2_b == local_max_b[:,:,1])
    
    output_r = np.where(peak_mask_r, dog2_r, 0)
    print("output_r", output_r)
    # output_g = np.where(peak_mask_g, dog2_g, 0)
    # output_b = np.where(peak_mask_b, dog2_b, 0)
    
    # output = cv2.merge([output_r, output_g, output_b])
    
    return output_r

# create a 3 3x3 arrays, random values
dog1 = np.round(np.random.rand(6, 6), 2)
print("dog1", dog1)
dog2 = np.round(np.random.rand(6, 6), 2)
print("dog2", dog2)
dog3 = np.round(np.random.rand(6, 6), 2)
print("dog3", dog3)

peak_detection_scipy(dog1, dog2, dog3)