import math, time, cv2, numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

img = cv2.imread("assets/blobs_1.jfif", cv2.IMREAD_COLOR)

# show the image
# cv2.imshow("Image", img)

scales = 5
octaves = 5

sigma0 = 1.6
k = 2 ** (1 / (scales-3))

gaussian_buffer = img.copy()
LoG_buffers = [img.copy() for _ in range(scales-1)]
resample_buffer = img.copy()
# conv = img.copy()

time_taken = 0
start_t = time.time()
end_t = time.time()

# scale space peak detection
# accept 3 images
# for every pixel in the second image, check if it is a peak in the 3x3x3 neighbourhood
# if it is a peak, then accept it as a keypoint, ie retain its color as white in the output image
# every other pixel is black

def peak_detection(img1, img2, img3):
    rows, cols, colors = img1.shape
    output = np.zeros((rows, cols, colors), dtype=np.uint8)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            for k in range(0, colors):
                # search for a peak in the 3x3x3 neighbourhood
                if img2[i, j, k] >= img2[i-1:i+2, j-1:j+2, k].max() and img2[i, j, k] > img1[i-1:i+2, j-1:j+2, k].max() and img2[i, j, k] > img3[i-1:i+2, j-1:j+2, k].max():
                    output[i, j, k] = 255
                else:
                    output[i, j, k] = 0 
    return output

# instead of returning a binary image, return the color image with the keypoints in its respective original colors
def peak_detection_scipy(dog1, dog2, dog3, size=3):
    rows, cols, _ = dog2.shape
    
    dog1_r, dog1_g, dog1_b = cv2.split(dog1)
    dog2_r, dog2_g, dog2_b = cv2.split(dog2)
    dog3_r, dog3_g, dog3_b = cv2.split(dog3)
    
    DoG_stack_r = np.stack([dog1_r, dog2_r, dog3_r], axis=-1)
    DoG_stack_g = np.stack([dog1_g, dog2_g, dog3_g], axis=-1)
    DoG_stack_b = np.stack([dog1_b, dog2_b, dog3_b], axis=-1)
    
    local_max_r = maximum_filter(DoG_stack_r, size=size)
    local_max_g = maximum_filter(DoG_stack_g, size=size)
    local_max_b = maximum_filter(DoG_stack_b, size=size)
    
    local_min_r = minimum_filter(DoG_stack_r, size=size)
    local_min_g = minimum_filter(DoG_stack_g, size=size)
    local_min_b = minimum_filter(DoG_stack_b, size=size)
    
    peak_mask_r = (dog2_r == local_max_r[:,:,1])
    peak_mask_g = (dog2_g == local_max_g[:,:,1])
    peak_mask_b = (dog2_b == local_max_b[:,:,1])
    
    # output_r = np.where(peak_mask_r, 255, 0)
    # output_g = np.where(peak_mask_g, 255, 0)
    # output_b = np.where(peak_mask_b, 255, 0)
    
    output_r = np.where(peak_mask_r, dog2_r, 0)
    output_g = np.where(peak_mask_g, dog2_g, 0)
    output_b = np.where(peak_mask_b, dog2_b, 0)
    
    peak_mask_r = (dog2_r == local_min_r[:,:,1])
    peak_mask_g = (dog2_g == local_min_g[:,:,1])
    peak_mask_b = (dog2_b == local_min_b[:,:,1])
    
    output_r = np.where(peak_mask_r, dog2_r, output_r)
    output_g = np.where(peak_mask_g, dog2_g, output_g)
    output_b = np.where(peak_mask_b, dog2_b, output_b)
    
    output = cv2.merge([output_r, output_g, output_b])
    
    return output

for octave in range(octaves):
    for scale in range(scales):
        # print(f"Convolution for octave {octave} and scale {scale}")
        start_t = time.time()
        
        sigma = sigma0 * (k ** scale)
        apron = math.ceil(3 * sigma)
        if (scale==0):
            gaussian_buffer = cv2.sepFilter2D(img, -1, cv2.getGaussianKernel(2*apron+1, sigma), cv2.getGaussianKernel(2*apron+1, sigma).T, borderType=cv2.BORDER_CONSTANT)
        else:
            temp = cv2.sepFilter2D(gaussian_buffer, -1, cv2.getGaussianKernel(2*apron+1, sigma), cv2.getGaussianKernel(2*apron+1, sigma).T, borderType=cv2.BORDER_CONSTANT)
            # store the LoG buffer by subtracting the previous gaussian buffer 
            # and dividing by the (k-1). Use cv2.subtract and cv2.divide
            LoG_buffers[scale-1] = cv2.divide(-(temp - gaussian_buffer), (k-1))
            # LoG_buffers[scale-1] = cv2.divide(cv2.subtract(temp, gaussian_buffer), k-1)
            gaussian_buffer = temp
            if scale == 2:
                resample_buffer = temp
        
        end_t = time.time()
        time_taken += end_t - start_t
        
        cv2.imwrite(f"assets/blobs/g_blob_{octave}_{scale}_0.png", gaussian_buffer)
        if scale != 0:
            cv2.imwrite(f"assets/blobs/dog_blob_{octave}_{scale}_k{scale + octave*2}_0.png", LoG_buffers[scale-1])
            
    # peak detection
    for scale in range(1, scales-2):
        start_t = time.time()
        # output = peak_detection_scipy(LoG_buffers[scale-1], LoG_buffers[scale], LoG_buffers[scale+1], size = math.ceil(3 * sigma0 ** (scale)))
        output = peak_detection_scipy(LoG_buffers[scale-1], LoG_buffers[scale], LoG_buffers[scale+1])
        # cv2.imshow("Output", output)
        # cv2.waitKey(0)
        end_t = time.time()
        time_taken += end_t - start_t
        cv2.imwrite(f"assets/blobs/peak_scpy_blob_{octave}_{scale-1}_{scale}_{scale+1}.png", output)
    
    start_t = time.time()
    img = cv2.resize(resample_buffer, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    end_t = time.time()
    time_taken += end_t - start_t

print(f"Time taken to convolve, peak detect {time_taken:.3f}")