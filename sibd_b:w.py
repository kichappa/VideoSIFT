import math, time, cv2, numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

suffix = "_small"
img = cv2.imread(f"assets/blobs{suffix}.png", cv2.IMREAD_GRAYSCALE)


scales = 5
octaves = 3
sigma0 = 1.6
k = 2 ** (1 / (scales - 3))

layer_buffer = img.copy()
LoGs = [[img.copy() for _ in range(scales - 1)] for _ in range(octaves)]
Peak_buffers = [[img.copy() for _ in range(scales - 3)] for _ in range(octaves)]
Gaussian_buffers = [[img.copy() for _ in range(scales)] for _ in range(octaves)]
resample_buffer = img.copy()

time_taken = 0
start_t = time.time()
end_t = time.time()

def peak_detection_scipy(dog1, dog2, dog3, size=3, contrast_threshold=0.03, print_output=False):
    rows, cols = dog2.shape
    DoG_stack = np.stack([dog1, dog2, dog3], axis=-1)
    
    local = maximum_filter(DoG_stack, size=size)
    peak_mask = dog2 == local[:, :, 1]
    output = np.where(peak_mask, dog2, 0)
    
    local = minimum_filter(DoG_stack, size=size)
    peak_mask = dog2 == local[:, :, 1]
    output = np.where(peak_mask, -dog2, output)
    
    contrast_mask = output >= contrast_threshold
    output = np.where(contrast_mask, output, 0)
    
    keypoints = np.where(output != 0)
    keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
    keypoints = keypoints.astype(np.int32)
    
    return output, keypoints

def compute_orientation_histograms(image, keypoint, scale, bin_size=10, threshold=0.5):
    x, y = keypoint
    radius = int(1.5 * scale)
    gaussianWindow = cv2.getGaussianKernel(2 * radius + 1, 1.5 * scale)
    histogram = np.zeros(360 // bin_size)
    
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            dx = -1/3 * (4 * (int(image[x + i + 1, y + j]) - int(image[x + i - 1, y + j])) / 2 
                        - (int(image[x + i + 2, y + j]) - int(image[x + i - 2, y + j])) / 4)
            dy = 1/3 * (4 * (int(image[x + i, y + j + 1]) - int(image[x + i, y + j - 1])) / 2 
                       - (int(image[x + i, y + j + 2]) - int(image[x + i, y + j - 2])) / 4)
            
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = math.degrees(np.arctan2(dx, dy))
            if orientation < 0:
                orientation += 360
                
            bin = int(orientation // bin_size) % (360 // bin_size)
            histogram[bin] += magnitude * gaussianWindow[i + radius] * gaussianWindow[j + radius]
    
    cv = np.std(histogram) / np.mean(histogram)
    return (True, cv, histogram) if cv < threshold else (False, cv, histogram)

for octave in range(octaves):
    if octave == 0:
        apron = math.ceil(3 * sigma0)
        layer_buffer = cv2.sepFilter2D(
            img, -1,
            cv2.getGaussianKernel(2 * apron + 1, sigma0),
            cv2.getGaussianKernel(2 * apron + 1, sigma0).T,
            borderType=cv2.BORDER_CONSTANT
        )
    else:
        layer_buffer = cv2.resize(resample_buffer, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(f"assets/blobs/g_blob{suffix}_{octave}_{0}_g{2**(octave)}.png", layer_buffer)
    Gaussian_buffers[octave][0] = layer_buffer
    
    for scale in range(1, scales):
        start_t = time.time()
        sigma = sigma0 * (k ** (scale - 1))
        apron = math.ceil(3 * sigma)
        temp = cv2.sepFilter2D(
            layer_buffer, -1,
            cv2.getGaussianKernel(2 * apron + 1, sigma),
            cv2.getGaussianKernel(2 * apron + 1, sigma).T,
            borderType=cv2.BORDER_CONSTANT
        )
        
        LoGs[octave][scale - 1] = (np.array(temp, dtype=np.float32) - np.array(layer_buffer, dtype=np.float32)) / (k - 1)
        layer_buffer = temp
        Gaussian_buffers[octave][scale] = layer_buffer
        
        if scale == scales - 2 - 1:
            resample_buffer = temp
            
        end_t = time.time()
        time_taken += end_t - start_t
        
        cv2.imwrite(f"assets/blobs/g_blob{suffix}_{octave}_{scale}_g{2**(octave) * 2**((scale)//2)}.png", layer_buffer)
        cv2.imwrite(f"assets/blobs/dog_blob{suffix}_{octave}_{scale-1}_k{scale + octave*2}.png",
                   ((LoGs[octave][scale - 1] - LoGs[octave][scale - 1].min()) / 
                    (LoGs[octave][scale - 1].max() - LoGs[octave][scale - 1].min()) * 255))

min_val = min([np.min(LoGs[octave][scale - 1]) for octave in range(octaves) for scale in range(scales - 1)])
max_val = max([np.max(LoGs[octave][scale - 1]) for octave in range(octaves) for scale in range(scales - 1)])
LoGs = [[(LoGs[octave][scale - 1] - min_val) / (max_val - min_val) * 255 
         for scale in range(scales - 1)] for octave in range(octaves)]

keypoints = []
histograms = []

for octave in range(octaves):
    keypoints.append([None for _ in range(scales - 3)])
    for scale in range(1, scales - 2):
        print(f"Octave {octave}, Scale {scale}")
        start_t = time.time()
        
        Peak_buffers[octave][scale - 3], kp = peak_detection_scipy(
            LoGs[octave][scale - 1],
            LoGs[octave][scale],
            LoGs[octave][scale + 1],
            3, 0.7 * 255
        )
        keypoints[octave][scale - 3] = kp
        
        for keypoint in kp:
            isBlob, cv, histogram = compute_orientation_histograms(
                Gaussian_buffers[octave][scale],
                keypoint,
                sigma0 * (k ** (scale - 1)),
                30,
                threshold=0.5
            )
            histograms.append([keypoint[0], keypoint[1], octave, scale-1, histogram])
            
        end_t = time.time()
        time_taken += end_t - start_t

with open(f"assets/blobs/keypoints{suffix}.csv", "w") as f:
    f.write("x,y,octave,scale,histogram\n")
    for histogram in histograms:
        f.write(f"{histogram[0]},{histogram[1]},{histogram[2]},{histogram[3]},{','.join(map(str, histogram[4]))}\n")

filtered_peaks = []
for octave in range(octaves):
    filtered_peaks.append([[] for _ in range(scales - 3)])
    for scale in range(1, scales - 2):
        for keypoint in keypoints[octave][scale - 1]:
            isBlob, cv, histogram = compute_orientation_histograms(
                Gaussian_buffers[octave][scale],
                keypoint,
                sigma0 * (k ** (scale - 1)),
                10,
                threshold=1.5
            )
            if isBlob:
                filtered_peaks[octave][scale - 1].append(keypoint)

for octave in range(octaves):
    for scale in range(1, scales - 2):
        filtered_peaks_image = np.zeros_like(Peak_buffers[octave][scale - 1])
        for keypoint in filtered_peaks[octave][scale - 3]:
            filtered_peaks_image[keypoint[0], keypoint[1]] = 255
        cv2.imwrite(f"assets/blobs/filtered_peaks{suffix}_{octave}_{scale-1}_{scale}_{scale+1}.png",
                   filtered_peaks_image)

for octave in range(octaves):
    for scale in range(3, scales):
        cv2.imwrite(f"assets/blobs/peak_scpy_blob{suffix}_{octave}_{scale-3}_{scale-2}_{scale-1}_1.png",
                   Peak_buffers[octave][scale - 3])