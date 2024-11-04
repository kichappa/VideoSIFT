import math, time, cv2

img = cv2.imread("assets/blobs_1.jfif", cv2.IMREAD_COLOR)

# show the image
# cv2.imshow("Image", img)

layers = 5
octaves = 4

sigma0 = 1.6
k = 2 ** (1 / (layers-3))

buffer = img.copy()
conv = img.copy()

time_taken = 0
start_t = time.time()
end_t = time.time()

for octave in range(octaves):
    for layer in range(layers):
        # print(f"Convolution for octave {octave} and layer {layer}")
        start_t = time.time()
        
        sigma = sigma0 * (k ** layer)
        apron = math.ceil(3 * sigma)
        if (layer==0):
            buffer = cv2.sepFilter2D(img, -1, cv2.getGaussianKernel(2*apron+1, sigma), cv2.getGaussianKernel(2*apron+1, sigma).T, borderType=cv2.BORDER_CONSTANT)
        else:
            temp = cv2.sepFilter2D(buffer, -1, cv2.getGaussianKernel(2*apron+1, sigma), cv2.getGaussianKernel(2*apron+1, sigma).T, borderType=cv2.BORDER_CONSTANT)
            conv = -(temp - buffer) / (k-1)
            buffer = temp
        
        end_t = time.time()
        time_taken += end_t - start_t
        
        if layer != 0:
            cv2.imwrite(f"assets/blobs/dog_blob_{octave}_{layer}_k{layer + octave*2}_0.png", conv)
            
    start_t = time.time()
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    end_t = time.time()
    time_taken += end_t - start_t

print(f"Time taken to convolve using filter2D {time_taken:.3f}")