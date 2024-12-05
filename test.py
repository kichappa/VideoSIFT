import cv2, numpy as np
from scipy.ndimage import maximum_filter

img = cv2.imread(f"assets/images/20241203_000635.mp4_frame_1.png", cv2.IMREAD_COLOR)

# blur the image
sigma = 1.6*(np.sqrt(2))**10
radius = int(3*sigma)
img0 = cv2.sepFilter2D(
            img,
            -1,
            cv2.getGaussianKernel(radius, sigma),
            cv2.getGaussianKernel(radius, sigma).T,
            borderType=cv2.BORDER_CONSTANT,
        )
#save it as "assets/blurred_0.png"
cv2.imwrite("assets/blurred_0.png", img0)


imgr = cv2.sepFilter2D(
            img,
            -1,
            cv2.getGaussianKernel(radius, sigma),
            cv2.getGaussianKernel(radius, sigma).T,
            borderType=cv2.BORDER_REFLECT_101,
        )
#save it as "assets/blurred_0.png"
cv2.imwrite("assets/blurred_r.png", imgr)

imgc = cv2.sepFilter2D(
            img,
            -1,
            cv2.getGaussianKernel(radius, sigma),
            cv2.getGaussianKernel(radius, sigma).T,
            borderType=cv2.BORDER_REPLICATE,
        )
#save it as "assets/blurred_0.png"
cv2.imwrite("assets/blurred_c.png", imgc)