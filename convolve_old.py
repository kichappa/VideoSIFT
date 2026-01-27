import math, time
import cv2
schema = {
    "sigma":1.6,
    "epsilon": 0.1725,
}

apron1 = math.ceil(schema["sigma"]  * math.sqrt(-2 * math.log(schema["epsilon"])))
apron2 = math.ceil(schema["sigma"]  * math.sqrt(2) * math.sqrt(-2 * math.log(schema["epsilon"])))
apron3 = math.ceil(schema["sigma"]  * math.sqrt(2) * math.sqrt(2) * math.sqrt(-2 * math.log(schema["epsilon"])))
apron4 = math.ceil(schema["sigma"]  * math.sqrt(2) * math.sqrt(2) * math.sqrt(2) * math.sqrt(-2 * math.log(schema["epsilon"])))
apron5 = math.ceil(schema["sigma"]  * math.sqrt(2) * math.sqrt(2) * math.sqrt(2) * math.sqrt(2) * math.sqrt(-2 * math.log(schema["epsilon"])))
print(apron1, apron2, apron3, apron4, apron5)

gaussian1 = cv2.getGaussianKernel(2*apron1+1, schema["sigma"])
gaussian2 = cv2.getGaussianKernel(2*apron2+1, schema["sigma"] * math.sqrt(2))
gaussian3 = cv2.getGaussianKernel(2*apron3+1, schema["sigma"]* math.sqrt(2)* math.sqrt(2))
gaussian4 = cv2.getGaussianKernel(2*apron4+1, schema["sigma"]* math.sqrt(2)* math.sqrt(2)* math.sqrt(2))
gaussian5 = cv2.getGaussianKernel(2*apron5+1, schema["sigma"]* math.sqrt(2)* math.sqrt(2)* math.sqrt(2)* math.sqrt(2))
# print(gaussian)

# load image from assets/DJI_20240328_234918_14_null_beauty.mp4_frame_1.png
img = cv2.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_1.png", cv2.IMREAD_GRAYSCALE)

start_t = time.time()
convolved1 = cv2.filter2D(cv2.filter2D(img, -1, gaussian1), -1, gaussian1.T)
convolved2 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian2), -1, gaussian2.T)
convolved3 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian3), -1, gaussian3.T)
convolved4 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian4), -1, gaussian4.T)
convolved5 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian5), -1, gaussian5.T)
end_t = time.time()
print("Time taken to convolve using filter2D ", end_t - start_t)

# save these images
cv2.imwrite("assets/convolved1.png", convolved1)
cv2.imwrite("assets/convolved2.png", convolved2)
cv2.imwrite("assets/convolved3.png", convolved3)
cv2.imwrite("assets/convolved4.png", convolved4)
cv2.imwrite("assets/convolved5.png", convolved5)

start_t = time.time()
convolved = cv2.GaussianBlur(img, (2*apron1+1, 2*apron1+1), schema["sigma"])
convolved = cv2.GaussianBlur(convolved, (2*apron2+1, 2*apron2+1), schema["sigma"] * math.sqrt(2))
convolved = cv2.GaussianBlur(convolved, (2*apron3+1, 2*apron3+1), schema["sigma"]* math.sqrt(2)* math.sqrt(2))
convolved = cv2.GaussianBlur(convolved, (2*apron4+1, 2*apron4+1), schema["sigma"]* math.sqrt(2)* math.sqrt(2)* math.sqrt(2))
convolved = cv2.GaussianBlur(convolved, (2*apron5+1, 2*apron5+1), schema["sigma"]* math.sqrt(2)* math.sqrt(2)* math.sqrt(2)* math.sqrt(2))
end_t = time.time()
print("Time taken to convolve using GaussianBlur ", end_t - start_t)