import numpy
import cv2
import scipy
import matplotlib.pyplot as plt

# load image
img = cv2.imread('assets/20240328_225813.png')

def first_moments(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate moments
    moments = cv2.moments(gray)
    # calculate first moments
    first_moments = numpy.array([moments['m00'], moments['m01'], moments['m10']])
    return first_moments

# plot image, grayscale image and first moments
plt.figure(figsize=(20, 20))
plt.subplot(131)
# plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(132)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.title('Grayscale Image')

# First Moments
plt.subplot(133)
first_moments_array = first_moments(img)
plt.bar(['m00', 'm01', 'm10'], first_moments_array)
plt.title('First Moments')

print(first_moments_array)