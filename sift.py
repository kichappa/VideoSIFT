import numpy as np
import cv2
import matplotlib.pyplot as plt

# load assets\dataset.jpeg
image = cv2.imread('assets/dataset.jpeg')

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply a Gaussian blur to the image
gaussian = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

# apply laplacian operator to the gaussian image, with multiple kernel sizes
# show them as a grid of images
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    ks = 1 + 2 * i
    print(ks)
    # gaussian = cv2.GaussianBlur(gray, ksize=(ks, ks), sigmaX=0)
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=ks)
    # apply non-maximum suppression
    # apply non-maximum suppression
    # kernel = np.ones(((ks-1)//2, (ks-1)//2), np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    laplacian = cv2.dilate(laplacian, kernel)
    # laplacian = np.where(laplacian == local_max, laplacian, 0)
    # threasold the image
    laplacian = np.where(laplacian > 0.999, laplacian, 0)
    
    ax.imshow(laplacian, cmap='gray')
    ax.set_title(f'Kernel size: {ks}')

plt.show()

# for ks in range(1, 20, 2):
#     # apply laplacian operator to the gaussian image
#     laplacian = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=ks)

#     #show the image
#     plt.imshow(laplacian, cmap='gray')
#     plt.show()


# # create a SIFT object
# sift = cv2.SIFT_create()

# # detect SIFT keypoints and descriptors in the image
# keypoints, descriptors = sift.detectAndCompute(gray, None)

# # draw the keypoints on the image, with size and orientation
# image = cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # display the image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

# # save the image
# cv2.imwrite('assets/sift_keypoints.png', image)