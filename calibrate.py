import cv2 as cv
import numpy as np
import glob, os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image "assets/videos/2v1.png"
# img = cv.imread("assets/calib_radial.jpg")
# img = cv.imread("assets/videos/cam1/DSC_0041.JPG")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img = cv.imread("assets/2v1.png")


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cam_name = "cam14"
checkerboard_size = (11, 8)  # number of inner corners per a chessboard row and column

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) 


# look for files like "assets\\videos\\{cam_name}\\*.JPG" but of type JPG, jpg, png, PNG, jpeg, JPEG
images = glob.glob(f"assets/videos/{cam_name}/*.[jJ][pP][gG]")
images += glob.glob(f"assets/videos/{cam_name}/*.[pP][nN][gG]")
images += glob.glob(f"assets/videos/{cam_name}/*.[jJ][pP][eE][gG]")

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# for i in range(3, 11):
#     for j in range(3, 11):
#         ret, corners = cv.findChessboardCorners(gray, (i, j), None)
#         print(i, j, ret)
count = 0
for fname in images:
    if count < 10:
        # print("Processing image: ", fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, None)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (26, 26), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            # # cv.imshow('img', img)
            base_name = os.path.basename(fname)
            print(f"Saving at \"assets/videos/{cam_name}/corners/{os.path.splitext(base_name)[0]}_corners.png\"")
            # create directory if it doesn't exist
            if not os.path.exists(f"assets/videos/{cam_name}/corners"):
                os.makedirs(f"assets/videos/{cam_name}/corners")
            save = cv.imwrite(
                f"assets/videos/{cam_name}/corners/{os.path.splitext(base_name)[0]}_corners.png",
                img,
            )
        else:
            print(f"\tChessboard not found in {fname}")

# ret, mtx, dist, rvecs, tvecs, new_obj_points = cv.calibrateCameraRO(
#     objpoints, imgpoints, gray.shape[::-1], 20*3 + 16, None, None
# )

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# print the calibration error
print("ret: ", ret)


# interactive 3D plot -R'*t for each view
def plot_camera_pose(rvecs, tvecs):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv.Rodrigues(rvec)
        camera_position = -R.T @ tvec
        ax.scatter(camera_position[0], camera_position[1], camera_position[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

plot_camera_pose(rvecs, tvecs)
