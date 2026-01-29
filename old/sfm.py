import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# perform SURF on the images
def do_surf(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    org = image.copy()
    
    for block in range(2, 10):
        for ks in range(3, 10, 2):
            # detect harris corners for multiple block sizes
            dst = cv2.cornerHarris(gray, block, ks, 0.04)
            # print(dst, "\n\t dilating...\n")
            # dst = cv2.dilate(dst, None)
            ret, dst_t = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
            dst_t = np.uint8(dst_t)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_t)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
                        
            # cv2.imshow('dst_t', dst_t)
            # if cv2.waitKey(0) & 0xff == 27:
            #     cv2.destroyAllWindows()
            # cv2.imshow('ret', ret)
            # if cv2.waitKey(0) & 0xff == 27:
            #     cv2.destroyAllWindows()
            
            res = np.hstack((centroids,corners))
            # res = np.hstack((centroids))
            res = np.intp(res)
            
            # org[res[:,1],res[:,0]]=[0,0,255]
            org[res[:,3],res[:,2]] = [0,255,0]
            
            # cv2.imshow('org', org)
            # if cv2.waitKey(0) & 0xff == 27:
            #     cv2.destroyAllWindows()
            
            # print(dst)
            
            # # plot the harris corners on the image as red circle markers
            # image[dst > 0.01 * dst.max()] = [0, 0, 255]
            # cv2.imshow('image', image)
            # # store the image with the harris corners drawn on it
            cv2.imwrite(f'assets/images/harris_corners(b={block},ks={ks},a={0.04}).png', org)
        
    # if cv2.waitKey(0) & 0xff == 27:
        # cv2.destroyAllWindows()
    # implot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.plot(dst, 'ro')
    
    # plt.show()
    
    


# Function to extract features and descriptors from an image
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # use cuda instead of cpu
    # gray = cv2.cuda_GpuMat()
    # gray.upload(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    # sift = cv2.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(gray, None)
    # keypoints = sift.detect(gray, None)
    # keypoints, descriptors = sift.compute(gray, keypoints)
    
    #no cuda
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    keypoints, descriptors = sift.compute(gray, keypoints)
    
    # keypoints = do_surf(image)
    
    # use surf instead of sift, no cuda
    # surf = cv2.xfeatures2d.SURF_create()
    # keypoints, descriptors = surf.detectAndCompute(gray, None)
    
    return keypoints, descriptors

# Function to match features between two images
def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return good_matches

# Function to estimate camera pose from matched features
def estimate_camera_pose(matched_keypoints1, matched_keypoints2, K):
    E, _ = cv2.findEssentialMat(matched_keypoints1, matched_keypoints2, K)
    _, R, t, _ = cv2.recoverPose(E, matched_keypoints1, matched_keypoints2, K)
    return R, t

# Function to triangulate 3D points from matched features and camera poses
# Function to triangulate 3D points from matched features and camera poses
def triangulate_points(matched_keypoints1, matched_keypoints2, R1, t1, R2, t2, K):
    P1 = np.dot(K, np.hstack((R1, t1)))
    P2 = np.dot(K, np.hstack((R2, t2)))

    # Reshape matched keypoints to have shape (2, N)
    matched_keypoints1 = np.squeeze(matched_keypoints1).T
    matched_keypoints2 = np.squeeze(matched_keypoints2).T

    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, matched_keypoints1, matched_keypoints2)
    points_3d = points_4d_homogeneous / points_4d_homogeneous[3]  # Normalize by homogeneous coordinate
    points_3d = points_3d[:3].T  # Transpose to get points in shape (N, 3)

    return points_3d

# Load video
path_to_video = 'assets/videos/DJI_20240328_234918_14_null_beauty.mp4'
cap = cv2.VideoCapture(path_to_video)

video_name = path_to_video.split('/')[-1]
print(f"Processing video \"{video_name}\"...")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
    
# Load images
# images = [cv2.imread(f'{i:04d}.png') for i in range(0, 11)]

# Intrinsics matrix (assuming the same for all images)
focal_length = 500  # Adjust according to your camera parameters
keypoints_list = []
descriptors_list = []
images = []

# record time taken for each process for each frame, as a pandas dataframe
time_taken = pd.DataFrame(columns=['keypoints', 'matching', 'pose_estimation', 'triangulation'])

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # print(f"\tProcessing frame {len(images)}")
    
    images.append(frame)
    
    # If the frame is not read correctly, break the loop
    if not ret:
        break
    
    image_width = frame.shape[1]
    image_height = frame.shape[0]
    K = np.array([[focal_length, 0, image_width / 2],
                [0, focal_length, image_height / 2],
                [0, 0, 1]])    

    # time 
    start = cv2.getTickCount()
    # Extract features and descriptors for all images
    keypoints, descriptors = extract_features(frame)
    end = cv2.getTickCount()
    time_taken.loc[len(images) - 1, 'keypoints'] = (end - start) / cv2.getTickFrequency()
    
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)
    
    # store the features extracted for each frame as an image with the features drawn on it
    # image_with_features = cv2.drawKeypoints(frame, keypoints, None)
    
    # save the image with features
    # cv2.imwrite(f'assets/images/{video_name}_frame_{len(images)}.png', image_with_features)
    
    # print(f"\tFeatures extracted for frame {len(images)}")
    
    if len(images) == 1:
        break

print("Features extracted successfully!")

# Match features between consecutive images and perform SfM
all_3d_points = []
for i in range(len(images) - 1):
    try:
        matched_keypoints1 = keypoints_list[i]
        matched_keypoints2 = keypoints_list[i + 1]
        descriptors1 = descriptors_list[i]
        descriptors2 = descriptors_list[i + 1]
        
        # compute time taken for matching
        start = cv2.getTickCount()
        good_matches = match_features(descriptors1, descriptors2)
        end = cv2.getTickCount()
        time_taken.loc[i, 'matching'] = (end - start) / cv2.getTickFrequency()
        matched_keypoints1 = np.float32([matched_keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matched_keypoints2 = np.float32([matched_keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        start = cv2.getTickCount()
        R, t = estimate_camera_pose(matched_keypoints1, matched_keypoints2, K)
        if i == 0:
            R1 = np.eye(3)
            t1 = np.zeros((3, 1))
        else:
            R1 = prev_R
            t1 = prev_t
        prev_R = R
        prev_t = t
        end = cv2.getTickCount()
        time_taken.loc[i, 'pose_estimation'] = (end - start) / cv2.getTickFrequency()
        
        start = cv2.getTickCount()
        points_3d = triangulate_points(matched_keypoints1, matched_keypoints2, R1, t1, R, t, K)
        end = cv2.getTickCount()
        time_taken.loc[i, 'triangulation'] = (end - start) / cv2.getTickFrequency()
        all_3d_points.append(points_3d)
    except:
        print(f"Error processing frame {i}")
        continue

# Visualize the 3D points or further process them as needed
print("3D points reconstructed successfully!")

# save the time taken for each process for each frame
time_taken.to_excel(f'time_taken_CUDA_{video_name}_2.xlsx')
# Concatenate all 3D points
all_3d_points = np.concatenate(all_3d_points)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(all_3d_points[:,0], all_3d_points[:,1], all_3d_points[:,2], c='b', marker='o')

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
