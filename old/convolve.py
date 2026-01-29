import math, time
import cv2

import glob
import os

import ffmpeg
import numpy as np
# from PIL import Image

def get_frame(video_path: str, frame_num: int, fps: int = 25) -> np.ndarray:
    """Extract specific frame using FFmpeg's precise seeking"""
    timestamp = frame_num / fps
    
    try:
        # Probe for video stream information
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Read frame as raw RGB bytes
        out, _ = (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e
    
    # Convert to grayscale float32 array
    frame = np.frombuffer(out, np.uint8).reshape([height, width, 3])
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)

sigma0 = 1.6
octaves = 5
layers = 5
k = pow(2, 1/(layers - 3))

gaussians = []
for i in range(1, layers+1):
    sigma = sigma0 * pow(k, i-1)
    apron = math.ceil(3 * sigma/2) * 2
    gaussian = cv2.getGaussianKernel(2*apron+1, sigma).astype(np.float32)
    gaussians.append(gaussian)
    print(f"layer {i}: apron {2*apron+1}, sigma {sigma}")
    # print(gaussian)
# print(gaussian)

# # load image from assets/DJI_20240328_234918_14_null_beauty.mp4_frame_1.png
# # img = cv2.imread("assets/DJI_20240328_234918_14_null_beauty.mp4_frame_1.png", cv2.IMREAD_GRAYSCALE)

path = "assets/videos/cam22"
video_files = (
    glob.glob(os.path.join(path, "*.mp4")) +
    glob.glob(os.path.join(path, "*.mov")) + 
    glob.glob(os.path.join(path, "*.MP4")) +
    glob.glob(os.path.join(path, "*.MOV"))
)
f = video_files[0]

v = []
v.append(get_frame(f, 339))
v.append(get_frame(f, 202))

images = len(v)

gpu_imgs = []
results = []
for i in range(images):
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(v[i])
    gpu_imgs.append(gpu_img)

t = [0.0] * 3
for i in range(images):
    for octave in range(octaves):
        outs = []
        DoG = []
        # ===================================== Gaussian =====================================
        for layer in range(layers):
            cuda_horizontal = cv2.cuda.createLinearFilter(
                srcType=cv2.CV_32F,
                dstType=cv2.CV_32F,
                kernel=gaussians[layer].reshape(1, -1),
            )
            cuda_vertical = cv2.cuda.createLinearFilter(
                srcType=cv2.CV_32F,
                dstType=cv2.CV_32F,
                kernel=gaussians[layer].reshape(-1, 1),
            )
            cv2.cuda.Stream.Null().waitForCompletion()
            start_time = time.time()
            temp = cuda_horizontal.apply(gpu_imgs[i]) 
            cv2.cuda.Stream.Null().waitForCompletion()
            end_time = time.time()
            outs.append(cuda_vertical.apply(temp))
            t[0] += (end_time - start_time)
        # ======================================== DoG =======================================
        for layer in range(octave-1):
            cv2.cuda.Stream.Null().waitForCompletion()
            start_time = time.time()
            # subtract the previous layer from the current layer, use cuda
            DoG.append(cv2.cuda.subtract(outs[layer], outs[layer+1]))
            cv2.cuda.Stream.Null().waitForCompletion()
            end_time = time.time()
            t[1] += (end_time - start_time)        

print(t)
print(f"Time taken to convolve, DoG using filter2D [{', '.join(f'{x/images:.6f}' for x in t)}] seconds per image for [Gaussian, DoG]")

# start_t = time.time()
# convolved1 = cv2.filter2D(cv2.filter2D(img, -1, gaussian1), -1, gaussian1.T)
# convolved2 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian2), -1, gaussian2.T)
# convolved3 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian3), -1, gaussian3.T)
# convolved4 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian4), -1, gaussian4.T)
# convolved5 = cv2.filter2D(cv2.filter2D(convolved, -1, gaussian5), -1, gaussian5.T)
# end_t = time.time()
# print("Time taken to convolve using filter2D ", end_t - start_t)

# # save these images
# cv2.imwrite("assets/convolved1.png", convolved1)
# cv2.imwrite("assets/convolved2.png", convolved2)
# cv2.imwrite("assets/convolved3.png", convolved3)
# cv2.imwrite("assets/convolved4.png", convolved4)
# cv2.imwrite("assets/convolved5.png", convolved5)

# start_t = time.time()
# convolved = cv2.GaussianBlur(img, (2*apron1+1, 2*apron1+1), schema["sigma"])
# convolved = cv2.GaussianBlur(convolved, (2*apron2+1, 2*apron2+1), schema["sigma"] * math.sqrt(2))
# convolved = cv2.GaussianBlur(convolved, (2*apron3+1, 2*apron3+1), schema["sigma"]* math.sqrt(2)* math.sqrt(2))
# convolved = cv2.GaussianBlur(convolved, (2*apron4+1, 2*apron4+1), schema["sigma"]* math.sqrt(2)* math.sqrt(2)* math.sqrt(2))
# convolved = cv2.GaussianBlur(convolved, (2*apron5+1, 2*apron5+1), schema["sigma"]* math.sqrt(2)* math.sqrt(2)* math.sqrt(2)* math.sqrt(2))
# end_t = time.time()
# print("Time taken to convolve using GaussianBlur ", end_t - start_t)