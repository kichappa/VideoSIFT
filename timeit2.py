import math, time
import cv2

import glob
import os

import ffmpeg
import numpy as np

prints = False

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
    # print(f"layer {i}: apron {2*apron+1}, sigma {sigma}")
    
for images in [pow(2, i) for i in range(6)]:
    path = "assets/videos/cam3"
    video_files = (
        glob.glob(os.path.join(path, "*.mp4")) +
        glob.glob(os.path.join(path, "*.mov")) + 
        glob.glob(os.path.join(path, "*.MP4")) +
        glob.glob(os.path.join(path, "*.MOV"))
    )
    f = video_files[0]

    v = []
    # v.append(get_frame(f, 339))
    # v.append(get_frame(f, 202))
    for i in range(images):
        try:
            v.append(get_frame(f, i))
        except:
            v.append(get_frame(f, i-256))

    # images = len(v)

    gpu_imgs = []
    results = []
    for i in range(images):
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(v[i])
        gpu_img = gpu_img.convertTo(cv2.CV_32F)
        gpu_imgs.append(gpu_img)

    runs = 100
    # ============================= Single Gaussian =============================
    t = 0
    gaussian_index = 0

    for run in range(runs):
        for i in range(images):
            img = v[i]
            # if run == 0:
            #     print(f"image {i}: {img.shape}")
            
            cuda_horizontal = cv2.cuda.createLinearFilter(
                srcType=cv2.CV_32F,
                dstType=cv2.CV_32F,
                kernel=gaussians[gaussian_index].reshape(1, -1),
            )
            cuda_vertical = cv2.cuda.createLinearFilter(
                srcType=cv2.CV_32F,
                dstType=cv2.CV_32F,
                kernel=gaussians[gaussian_index].reshape(-1, 1),
            )
            cv2.cuda.Stream.Null().waitForCompletion()
            start_time = time.time()
            temp = cuda_horizontal.apply(gpu_imgs[i]) 
            cv2.cuda.Stream.Null().waitForCompletion()
            end_time = time.time()
            if run > 0:
                t += (end_time - start_time)

    print(f"Gaussian time: {t/(images*(runs-1)):.6f} seconds per image @ {images} images at once")

    # ============================= Gaussian Pyramid =============================
    t = [0.0] * 2
    resample_buffer = cv2.cuda_GpuMat()
    for run in range(runs):
        for i in range(images):
            outs = []
            DoGs = []
            for octave in range(octaves):
                local_outs = []
                local_DoG = []
                # ===================================== Gaussian =====================================
                for layer in range(layers):
                    if layer == 0:
                        if run == 0 and i == 0 and prints:
                            print(f"\to{octave}l{layer}: new octave, reinitializing local_outs")
                        local_outs = []
                    if (octave == 0) or (octave > 0 and layer > 0):
                        if run == 0 and i == 0 and prints:
                            print(f"o{octave}l{layer}: performing Gaussian convolution", end=" ")
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
                        temp = None
                        if layer == 0:
                            if run == 0 and i == 0 and prints:
                                print("on input image")
                            temp = cuda_horizontal.apply(gpu_imgs[i]) 
                        else:
                            if run == 0 and i == 0 and prints:
                                print(f"on previous layer output")
                            temp = cuda_horizontal.apply(local_outs[layer-1])
                        out = cuda_vertical.apply(temp)
                        cv2.cuda.Stream.Null().waitForCompletion()
                        end_time = time.time()
                        local_outs.append(out)
                        if run > 0:
                            t[0] += (end_time - start_time)
                        if layer == layers - 1:
                            # append the whole layer output to outs
                            outs.append(local_outs)
                            # downsample the out[octave][layer-2] to the next octave, take alternate pixels, don't interpolate
                            if run == 0 and i == 0 and prints:
                                print(f"octave is complete. downsampling o{octave}l{layer-2} to next octave (from o{octave}l{layer})")
                            cv2.cuda.Stream.Null().waitForCompletion()
                            start_time = time.time()
                            resample_buffer = cv2.cuda.resize(local_outs[layer-2], (local_outs[layer-2].size()[0] // 2, local_outs[layer-2].size()[1] // 2), interpolation=cv2.INTER_NEAREST)
                            resample_buffer = resample_buffer.convertTo(cv2.CV_32F)
                            cv2.cuda.Stream.Null().waitForCompletion()
                            end_time = time.time()
                            if run > 0:
                                t[0] += (end_time - start_time)
                    else:
                        if run == 0 and i == 0 and prints:
                            print(f"o{octave}l{layer}: replacing output with downsampled image from previous octave ({octave-1})")
                        local_outs.append(resample_buffer)

                # ======================================== DoG =======================================
                for layer in range(layers-1):
                    if layer == 0:
                        if run == 0 and i == 0 and prints:
                            print(f"\to{octave}l{layer}: new octave, reinitializing local_DoG")                
                        local_DoG = []
                    if run == 0 and i == 0 and prints:
                        print(f"o{octave}l{layer}: performing DoG between o{octave}l{layer} and o{octave}l{layer+1}.")
                    cv2.cuda.Stream.Null().waitForCompletion()
                    start_time = time.time()
                    # subtract the previous layer from the current layer, use cuda
                    DoG = (cv2.cuda.subtract(outs[octave][layer], outs[octave][layer+1]))
                    cv2.cuda.Stream.Null().waitForCompletion()
                    end_time = time.time()
                    local_DoG.append(DoG)
                    if run > 0:
                        t[1] += (end_time - start_time)   
                    if layer == layer - 1:
                        DoGs.append(local_DoG)                
    print(f"Pyramid time: [{', '.join(f'{x/(images * (runs-1)):.6f}' for x in t)}] seconds per image @ {images} images at once")