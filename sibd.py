import math, time, cv2, numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

suffix = "_video"
# read image as rgb
# img = cv2.imread(f"assets/blobs{suffix}.png", cv2.IMREAD_COLOR)
img = cv2.imread(f"assets/images/20241203_000635.mp4_frame_1.png", cv2.IMREAD_COLOR)

scales = 5
octaves = 3

sigma0 = 1.6
k = 2 ** (1 / (scales - 3))

layer_buffer = img.copy()  # buffer to store the gaussian of the previous scale/layer
# LoG_buffers = [img.copy() for _ in range(scales-1)]
LoGs = [[img.copy() for _ in range(scales - 1)] for _ in range(octaves)]
Peak_buffers = [[img.copy() for _ in range(scales - 3)] for _ in range(octaves)]
Gaussian_buffers = [[img.copy() for _ in range(scales)] for _ in range(octaves)]
resample_buffer = img.copy()

time_taken = 0
start_t = time.time()
end_t = time.time()


# scale space peak detection
# accept 3 images
# for every pixel in the second image, check if it is a peak in the 3x3x3 neighbourhood
# if it is a peak, then accept it as a keypoint, ie retain its color as white in the output image
# every other pixel is black
# instead of returning a binary image, return the color image with the keypoints in its respective original colors
def peak_detection_scipy(
    dog1, dog2, dog3, size=3, contrast_threshold=0.03, print_output=False
):
    rows, cols, _ = dog2.shape

    dog1_r, dog1_g, dog1_b = dog1[:, :, 0], dog1[:, :, 1], dog1[:, :, 2]
    dog2_r, dog2_g, dog2_b = dog2[:, :, 0], dog2[:, :, 1], dog2[:, :, 2]
    dog3_r, dog3_g, dog3_b = dog3[:, :, 0], dog3[:, :, 1], dog3[:, :, 2]

    DoG_stack_r = np.stack([dog1_r, dog2_r, dog3_r], axis=-1)
    DoG_stack_g = np.stack([dog1_g, dog2_g, dog3_g], axis=-1)
    DoG_stack_b = np.stack([dog1_b, dog2_b, dog3_b], axis=-1)

    local_r = maximum_filter(DoG_stack_r, size=size)
    local_g = maximum_filter(DoG_stack_g, size=size)
    local_b = maximum_filter(DoG_stack_b, size=size)

    peak_mask_r = dog2_r == local_r[:, :, 1]
    peak_mask_g = dog2_g == local_g[:, :, 1]
    peak_mask_b = dog2_b == local_b[:, :, 1]

    output_r = np.where(peak_mask_r, dog2_r, 0)
    output_g = np.where(peak_mask_g, dog2_g, 0)
    output_b = np.where(peak_mask_b, dog2_b, 0)

    local_r = minimum_filter(DoG_stack_r, size=size)
    local_g = minimum_filter(DoG_stack_g, size=size)
    local_b = minimum_filter(DoG_stack_b, size=size)

    peak_mask_r = dog2_r == local_r[:, :, 1]
    peak_mask_g = dog2_g == local_g[:, :, 1]
    peak_mask_b = dog2_b == local_b[:, :, 1]

    output_r = np.where(peak_mask_r, -dog2_r, output_r)
    output_g = np.where(peak_mask_g, -dog2_g, output_g)
    output_b = np.where(peak_mask_b, -dog2_b, output_b)

    # print out outputs for (40, 63) reds, in the size x size x 3 neighbourhood
    # try:
    #     I, J = (78, 109)
    #     if print_output:
    #         for i in range(-size // 2, size // 2 + 1):
    #             for j in range(-size // 2, size // 2 + 1):
    #                 print(f"({I+i}, {J+j}): {DoG_stack_b[I+i, J+j]}")
    #         for i in range(-size // 2, size // 2 + 1):
    #             for j in range(-size // 2, size // 2 + 1):
    #                 print(f"({I+i}, {J+j}): {output_b[I+i, J+j]}")
    # except:
    #     pass

    contrast_mask_r = output_r >= contrast_threshold
    contrast_mask_g = output_g >= contrast_threshold
    contrast_mask_b = output_b >= contrast_threshold

    output_r = np.where(contrast_mask_r, output_r, 0)
    output_g = np.where(contrast_mask_g, output_g, 0)
    output_b = np.where(contrast_mask_b, output_b, 0)

    # extract the keypoints
    keypoints_r = np.where(output_r != 0)
    keypoints_g = np.where(output_g != 0)
    keypoints_b = np.where(output_b != 0)

    # print(f"Keypoints_r: {keypoints_r}")
    # print(f"Keypoints_g: {keypoints_g}")
    # print(f"Keypoints_b: {keypoints_b}")

    # keypoints_r, keypoints_g, keypoints_b are 2 lists of arrays, first with x, second with y
    # stack them together to get the keypoints
    keypoints_r = np.stack(
        [keypoints_r[0], keypoints_r[1], [0 for _ in range(len(keypoints_r[0]))]],
        axis=-1,
    )
    keypoints_g = np.stack(
        [keypoints_g[0], keypoints_g[1], [1 for _ in range(len(keypoints_g[0]))]],
        axis=-1,
    )
    keypoints_b = np.stack(
        [keypoints_b[0], keypoints_b[1], [2 for _ in range(len(keypoints_b[0]))]],
        axis=-1,
    )

    # print(f"Keypoints_r: {keypoints_r}")
    # print(f"Keypoints_g: {keypoints_g}")
    # print(f"Keypoints_b: {keypoints_b}")

    # now stack the keypoints together
    keypoints = np.concatenate([keypoints_r, keypoints_g, keypoints_b], axis=0)
    # convert keypoints to integers
    keypoints = keypoints.astype(np.int32)
    # print(f"Keypoints: {keypoints}")

    output = np.stack([output_r, output_g, output_b], axis=-1)

    return output, keypoints


def compute_orientation_histograms(image, keypoint, scale, bin_size=10, threshold=0.5):
    x, y, color_bit = keypoint
    # color_bit = 2 - color_bit
    # print(f"Keypoint: {keypoint}")
    # 16x16 window around the keypoint
    radius = int(1.5 * scale)

    gaussianWindow = cv2.getGaussianKernel(2 * radius + 1, 1.5 * scale)

    histogram = np.zeros(360 // bin_size)

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            # compute the gradient
            # fmt: off
            dx = - 1 / 3 * (4 * (int(image[x + i + 1, y + j, color_bit]) - int(image[x + i - 1, y + j, color_bit])) / 2 - (int(image[x + i + 2, y + j, color_bit]) - int(image[x + i - 2, y + j, color_bit])) / 4)
            dy =   1 / 3 * (4 * (int(image[x + i, y + j + 1, color_bit]) - int(image[x + i, y + j - 1, color_bit])) / 2 - (int(image[x + i, y + j + 2, color_bit]) - int(image[x + i, y + j - 2, color_bit])) / 4)
            # fmt: on
            # compute the magnitude and orientation
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = math.degrees(np.arctan2(dx, dy))

            if orientation < 0:
                orientation += 360

            # compute the bin
            bin = int(orientation // bin_size) % (360 // bin_size)
            histogram[bin] += (
                magnitude * gaussianWindow[i + radius] * gaussianWindow[j + radius]
            )

    # calculate coeffieint of variation of the histogram
    cv = np.std(histogram) / np.mean(histogram)

    # if keypoint is I, J, save the orientations as an xlsx file for all the +radius x +radius window
    I, J = (79, 189)
    radius += 2
    if x == I and y == J:
        with open(f"assets/blobs/orientations{suffix}.csv", "a+") as f:
            if color_bit == 0:
                f.write("position\n")
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        f.write(f"[{J+j} {I+i}],")
                    f.write("\n")
            f.write(
                f"Keypoint[{['red', 'green', 'blue'][2-color_bit]}]: {keypoint}, color: {image[I, J, color_bit], image[J, I, color_bit]}\n"
            )
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    f.write(f"{image[I+i, J+j, color_bit]},")
                f.write("\n")
            # f.write("C\n")
            # for i in range(-radius, radius + 1):
            #     for j in range(-radius, radius + 1):
            #         Cx = -(
            #             image[I + i + 1, J + j, color_bit]
            #             - image[I + i - 1, J + j, color_bit]
            #         ) / 2
            #         Cy = (
            #             image[I + i, J + j + 1, color_bit]
            #             - image[I + i, J + j - 1, color_bit]
            #         ) / 2
            #         f.write(
            #             f"{Cx:.2f} = ({image[I+i+1, J+j, color_bit]} - {image[I+i-1, J+j, color_bit]})/2 = {(image[I+i+1, J+j, color_bit] - image[I+i-1, J+j, color_bit])/2} {Cy:.2f} = ({image[I+i, J+j+1, color_bit]} - {image[I+i, J+j-1, color_bit]})/2 = {(image[I+i, J+j+1, color_bit] - image[I+i, J+j-1, color_bit])/2},"
            #         )
            #     f.write("\n")

            # f.write("D\n")
            # for i in range(-radius, radius + 1):
            #     for j in range(-radius, radius + 1):
            #         Dx = -(
            #             image[I + i + 2, J + j, color_bit]
            #             - image[I + i - 2, J + j, color_bit]
            #         ) / 4
            #         Dy = (
            #             image[I + i, J + j + 2, color_bit]
            #             - image[I + i, J + j - 2, color_bit]
            #         ) / 4
            #         f.write(
            #             f"{Dx:.2f} = ({image[I+i+2, J+j, color_bit]} - {image[I+i-2, J+j, color_bit]})/4 = {(image[I+i+2, J+j, color_bit] - image[I+i-2, J+j, color_bit])/4} {Dy:.2f} = ({image[I+i, J+j+2, color_bit]} - {image[I+i, J+j-2, color_bit]})/4 = {(image[I+i, J+j+2, color_bit] - image[I+i, J+j-2, color_bit])/4},"
            #         )
            #     f.write("\n")
            f.write("dx dy\n")
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    # fmt: off
                    dx = - 1 / 3 * (4 * (int(image[x + i + 1, y + j, color_bit]) - int(image[x + i - 1, y + j, color_bit])) / 2 - (int(image[x + i + 2, y + j, color_bit]) - int(image[x + i - 2, y + j, color_bit])) / 4)
                    dy =   1 / 3 * (4 * (int(image[x + i, y + j + 1, color_bit]) - int(image[x + i, y + j - 1, color_bit])) / 2 - (int(image[x + i, y + j + 2, color_bit]) - int(image[x + i, y + j - 2, color_bit])) / 4)
                    # fmt: on
                    # write it with 2 decimal places
                    f.write(f"'{dy:.2f}' '{dx:.2f}',")
                f.write("\n")
            f.write("orientation\n")
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    # fmt: off
                    dx = - 1 / 3 * (4 * (int(image[x + i + 1, y + j, color_bit]) - int(image[x + i - 1, y + j, color_bit])) / 2 - (int(image[x + i + 2, y + j, color_bit]) - int(image[x + i - 2, y + j, color_bit])) / 4)
                    dy =   1 / 3 * (4 * (int(image[x + i, y + j + 1, color_bit]) - int(image[x + i, y + j - 1, color_bit])) / 2 - (int(image[x + i, y + j + 2, color_bit]) - int(image[x + i, y + j - 2, color_bit])) / 4)
                    # fmt: on
                    orientation = math.degrees(np.arctan2(dx, dy))
                    if orientation < 0:
                        orientation += 360
                    f.write(f"{orientation},")
                f.write("\n")
            f.write("magnitude\n")
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    # fmt: off
                    dx = - 1 / 3 * (4 * (int(image[x + i + 1, y + j, color_bit]) - int(image[x + i - 1, y + j, color_bit])) / 2 - (int(image[x + i + 2, y + j, color_bit]) - int(image[x + i - 2, y + j, color_bit])) / 4)
                    dy =   1 / 3 * (4 * (int(image[x + i, y + j + 1, color_bit]) - int(image[x + i, y + j - 1, color_bit])) / 2 - (int(image[x + i, y + j + 2, color_bit]) - int(image[x + i, y + j - 2, color_bit])) / 4)
                    # fmt: on
                    magnitude = np.sqrt(dx**2 + dy**2)
                    f.write(f"{magnitude},")
                f.write("\n")
            f.write("gaussian weights\n")
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    try:
                        f.write(
                            f"{(gaussianWindow[i+radius] * gaussianWindow[j+radius])[0]},"
                        )
                    except:
                        f.write("0,")
                f.write("\n")
            f.close()

    # return cv
    if cv < threshold:
        return True, cv, histogram
    else:
        return False, cv, histogram
    return histogram


for octave in range(octaves):
    # for the first scale, we need to
    # OCTAVE 0: convolve the image with the gaussian kernel(sigma0)
    # REST: downsample the previous scale's scale 2 by 2, which is stored in resample_buffer
    if octave == 0:
        apron = math.ceil(3 * sigma0)
        layer_buffer = cv2.sepFilter2D(
            img,
            -1,
            cv2.getGaussianKernel(2 * apron + 1, sigma0),
            cv2.getGaussianKernel(2 * apron + 1, sigma0).T,
            borderType=cv2.BORDER_CONSTANT,
        )
    else:
        layer_buffer = cv2.resize(
            resample_buffer, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
        )
    cv2.imwrite(
        f"assets/blobs/g_blob{suffix}_{octave}_{0}_g{2**(octave)}.png", layer_buffer
    )
    Gaussian_buffers[octave][0] = layer_buffer
    for scale in range(1, scales):
        start_t = time.time()

        sigma = sigma0 * (k ** (scale - 1))
        apron = math.ceil(3 * sigma)
        temp = cv2.sepFilter2D(
            layer_buffer,
            -1,
            cv2.getGaussianKernel(2 * apron + 1, sigma),
            cv2.getGaussianKernel(2 * apron + 1, sigma).T,
            borderType=cv2.BORDER_CONSTANT,
        )
        # store the LoG buffer by subtracting the previous layer buffer
        # LoG_buffers[scale-1] = ((np.array(temp, dtype=np.float32) - np.array(layer_buffer, dtype=np.float32)) / (k-1))
        LoGs[octave][scale - 1] = (
            np.array(temp, dtype=np.float32) - np.array(layer_buffer, dtype=np.float32)
        ) / (k - 1)
        # replace the layer buffer with the current layer gaussian
        layer_buffer = temp
        Gaussian_buffers[octave][scale] = layer_buffer

        if scale == scales - 2 - 1:
            resample_buffer = temp

        end_t = time.time()
        time_taken += end_t - start_t

        if scale % 2 == 0:
            cv2.imwrite(
                f"assets/blobs/g_blob{suffix}_{octave}_{scale}_g{2**(octave) * 2**((scale)//2)}.png",
                layer_buffer,
            )
        else:
            cv2.imwrite(
                f"assets/blobs/g_blob{suffix}_{octave}_{scale}_g{2**(octave) * 2**((scale)//2)}r2.png",
                layer_buffer,
            )

        # cv2.imwrite(f"assets/blobs/dog_blob{suffix}_{octave}_{scale-1}_k{scale + octave*2}.png", ((LoG_buffers[scale-1]  - LoG_buffers[scale-1].min())/ (LoG_buffers[scale-1].max() - LoG_buffers[scale-1].min()) * 255))
        cv2.imwrite(
            f"assets/blobs/dog_blob{suffix}_{octave}_{scale-1}_k{scale + octave*2}.png",
            (
                (LoGs[octave][scale - 1] - LoGs[octave][scale - 1].min())
                / (LoGs[octave][scale - 1].max() - LoGs[octave][scale - 1].min())
                * 255
            ),
        )

min_val = min(
    [
        np.min(LoGs[octave][scale - 1])
        for octave in range(octaves)
        for scale in range(scales - 1)
    ]
)
max_val = max(
    [
        np.max(LoGs[octave][scale - 1])
        for octave in range(octaves)
        for scale in range(scales - 1)
    ]
)
LoGs = [
    [
        (LoGs[octave][scale - 1] - min_val) / (max_val - min_val) * 255
        for scale in range(scales - 1)
    ]
    for octave in range(octaves)
]

keypoints = []
histograms = []

# print blue channel of I, J in LoGs oct 0, scale 2
I, J = (78, 109)
for i in range(3):
    for j in range(3):
        print(
            f"LoGs({I+i}, {J+j}): {(LoGs[0][2][I+i, J+j]-LoGs[0][2].min())/(LoGs[0][2].max()-LoGs[0][2].min())*255}"
        )

for octave in range(octaves):
    keypoints.append([None for _ in range(scales - 3)])
    for scale in range(1, scales - 2):
        print(f"Octave {octave}, Scale {scale}")
        # peak detection
        start_t = time.time()
        # output = peak_detection_scipy(LoG_buffers[scale-3], LoG_buffers[scale-2], LoG_buffers[scale-1], 3, 0.5)
        Peak_buffers[octave][scale - 3], kp = peak_detection_scipy(
            LoGs[octave][scale - 1],
            LoGs[octave][scale],
            LoGs[octave][scale + 1],
            3,
            0.7 * 255,
            print_output=(octave == 0 and scale == 2),
        )
        keypoints[octave][scale - 3] = kp
        for keypoint in kp:
            isBlob, cv, histogram = compute_orientation_histograms(
                Gaussian_buffers[octave][scale],
                keypoint,
                sigma0 * (k ** (scale - 1)),
                30,
                threshold=0.5,
            )
            histograms.append(
                [
                    keypoint[0],
                    keypoint[1],
                    keypoint[2],
                    octave,
                    scale-1,
                    histogram,
                ]
            )
        end_t = time.time()
        time_taken += end_t - start_t

# save keypoint x, y, scale, histogram in a spreadsheet
with open(f"assets/blobs/keypoints{suffix}.csv", "w") as f:
    f.write("x,y,color,octave,scale,histogram\n")
    for histogram in histograms:
        f.write(
            f"{histogram[0]},{histogram[1]},{histogram[2]},{histogram[3]},{histogram[4]},{','.join(map(str, histogram[5]))}\n"
        )

# create filtered peaks image, retain each keypoint if compute_orentation_histograms returns True
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
            # print(
            #     f"Octave {octave}, Scale {scale-1}: {keypoint}. Is blob? {isBlob}, CV: {cv:.3f}"
            # )
            if isBlob:
                filtered_peaks[octave][scale - 1].append(keypoint)
        # if filtered_peaks[octave][scale - 1] is not None:
        #     print(f"Length: Octave {octave}, Scale {scale-1}: {len(filtered_peaks[octave][scale-3])} from {len(keypoints[octave][scale-3])}")
        # else:
        #     print(f"Length: Octave {octave}, Scale {scale-1}: 0 from {len(keypoints[octave][scale-3])}")

# save filtered peaks image
for octave in range(octaves):
    for scale in range(1, scales - 2):
        filtered_peaks_image = np.zeros_like(Peak_buffers[octave][scale - 1])
        for keypoint in filtered_peaks[octave][scale - 3]:
            filtered_peaks_image[keypoint[0], keypoint[1], keypoint[2]] = 255
        cv2.imwrite(
            f"assets/blobs/filtered_peaks{suffix}_{octave}_{scale-1}_{scale}_{scale+1}.png",
            filtered_peaks_image,
        )


# print("Keypoints")
# for octave in range(octaves):
#     for scale in range(3, scales):
#         print(f"Octave {octave}, Scale {scale-3}: {keypoints[octave][scale-3]}")

for octave in range(octaves):
    for scale in range(3, scales):
        # min_val = min([np.min(Peak_buffers[octave][scale-3][:,:,i]) for i in range(3)])
        # max_val = max([np.max(Peak_buffers[octave][scale-3][:,:,i]) for i in range(3)])
        # cv2.imwrite(f"assets/blobs/peak_scpy_blob{suffix}_{octave}_{scale-3}_{scale-2}_{scale-1}_1.png", ((Peak_buffers[octave][scale-3] - min_val) / (max_val - min_val) * 255))
        cv2.imwrite(
            f"assets/blobs/peak_scpy_blob{suffix}_{octave}_{scale-3}_{scale-2}_{scale-1}_1.png",
            Peak_buffers[octave][scale - 3],
        )

print(f"Time taken to convolve, peak detect {time_taken:.3f}")
