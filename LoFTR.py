from kornia.feature import LoFTR
import glob, cv2, torch, numpy as np, os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resize_to_max(img, max_dim):
    """Resize *img* so its largest dimension is at most *max_dim*.

    Output dimensions are rounded down to the nearest multiple of 32, which is
    required by LoFTR's ResNet FPN backbone.  Images smaller than *max_dim* are
    never upscaled.
    """
    h, w = img.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    nh, nw = int(h * scale // 32 * 32), int(w * scale // 32 * 32)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def filter_same_position_matches(mkpts0, mkpts1, conf):
    """Remove matches where both keypoints land on the exact same pixel.

    These arise when LoFTR finds no distinctive structure and falls back to
    trivial identity matches.  Filtering them out avoids cluttering the
    visualisation with zero-displacement lines.

    Parameters
    ----------
    mkpts0, mkpts1 : np.ndarray  shape (N, 2)
    conf           : np.ndarray  shape (N,)

    Returns
    -------
    Filtered (mkpts0, mkpts1, conf) as new arrays.
    """
    mask = ~((mkpts0[:, 0] == mkpts1[:, 0]) & (mkpts0[:, 1] == mkpts1[:, 1]))
    return mkpts0[mask], mkpts1[mask], conf[mask]


def load_frame(cap, frame_idx, max_dim, device):
    """Read *frame_idx* from *cap*, resize, and return (bgr_img, gray_tensor).

    Returns (None, None) if the frame cannot be read.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, raw = cap.read()
    if not ret:
        return None, None
    bgr = resize_to_max(raw, max_dim)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(gray).float()[None, None] / 255.0  # (1,1,H,W)
    return bgr, t.to(device)


# ---------------------------------------------------------------------------
# Core: build a single correspondence canvas for one pair of frames
# ---------------------------------------------------------------------------

def draw_correspondences(
    img0_bgr, frame0_idx,
    img1_bgr, frame1_idx,
    mkpts0, mkpts1, conf,
    threshold=0.8,
    label0_suffix="",
):
    """Return a side-by-side BGR canvas with LoFTR correspondence lines blended in.

    Parameters
    ----------
    img0_bgr, img1_bgr : np.ndarray  BGR images (already resized).
    frame0_idx, frame1_idx : int     Frame numbers used for on-screen labels.
    mkpts0, mkpts1 : np.ndarray      Matched keypoint coordinates (Nx2).
    conf : np.ndarray                Per-match confidence scores (N,).
    threshold : float                Minimum confidence to draw a match.
    label0_suffix : str              Extra text appended to the left-image label
                                     (e.g. " (ref)").

    Returns
    -------
    np.ndarray  The blended canvas frame ready to pass to a VideoWriter.
    """
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    h = max(h0, h1)

    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]       = img0_bgr
    canvas[:h1, w0:w0+w1]  = img1_bgr

    cv2.putText(canvas, f"frame {frame0_idx}{label0_suffix}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame1_idx}", (w0 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    overlay = canvas.copy()
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, conf):
        if c <= threshold:
            continue
        t = (c - threshold) / (1.0 - threshold)
        color = (0, int(t * 255), int((1 - t) * 255))   # BGR: red → green
        pt0 = (int(x0),       int(y0))
        pt1 = (int(x1) + w0,  int(y1))
        cv2.line(overlay,   pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, pt0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, pt1, 2, color, -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)


# ---------------------------------------------------------------------------
# High-level: one reference frame vs. a range of frames
# ---------------------------------------------------------------------------

def plot_ref_vs_range(
    cap, frame_ref, frame_start, frame_end,
    matcher, device, max_dim,
    writer,
    threshold=0.8,
    filter_same_position=True,
):
    """Write correspondence frames comparing *frame_ref* against every frame in
    [*frame_start*, *frame_end*] (inclusive) to *writer*.

    Parameters
    ----------
    filter_same_position : bool
        When True, matches where both keypoints share the exact same pixel
        coordinates are removed before drawing (see
        ``filter_same_position_matches``).

    *writer* must already be open; it is written to but never released here so
    the caller can chain multiple calls together.
    """
    img_ref_bgr, img_ref_t = load_frame(cap, frame_ref, max_dim, device)
    assert img_ref_bgr is not None, f"Could not read reference frame {frame_ref}"

    for frame_idx in range(frame_start, frame_end + 1):
        img1_bgr, img1_t = load_frame(cap, frame_idx, max_dim, device)
        if img1_bgr is None:
            print(f"Warning: could not read frame {frame_idx}, skipping.")
            continue

        with torch.no_grad():
            corr = matcher({"image0": img_ref_t, "image1": img1_t})

        mkpts0 = corr["keypoints0"].cpu().numpy()
        mkpts1 = corr["keypoints1"].cpu().numpy()
        conf   = corr["confidence"].cpu().numpy()

        if filter_same_position:
            mkpts0, mkpts1, conf = filter_same_position_matches(mkpts0, mkpts1, conf)

        frame_out = draw_correspondences(
            img_ref_bgr, frame_ref,
            img1_bgr,    frame_idx,
            mkpts0, mkpts1, conf,
            threshold=threshold,
            label0_suffix=" (ref)",
        )

        if writer.isOpened() is False:
            raise RuntimeError("VideoWriter is not open.")
        writer.write(frame_out)

        if (frame_idx + 1) % 100 == 0:
            n = int((conf > threshold).sum())
            print(f"  frame {frame_idx} done  ({n} matches above threshold)")


# ---------------------------------------------------------------------------
# High-level: every consecutive pair [i, i+1]
# ---------------------------------------------------------------------------

def plot_consecutive_frames(
    cap, frame_start, frame_end,
    matcher, device, max_dim,
    writer,
    threshold=0.8,
    filter_same_position=True,
):
    """Write correspondence frames for every consecutive pair
    (*frame_start*, *frame_start+1*), …, (*frame_end-1*, *frame_end*).

    Parameters
    ----------
    filter_same_position : bool
        When True, matches where both keypoints share the exact same pixel
        coordinates are removed before drawing (see
        ``filter_same_position_matches``).

    Frames are loaded lazily: img1 of pair i becomes img0 of pair i+1, so each
    frame is decoded only once (except when a read fails).
    """
    img0_bgr, img0_t = load_frame(cap, frame_start, max_dim, device)
    if img0_bgr is None:
        print(f"Warning: could not read frame {frame_start}, aborting.")
        return

    for frame_idx in range(frame_start + 1, frame_end + 1):
        img1_bgr, img1_t = load_frame(cap, frame_idx, max_dim, device)
        if img1_bgr is None:
            print(f"Warning: could not read frame {frame_idx}, skipping pair.")
            continue

        with torch.no_grad():
            corr = matcher({"image0": img0_t, "image1": img1_t})

        mkpts0 = corr["keypoints0"].cpu().numpy()
        mkpts1 = corr["keypoints1"].cpu().numpy()
        conf   = corr["confidence"].cpu().numpy()

        if filter_same_position:
            mkpts0, mkpts1, conf = filter_same_position_matches(mkpts0, mkpts1, conf)

        frame_out = draw_correspondences(
            img0_bgr, frame_idx - 1,
            img1_bgr, frame_idx,
            mkpts0, mkpts1, conf,
            threshold=threshold,
        )

        writer.write(frame_out)

        if frame_idx % 100 == 0:
            n = int((conf > threshold).sum())
            print(f"  pair ({frame_idx - 1}, {frame_idx}) done  ({n} matches above threshold)")

        # slide the window
        img0_bgr, img0_t = img1_bgr, img1_t


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_writer(out_path, fps, frame_shape):
    """Create and return an opencv VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fh, fw = frame_shape[:2]
    return cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))


if __name__ == "__main__":
    cam = 31
    path = f"assets/videos/cam{cam}"
    v_files = (glob.glob(f"{path}/*.mp4") + glob.glob(f"{path}/*.MP4") +
               glob.glob(f"{path}/*.mov") + glob.glob(f"{path}/*.MOV"))
    v = v_files[0]

    # LoFTR attention is O(N^2) in spatial tokens (at 1/8 res), so full 4K
    # frames exhaust GPU memory. Dimensions must be multiples of 32.
    max_dim = 840 * 2

    threshold            = 0.8
    filter_same_position = True

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained="indoor").to(device)

    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("assets/LoFTR", exist_ok=True)

    # # ------------------------------------------------------------------
    # # Example 1: one reference frame versus a range of frames
    # # ------------------------------------------------------------------
    # frame_ref   = 900
    # frame_start = 0
    # frame_end   = 1500

    # # peek at the canvas size with a dummy load so the writer can be opened
    # _bgr, _ = load_frame(cap, frame_ref, max_dim, device)
    # assert _bgr is not None
    # _bgr1, _ = load_frame(cap, frame_start, max_dim, device)
    # assert _bgr1 is not None
    # dummy = draw_correspondences(_bgr, frame_ref, _bgr1, frame_start,
    #                              np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))

    # out_path = f"assets/LoFTR/cam{cam}_frame{frame_ref}_vs_{frame_start}-{frame_end}.mp4"
    # writer = make_writer(out_path, fps, dummy.shape)

    # plot_ref_vs_range(cap, frame_ref, frame_start, frame_end,
    #                   matcher, device, max_dim, writer,
    #                   threshold=threshold,
    #                   filter_same_position=filter_same_position)

    # writer.release()
    # print(f"Saved: {out_path}")

    # ------------------------------------------------------------------
    # Example 2: consecutive pairs over a range
    # ------------------------------------------------------------------
    consec_start = 0
    consec_end   = 1500

    _bgr0, _ = load_frame(cap, consec_start,     max_dim, device)
    _bgr1, _ = load_frame(cap, consec_start + 1, max_dim, device)
    assert _bgr0 is not None and _bgr1 is not None
    dummy = draw_correspondences(_bgr0, consec_start, _bgr1, consec_start + 1,
                                 np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))

    out_path_consec = f"assets/LoFTR/cam{cam}_consecutive_{consec_start}-{consec_end}.mp4"
    writer_consec = make_writer(out_path_consec, fps, dummy.shape)

    plot_consecutive_frames(cap, consec_start, consec_end,
                            matcher, device, max_dim, writer_consec,
                            threshold=threshold,
                            filter_same_position=filter_same_position)

    writer_consec.release()
    print(f"Saved: {out_path_consec}")

    cap.release()