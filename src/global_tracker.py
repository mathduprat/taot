import sys
import os
import csv
import cv2
import numpy as np

# global_tracker_core is compiled via pybind11 and provides a native C++ FrameEngine.
# Fail immediately on missing module rather than silently crashing at first frame.
try:
    import global_tracker_core
except ImportError as e:
    sys.exit(f"[ERROR] Could not import 'global_tracker_core'. Details: {e}")


def _parse_roi(roi_str):
    """Parse a shape-tagged ROI string into (shape, coords).

    Accepts "poly:x0,y0,x1,y1,x2,y2,x3,y3" or "circle:cx,cy,r".
    A missing "shape:" prefix defaults to "poly" for backward compatibility.
    """
    if ":" in roi_str:
        shape, coords_part = roi_str.split(":", 1)
    else:
        shape, coords_part = "poly", roi_str
    coords = [int(c) for c in coords_part.split(",")]
    return shape, coords


def _draw_roi_overlay(frame, shape, coords):
    """Draw the ROI outline on frame for the live-view preview."""
    if shape == "circle":
        cx, cy, r = coords
        cv2.circle(frame, (cx, cy), r, (255, 140, 0), 2, cv2.LINE_AA)
    else:
        polygon_pts = np.array(coords, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(frame, [polygon_pts], True, (255, 140, 0), 2, cv2.LINE_AA)


def _run_live_tracking(video_path, output_csv, roi_str, scale_factor, light_on_dark=False,
                       multi_animal=True):
    """Interactive Python backend for live tracking visualization (live_view=True).

    Plays the processing pipeline frame-by-frame in an OpenCV window with a trackbar,
    pause (Space), and step controls (A/D). Used for validation and debugging only.
    Detection itself runs through FrameEngine.process_frame_py() — the same C++ pipeline
    used by the batch engine (track_video_native) — so this file never re-implements the
    coat-mask / thresholding / blob-detection logic; it only handles UI and drawing.
    """
    roi_shape, roi_coords = _parse_roi(roi_str)
    engine = global_tracker_core.FrameEngine()

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FileNotFoundError(f"CAP_FFMPEG failed to open video (MSMF fallback prevented): {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        fps = 12

    # flow_scale_inv remaps centroids from the reduced working resolution back to the original.
    # Example: centroid at (120, 80) with scale=0.5 original coordinate (240, 160).
    flow_scale_inv = 1.0 / float(scale_factor)

    WINDOW = "Live Tracking Preview"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Mutable lists are used as a workaround for the OpenCV callback closure limitation:
    # Python closures cannot rebind a plain scalar variable from an inner function.
    trackbar_pos    = [0]
    trackbar_changed= [False]
    def on_trackbar(val):
        trackbar_pos[0] = val
        trackbar_changed[0] = True
    cv2.createTrackbar("Frame", WINDOW, 0, max(total_frames - 1, 1), on_trackbar)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    paused = False
    frame_idx = 0
    last_display = None  # last rendered frame, re-shown while paused to avoid a black window

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_sec", "detected_blobs_count", "centroids_xy", "blob_sizes_px"])

        while True:
            # waitKey(30) in pause mode keeps OpenCV events responsive without spinning the CPU.
            if paused:
                wait_ms = 30
            else:
                wait_ms = 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('a'):
                frame_idx = max(0, frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            elif key == ord('d'):
                frame_idx = min(total_frames - 1, frame_idx + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            if trackbar_changed[0]:
                frame_idx = trackbar_pos[0]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                trackbar_changed[0] = False

            if paused and last_display is not None:
                cv2.imshow(WINDOW, last_display)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            cv2.setTrackbarPos("Frame", WINDOW, frame_idx)

            # Detection: delegates to the C++ engine (same pipeline as batch mode).
            contours, centroids_low, centroids_full, areas_full = engine.process_frame_py(
                frame, roi_str, light_on_dark, multi_animal, flow_scale_inv)

            out = frame.copy()
            _draw_roi_overlay(out, roi_shape, roi_coords)

            blob_count = len(centroids_full)
            for cnt_pts, (cx, cy) in zip(contours, centroids_low):
                cnt_np = np.array(cnt_pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.drawContours(out, [cnt_np], -1, (0, 255, 80), 2, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
                cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

            timestamp = round(frame_idx / fps, 3)
            writer.writerow([frame_idx, timestamp, blob_count, str(list(centroids_full)), str(list(areas_full))])

            if paused:
                pause_txt = "|| PAUSED"
            else:
                pause_txt = "|>"
            cv2.putText(out,
                        f"{pause_txt}  Mice: {blob_count} | Frame: {frame_idx}/{total_frames} | SPC=pause  A/D=step  ESC=stop",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            last_display = out
            cv2.imshow(WINDOW, out)

    cap.release()
    cv2.destroyWindow(WINDOW)


def run_global_tracking(input_video, output_csv, roi_str, scale_factor, progress_callback,
                        live_view=False, light_on_dark=False, multi_animal=True):
    """Main entry point for global tracking.

    roi_str         : shape-tagged ROI, "poly:x0,y0,...,x3,y3" or "circle:cx,cy,r".
    light_on_dark   : False (default) = dark mouse on a light background;
                      True = light mouse on a dark background (e.g. night/IR footage).
    multi_animal    : True (default) keeps every blob detected in the ROI. False assumes a
                      single animal and keeps only the largest blob each frame, discarding
                      smaller ones (noise, reflections, fragments).
    live_view=True  Python interactive backend: OpenCV window with trackbar and pause/step,
                    detection delegated per-frame to FrameEngine.process_frame_py().
    live_view=False native C++ engine (track_video_native): headless, no GIL, optimized for batch.
    flow_scale_inv is passed to the C++ engine so CSV coordinates are in the original video space.
    """
    # Inverse scale factor: maps coordinates from the working (reduced) space back to the original.
    flow_scale_inv = 1.0 / float(scale_factor)
    clean_video_path = os.path.normpath(os.path.abspath(input_video))
    clean_output_csv = os.path.normpath(os.path.abspath(output_csv))
    os.makedirs(os.path.dirname(clean_output_csv), exist_ok=True)

    progress_callback(10.0)

    if live_view:
        _run_live_tracking(clean_video_path, clean_output_csv, roi_str, scale_factor,
                           light_on_dark, multi_animal)
        progress_callback(100.0)
        return

    # Native C++ path: same pipeline as the live-view backend but headless and optimized.
    engine = global_tracker_core.FrameEngine()
    engine.track_video_native(
        str(clean_video_path),
        str(clean_output_csv),
        str(roi_str),
        bool(light_on_dark),
        bool(multi_animal),
        float(flow_scale_inv)
    )
    progress_callback(100.0)
