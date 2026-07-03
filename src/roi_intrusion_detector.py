import cv2
import sys
import os
import csv

# Fail immediately on missing module rather than crashing silently at the first analyzed frame.
try:
    import roi_intrusion_detector_core
except ImportError as e:
    sys.exit(f"[ERROR] Could not import 'roi_intrusion_detector_core'. Details: {e}")


def _save_screenshot(source_video, timestamp_sec, output_path, rx, ry, rw, rh, avg_gray):
    """Seek to a timestamp in source_video and save the frame as a PNG with an ROI overlay
    on the frame from the original video (not the processed one).
    """
    scap = cv2.VideoCapture(os.path.normpath(os.path.abspath(source_video)), cv2.CAP_FFMPEG)
    if not scap.isOpened():
        return
    scap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = scap.read()
    scap.release()
    if not ret:
        return
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(frame, f"Avg Gray: {avg_gray:.1f}", (rx, max(ry - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, frame)


def run_roi_intrusion(input_video, output_dir, roi_str, min_duration,
                      calib_window=600, detection_mode="intrusion",
                      screenshot_video=None, screenshot_roi_str=None):
    """Detect presence sessions inside a rectangular ROI using an adaptive gray threshold.

    The threshold is derived per frame from the P85 of the last N mean-gray values
    (N = calib_window). This removes the need to tune a threshold manually per video,
    even when the mouse is already inside the ROI at the first frame.

    A session is validated when presence is continuous for at least min_duration seconds.
    A screenshot of the median frame is saved for each validated session.

    calib_window       : number of frames forming the sliding window for background estimation
                         (P85). A larger window gives a more stable baseline but adapts
                         more slowly to lighting changes.
    detection_mode     : "intrusion" (default) rejects drops so large the ROI is fully
                         covered (e.g. an animal passing over the biberon, not drinking).
                         "light" removes that upper bound, so any large enough gray drop
                         counts as presence — more false positives, but nothing is
                         excluded based on how dark the ROI gets.
    screenshot_video   : video used for screenshots (e.g. the high-resolution source).
                         Falls back to input_video if None.
    screenshot_roi_str : "x,y,w,h" in screenshot_video coordinate space for the overlay
                         rectangle. Required when the two videos have different resolutions.
    """
    clean_video_path = os.path.normpath(os.path.abspath(input_video))
    cap = cv2.VideoCapture(clean_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video stream: {input_video}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "roi_intrusion_data.csv")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        fps = 12
    analyzer = roi_intrusion_detector_core.RoiAnalyzer()

    roi_parts = roi_str.split(",")
    x = int(roi_parts[0])
    y = int(roi_parts[1])
    w = int(roi_parts[2])
    h = int(roi_parts[3])
    # Pass the history window size and mode to the C++ object for adaptive threshold computation.
    mode_int = 1 if detection_mode == "light" else 0
    analyzer.set_roi(x, y, w, h, calib_window, mode_int)

    # Overlay rect in screenshot_video space.
    # Use screenshot_roi_str if provided; otherwise fall back to the analysis coordinates.
    if screenshot_roi_str:
        screenshot_parts = screenshot_roi_str.split(",")
        ox = int(screenshot_parts[0])
        oy = int(screenshot_parts[1])
        ow = int(screenshot_parts[2])
        oh = int(screenshot_parts[3])
    else:
        ox = x
        oy = y
        ow = w
        oh = h

    if screenshot_video:
        screenshot_source = screenshot_video
    else:
        screenshot_source = input_video

    csv_buffer = [["start_frame", "end_frame", "duration_seconds", "avg_gray", "screenshot_path"]]

    session_active = False
    start_frame_idx = None
    max_duration_reached = 0.0

    current_session_records = []  # (frame_idx, timestamp_sec, avg_gray)

    # cap.get(CAP_PROP_POS_FRAMES) is unreliable after a failed read(), so track the index manually.
    last_frame_idx = 0

    while True:
        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            last_frame_idx = current_frame_idx
            break
        last_frame_idx = current_frame_idx

        timestamp_sec = current_frame_idx / fps

        # analyze_frame (C++) computes mean gray, updates the P85 buffer,
        # derives the adaptive threshold, and handles temporal debounce.
        is_present, avg_gray, duration = analyzer.analyze_frame(
            frame, timestamp_sec, min_duration
        )

        if is_present:
            if not session_active:
                session_active = True
                start_frame_idx = current_frame_idx
                current_session_records = []
            current_session_records.append((current_frame_idx, timestamp_sec, avg_gray))
            max_duration_reached = duration
        else:
            if session_active:
                end_frame_idx = current_frame_idx - 1
                if max_duration_reached >= min_duration and current_session_records:
                    # Median frame is more representative than start/end, which may show
                    # the mouse entering or leaving.
                    median_idx = len(current_session_records) // 2
                    best_frame_idx, best_ts, best_gray = current_session_records[median_idx]
                    filename = os.path.join(output_dir, f"session_frame_{best_frame_idx:05d}_median.png")
                    _save_screenshot(screenshot_source, best_ts, filename, ox, oy, ow, oh, best_gray)
                    csv_buffer.append([start_frame_idx, end_frame_idx, round(max_duration_reached, 2), round(best_gray, 1), filename])

                session_active = False
                current_session_records = []
                max_duration_reached = 0.0

    # Session still active at the end of the video.
    if session_active and max_duration_reached >= min_duration and current_session_records:
        median_idx = len(current_session_records) // 2
        best_frame_idx, best_ts, best_gray = current_session_records[median_idx]
        filename = os.path.join(output_dir, f"session_frame_{best_frame_idx:05d}_median.png")
        _save_screenshot(screenshot_source, best_ts, filename, ox, oy, ow, oh, best_gray)
        csv_buffer.append([start_frame_idx, last_frame_idx, round(max_duration_reached, 2), round(best_gray, 1), filename])

    cap.release()

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_buffer)
