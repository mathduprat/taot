import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt6.QtCore import QThread, pyqtSignal

from video_preprocessor import run_preprocessing
from global_tracker import run_global_tracking
from roi_intrusion_detector import run_roi_intrusion
from video_summarizer import compute_hourly_summary


class BatchProcessorThread(QThread):
    """Dedicated thread for batch video processing, isolated from the main UI thread.

    Uses an internal thread pool to parallelize videos and emits Qt signals
    to communicate progress and errors back to the UI.
    """

    progress_signal  = pyqtSignal(int)         # global progress percentage
    log_signal       = pyqtSignal(str)          # status message displayed in the status bar
    error_signal     = pyqtSignal(str)          # blocking error to show in a dialog
    finished_signal  = pyqtSignal()             # emitted when the full batch completes normally
    # cv2.imshow/waitKey require the main thread; this signal delegates live tracking there.
    live_view_signal = pyqtSignal(str, str, str, float, bool, bool)  # video_path, csv_path, roi_str, scale, light_on_dark, multi_animal

    def __init__(self, video_configs, global_config):
        super().__init__()
        self.video_configs = video_configs
        self.global_config = global_config
        # Beyond 4 concurrent workers, disk I/O becomes the bottleneck.
        self.max_workers = 4
        self._live_view_event = threading.Event()

    def live_view_done(self):
        """Called by the main-thread slot once live tracking has finished."""
        self._live_view_event.set()

    def process_single_video(self, item):
        """Run the full pipeline (preprocess → track → detect) for one video.

        Returns True on success, False on any blocking error.
        """
        video         = item["path"]
        base_name     = os.path.splitext(os.path.basename(video))[0]
        out_dir       = self.global_config["output_dir"]
        video_out_dir = os.path.join(out_dir, base_name)
        os.makedirs(video_out_dir, exist_ok=True)
        scale         = self.global_config["scale"]
        run_ts        = time.strftime("%Y%m%d_%H%M%S")

        self.log_signal.emit(f"[{base_name}] - Starting pipeline...")

        # 1. Video preprocessing
        if self.global_config.get("skip_preprocessing", False):
            resampled_video = video
            self.log_signal.emit(f"[{base_name}] Preprocessing skipped using video as-is.")
        else:
            fps_str         = f"{self.global_config['fps']:.0f}fps"
            scale_str       = f"{scale:.2f}x".replace(".", "p")
            resampled_video = os.path.join(video_out_dir, f"{base_name}_{fps_str}_{scale_str}.mp4")

            def ignore_progress(_):
                pass

            def forward_log(msg):
                self.log_signal.emit(f"[{base_name}] {msg}")

            try:
                run_preprocessing(
                    input_path=video,
                    target_fps=self.global_config["fps"],
                    scale=scale,
                    output_path=resampled_video,
                    progress_callback=ignore_progress,
                    log_callback=forward_log,
                    crop_rect=item.get("crop_rect"),
                    use_gpu=self.global_config.get("use_gpu", False)
                )
                self.log_signal.emit(f"[{base_name}] Downscaled video saved.")
            except Exception as e:
                self.error_signal.emit(f"Preprocessing crash for {base_name}:\n{str(e)}")
                return False

            # FFmpeg writes the file asynchronously; poll until it exists and is non-empty.
            attempts = 0
            while not os.path.exists(resampled_video) or os.path.getsize(resampled_video) == 0:
                time.sleep(0.1)
                attempts += 1
                if attempts > 50:
                    self.error_signal.emit(
                        f"Timeout: preprocessed file not found or empty: {resampled_video}"
                    )
                    return False

        #  2. GLOBAL TRACKER
        csv_output = None
        if item.get("run_tracking", True):
            csv_output = os.path.join(video_out_dir, f"{base_name}_{run_ts}_data.csv")
            try:
                if self.global_config.get("skip_preprocessing", False):
                    tracking_scale = 1.0
                else:
                    tracking_scale = scale

                def ignore_tracking_progress(_):
                    pass

                light_on_dark = item.get("light_on_dark", False)
                multi_animal  = item.get("multi_animal", False)

                if item.get("run_review", False):
                    # cv2.imshow/waitKey crash when called from a pool worker thread.
                    # Delegate to the main thread via signal and block until done.
                    self._live_view_event.clear()
                    self.live_view_signal.emit(
                        resampled_video, csv_output,
                        item["roi_4pt_scaled"], float(tracking_scale),
                        bool(light_on_dark), bool(multi_animal)
                    )
                    self._live_view_event.wait()
                else:
                    run_global_tracking(
                        input_video=resampled_video,
                        output_csv=csv_output,
                        roi_str=item["roi_4pt_scaled"],
                        scale_factor=tracking_scale,
                        progress_callback=ignore_tracking_progress,
                        live_view=False,
                        light_on_dark=light_on_dark,
                        multi_animal=multi_animal
                    )
                self.log_signal.emit(f"[{base_name}] Global tracking completed.")
            except Exception as e:
                self.error_signal.emit(f"Global tracker crash for {base_name}:\n{str(e)}")
                return False
        else:
            self.log_signal.emit(f"[{base_name}] Global tracking skipped.")

        # 3. ROI INTRUSION DETECTOR
        if item["run_intrusion"] and item["roi_rect"]:
            self.log_signal.emit(f"[{base_name}] Running ROI Intrusion Detector...")

            if self.global_config.get("skip_preprocessing", False):
                roi_str_for_analysis = item["roi_rect"]
            else:
                roi_parts = item["roi_rect"].split(",")
                rx = int(roi_parts[0])
                ry = int(roi_parts[1])
                rw = int(roi_parts[2])
                rh = int(roi_parts[3])
                crop_rect = item.get("crop_rect")
                if crop_rect:
                    rx -= crop_rect[0]
                    ry -= crop_rect[1]
                roi_str_for_analysis = (
                    f"{int(rx * scale)},{int(ry * scale)},"
                    f"{int(rw * scale)},{int(rh * scale)}"
                )

            try:
                run_roi_intrusion(
                    input_video=resampled_video,
                    output_dir=os.path.join(video_out_dir, f"roi_intrusion_{base_name}_{run_ts}"),
                    roi_str=roi_str_for_analysis,
                    min_duration=self.global_config["min_duration"],
                    calib_window=self.global_config["calib_window"],
                    detection_mode=item.get("roi_mode", "intrusion"),
                    screenshot_video=video,
                    screenshot_roi_str=item["roi_rect"]
                )
                self.log_signal.emit(f"[{base_name}] Intrusion detection completed.")
            except Exception as e:
                self.error_signal.emit(f"Intrusion detector crash for {base_name}:\n{str(e)}")
                return False

        # 4. HOURLY SUMMARY
        if csv_output is not None:
            intrusion_csv = None
            if item["run_intrusion"] and item["roi_rect"]:
                intrusion_csv = os.path.join(
                    video_out_dir, f"roi_intrusion_{base_name}_{run_ts}", "roi_intrusion_data.csv"
                )
            resume_csv = os.path.join(video_out_dir, f"{base_name}_resume.csv")
            try:
                compute_hourly_summary(csv_output, resume_csv, sessions_csv=intrusion_csv)
                self.log_signal.emit(f"[{base_name}] Hourly summary written.")
            except Exception as e:
                self.log_signal.emit(f"[{base_name}] Warning: summary failed: {e}")

        self.log_signal.emit(f"[{base_name}] - Fully Finished.")
        return True

    def run(self):
        """QThread entry point: orchestrates the worker pool and updates global progress."""
        total_videos     = len(self.video_configs)
        completed_videos = 0

        self.log_signal.emit(
            f"Initializing parallel pool with max {self.max_workers} concurrent tasks."
        )

        # Live-view videos need the main thread for OpenCV GUI — keep them out of the pool
        # and process them sequentially after the parallel batch completes.
        batch_items = [v for v in self.video_configs if not v.get("run_review", False)]
        live_items  = [v for v in self.video_configs if v.get("run_review", False)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_video, item): item
                for item in batch_items
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.error_signal.emit(f"Thread execution fatal error: {str(e)}")
                completed_videos += 1
                self.progress_signal.emit(int((completed_videos / total_videos) * 100))

        for item in live_items:
            try:
                self.process_single_video(item)
            except Exception as e:
                self.error_signal.emit(f"Thread execution fatal error: {str(e)}")
            completed_videos += 1
            self.progress_signal.emit(int((completed_videos / total_videos) * 100))

        self.finished_signal.emit()
