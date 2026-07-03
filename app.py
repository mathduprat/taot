import sys
import os
import time
import json
import cv2
if getattr(sys, 'frozen', False):
    sys.path.append(os.path.join(sys._MEIPASS, "src"))
    sys.path.append(os.path.join(sys._MEIPASS, "ui"))
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(_BASE, "src"))
    sys.path.append(os.path.join(_BASE, "ui"))

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QListWidget, QLabel, QSpinBox, QDoubleSpinBox,
    QProgressBar, QMessageBox, QGroupBox, QComboBox, QTextEdit, QCheckBox,
    QDialog
)
from PyQt6.QtCore import QTimer

from batch_handler import BatchProcessorThread
from roi_window import RectRoiDialog, TrackerRoiDialog
from global_tracker import run_global_tracking

# Force UTF-8 on Windows to avoid encoding errors with accented filenames.
os.environ["PYTHONUTF8"] = "1"


class MainWindow(QWidget):
    """Main application window.

    Manages the video queue, global and per-video parameters,
    and launches the BatchProcessorThread when the pipeline starts.
    """

    def __init__(self):
        super().__init__()
        self.videos_data = []       # list of dicts, one per queued video
        self.output_directory = ""

        self.start_timestamp = 0.0
        self.elapsed_seconds = 0
        self.current_status_text = "Status: Idle"

        self.video_presets = {
            "Preset 1 (6 FPS, 0.25x)": {"fps": 6, "scale": 0.25},
            "Preset 2 (24 FPS, 0.50x)": {"fps": 24, "scale": 0.50},
            "Preset 3 (6 FPS, 0.10x)": {"fps": 6, "scale": 0.10},
            "Custom": None
        }


        self.chronometer_timer = QTimer()
        self.chronometer_timer.setInterval(1000)
        self.chronometer_timer.timeout.connect(self.update_chronometer_tick)

        self.init_ui()

    def init_ui(self):
        """Build and connect all UI widgets."""
        self.setWindowTitle("Mouse Tracker")
        self.resize(650, 760)

        main_layout = QVBoxLayout()

        group_out = QGroupBox("Global Output & Session")
        out_vlay = QVBoxLayout()

        out_row = QHBoxLayout()
        self.lbl_out_dir = QLabel("No output directory selected (Required)")
        self.lbl_out_dir.setWordWrap(True)
        btn_out_dir = QPushButton("Browse...")
        btn_out_dir.clicked.connect(self.select_output_directory)
        out_row.addWidget(self.lbl_out_dir, 1)
        out_row.addWidget(btn_out_dir)
        out_vlay.addLayout(out_row)

        btn_import = QPushButton("Import Settings from JSON...")
        btn_import.clicked.connect(self._import_settings_json)
        out_vlay.addWidget(btn_import)

        group_out.setLayout(out_vlay)
        main_layout.addWidget(group_out)

        list_and_config = QHBoxLayout()

        left_video_panel = QVBoxLayout()
        left_video_panel.addWidget(QLabel("<b>Input Videos Queue:</b>"))

        self.btn_select = QPushButton("Add Videos (Batch...)")
        self.btn_select.clicked.connect(self.select_videos)
        left_video_panel.addWidget(self.btn_select)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.load_video_settings)
        left_video_panel.addWidget(self.list_widget, 1)

        list_actions_layout = QHBoxLayout()
        self.btn_remove_item = QPushButton("Remove Selected")
        self.btn_remove_item.clicked.connect(self.remove_selected_video)
        self.btn_remove_item.setEnabled(False)

        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_video_queue)
        self.btn_clear_all.setEnabled(False)

        list_actions_layout.addWidget(self.btn_remove_item)
        list_actions_layout.addWidget(self.btn_clear_all)
        left_video_panel.addLayout(list_actions_layout)

        list_and_config.addLayout(left_video_panel, 1)

        self.group_indiv = QGroupBox("Selected Video Settings")
        indiv_layout = QVBoxLayout()

        self.chk_run_tracking = QCheckBox("Run Global Tracking")
        self.chk_run_tracking.setChecked(True)
        self.chk_run_tracking.toggled.connect(self.toggle_tracking_state)
        indiv_layout.addWidget(self.chk_run_tracking)

        self.btn_roi_tracker = QPushButton("Set Tracker ROI (Polygon or Circle)")
        self.btn_roi_tracker.clicked.connect(self.select_tracker_roi)
        indiv_layout.addWidget(self.btn_roi_tracker)

        self.chk_light_on_dark = QCheckBox("White mouse on dark background (Otsu threshold)")
        self.chk_light_on_dark.toggled.connect(self.toggle_light_on_dark_state)
        indiv_layout.addWidget(self.chk_light_on_dark)

        self.chk_multi_animal = QCheckBox("Multiple animals")
        self.chk_multi_animal.setToolTip(
            "Unchecked (default): only one animal is expected in the ROI — only the\n"
            "largest detected blob is kept each frame.\n"
            "Checked: every detected blob is kept, for multiple animals in the same ROI."
        )
        self.chk_multi_animal.toggled.connect(self.toggle_multi_animal_state)
        indiv_layout.addWidget(self.chk_multi_animal)

        self.chk_live_vis = QCheckBox("Live Visualisation (Not recommended slower process)")
        self.chk_live_vis.toggled.connect(self.toggle_review_state)
        indiv_layout.addWidget(self.chk_live_vis)

        self.chk_run_intrusion = QCheckBox("Run ROI Intrusion Analysis")
        self.chk_run_intrusion.toggled.connect(self.toggle_intrusion_state)
        indiv_layout.addWidget(self.chk_run_intrusion)

        self.btn_roi_intrus = QPushButton("Set Intrusion ROI (Rect)")
        self.btn_roi_intrus.clicked.connect(self.select_rect_roi)
        indiv_layout.addWidget(self.btn_roi_intrus)

        self.lbl_indiv_status = QLabel("ROIs: Not configured")
        self.lbl_indiv_status.setWordWrap(True)
        indiv_layout.addWidget(self.lbl_indiv_status)

        self.group_indiv.setLayout(indiv_layout)
        # Disabled until a video is selected in the list.
        self.group_indiv.setEnabled(False)

        list_and_config.addWidget(self.group_indiv, 1)
        main_layout.addLayout(list_and_config)

        group_params = QGroupBox("Global Pipeline Parameters")
        param_layout = QVBoxLayout()

        param_layout.addWidget(QLabel("<b>Video Downscaling</b>"))
        v_preset_lay = QHBoxLayout()
        v_preset_lay.addWidget(QLabel("Preset:"))
        self.combo_v_preset = QComboBox()
        self.combo_v_preset.addItems(self.video_presets.keys())
        self.combo_v_preset.currentTextChanged.connect(self.apply_video_preset)
        v_preset_lay.addWidget(self.combo_v_preset, 1)
        param_layout.addLayout(v_preset_lay)

        v_lay = QHBoxLayout()
        v_lay.addWidget(QLabel("Target FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(12)
        self.spin_fps.valueChanged.connect(self.set_video_custom)
        v_lay.addWidget(self.spin_fps)

        v_lay.addWidget(QLabel("Scale Factor:"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.05, 1.0)
        self.spin_scale.setSingleStep(0.05)
        self.spin_scale.setValue(0.25)
        self.spin_scale.valueChanged.connect(self.set_video_custom)
        v_lay.addWidget(self.spin_scale)
        param_layout.addLayout(v_lay)

        param_layout.addWidget(QLabel("<b>ROI Intrusion Analyzer</b>"))

        r_lay = QHBoxLayout()
        r_lay.addWidget(QLabel("Min Dur (s):"))
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 3600.0)
        self.spin_duration.setValue(4.0)
        r_lay.addWidget(self.spin_duration)
        lbl_dur_info = QLabel("Minimal duration for a valid intrusion")
        lbl_dur_info.setStyleSheet("color: gray; font-style: italic;")
        r_lay.addWidget(lbl_dur_info)
        param_layout.addLayout(r_lay)

        param_layout.addWidget(QLabel("<b>Preprocessing Encoder</b>"))
        self.chk_skip_preprocessing = QCheckBox("Skip preprocessing (videos are already at the correct FPS/scale)")
        self.chk_skip_preprocessing.setChecked(False)
        self.chk_skip_preprocessing.toggled.connect(self._on_skip_preprocessing_toggled)
        param_layout.addWidget(self.chk_skip_preprocessing)

        self.chk_use_gpu = QCheckBox("Use GPU acceleration (NVENC/VAAPI) — not always faster than CPU")
        self.chk_use_gpu.setChecked(False)
        param_layout.addWidget(self.chk_use_gpu)

        # Widgets to gray out when preprocessing is skipped.
        self._preprocessing_widgets = [self.combo_v_preset, self.spin_fps, self.spin_scale, self.chk_use_gpu]

        group_params.setLayout(param_layout)
        main_layout.addWidget(group_params)

        self.lbl_status = QLabel(self.current_status_text)
        main_layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.btn_start = QPushButton("Run Batch Pipeline")
        self.btn_start.clicked.connect(self.start_batch_processing)
        # Disabled until output dir, videos, and all ROIs are configured.
        self.btn_start.setEnabled(False)
        main_layout.addWidget(self.btn_start)

        self.setLayout(main_layout)

    def _on_skip_preprocessing_toggled(self, checked):
        """Gray out FPS/scale/GPU controls when preprocessing is skipped — they have no effect."""
        for w in self._preprocessing_widgets:
            w.setEnabled(not checked)
        self.update_run_button_state()

    def apply_video_preset(self, text):
        """Apply a FPS/scale preset without triggering the Custom fallback on each spinbox change.

        Signals are blocked during the update to avoid each value change switching the combo to Custom.
        """
        preset = self.video_presets[text]
        if preset is not None:
            self.spin_fps.blockSignals(True)
            self.spin_scale.blockSignals(True)
            self.spin_fps.setValue(preset["fps"])
            self.spin_scale.setValue(preset["scale"])
            self.spin_fps.blockSignals(False)
            self.spin_scale.blockSignals(False)

    def set_video_custom(self):
        """Switch the video combo to 'Custom' when FPS or scale is edited manually.

        Signal is blocked to avoid recursion: changing the combo text would trigger apply_video_preset.
        """
        self.combo_v_preset.blockSignals(True)
        self.combo_v_preset.setCurrentText("Custom")
        self.combo_v_preset.blockSignals(False)

    def select_output_directory(self):
        """Open a folder picker and update the global output path."""
        folder = QFileDialog.getExistingDirectory(self, "Select Global Output Directory")
        if folder:
            self.output_directory = folder
            self.lbl_out_dir.setText(folder)
            self.update_run_button_state()

    def select_videos(self):
        """Add videos to the queue with default empty ROIs and disabled analyses."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Videos (*.mp4 *.avi *.mkv)")
        if files:
            for f in files:
                self.videos_data.append({
                    "path": f,
                    "roi_4pt": "",
                    "roi_rect": "",
                    "roi_mode": "intrusion",
                    "run_tracking": True,
                    "run_intrusion": False,
                    "run_review": False,
                    "light_on_dark": False,
                    "multi_animal": False,
                })
                self.list_widget.addItem(os.path.basename(f))
            self.update_run_button_state()
            self.update_queue_buttons_state()

    def remove_selected_video(self):
        """Remove the selected video from both the list widget and the data array."""
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.list_widget.takeItem(idx)
            self.videos_data.pop(idx)
            self.update_run_button_state()
            self.update_queue_buttons_state()

    def clear_video_queue(self):
        """Clear the entire video queue."""
        self.list_widget.clear()
        self.videos_data.clear()
        self.update_run_button_state()
        self.update_queue_buttons_state()

    def update_queue_buttons_state(self):
        """Enable/disable queue action buttons based on current queue state."""
        has_items = (len(self.videos_data) > 0)
        self.btn_clear_all.setEnabled(has_items)
        self.btn_remove_item.setEnabled(self.list_widget.currentRow() >= 0)

    def load_video_settings(self, index):
        """Refresh the per-video settings panel for the selected video."""
        if index < 0 or index >= len(self.videos_data):
            self.group_indiv.setEnabled(False)
            self.btn_remove_item.setEnabled(False)
            return

        self.group_indiv.setEnabled(True)
        self.btn_remove_item.setEnabled(True)
        data = self.videos_data[index]

        # Block signals to prevent programmatic checkbox updates from triggering toggle_ slots.
        self.chk_run_tracking.blockSignals(True)
        self.chk_run_tracking.setChecked(data.get("run_tracking", True))
        self.chk_run_tracking.blockSignals(False)

        tracking_on = data.get("run_tracking", True)
        self.btn_roi_tracker.setEnabled(tracking_on)
        self.chk_light_on_dark.setEnabled(tracking_on)
        self.chk_multi_animal.setEnabled(tracking_on)
        self.chk_live_vis.setEnabled(tracking_on)

        self.chk_light_on_dark.blockSignals(True)
        self.chk_light_on_dark.setChecked(data.get("light_on_dark", False))
        self.chk_light_on_dark.blockSignals(False)

        self.chk_multi_animal.blockSignals(True)
        self.chk_multi_animal.setChecked(data.get("multi_animal", False))
        self.chk_multi_animal.blockSignals(False)

        self.chk_live_vis.blockSignals(True)
        self.chk_live_vis.setChecked(data["run_review"])
        self.chk_live_vis.blockSignals(False)

        self.chk_run_intrusion.blockSignals(True)
        self.chk_run_intrusion.setChecked(data["run_intrusion"])
        self.chk_run_intrusion.blockSignals(False)

        self.btn_roi_intrus.setEnabled(data["run_intrusion"])
        self.update_indiv_roi_label(data)

    def toggle_tracking_state(self, checked):
        """Enable/disable global tracking for the current video and refresh the UI."""
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.videos_data[idx]["run_tracking"] = checked
            self.btn_roi_tracker.setEnabled(checked)
            self.chk_live_vis.setEnabled(checked)
            self.update_indiv_roi_label(self.videos_data[idx])
            self.update_run_button_state()

    def toggle_light_on_dark_state(self, checked):
        """Store the white-mouse-on-dark-background flag for the current video."""
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.videos_data[idx]["light_on_dark"] = checked

    def toggle_multi_animal_state(self, checked):
        """Store the multi-animal flag for the current video."""
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.videos_data[idx]["multi_animal"] = checked

    def toggle_review_state(self, checked):
        """Store the live view flag for the current video."""
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.videos_data[idx]["run_review"] = checked

    def toggle_intrusion_state(self, checked):
        """Enable/disable intrusion analysis for the current video and refresh the UI."""
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.videos_data[idx]["run_intrusion"] = checked
            self.btn_roi_intrus.setEnabled(checked)
            self.update_indiv_roi_label(self.videos_data[idx])
            # Re-evaluate: enabling intrusion without a rect ROI should disable the Run button.
            self.update_run_button_state()

    @staticmethod
    def _roi_shape(roi_str):
        """Return the tracker ROI shape ("circle" or "poly") encoded in a shape-tagged string."""
        if roi_str.startswith("circle:"):
            return "circle"
        return "poly"

    def select_tracker_roi(self):
        """Open the tracker ROI dialog (shape chosen inside the dialog: polygon or circle)."""
        idx = self.list_widget.currentRow()
        if idx < 0:
            return
        cap = cv2.VideoCapture(self.videos_data[idx]["path"])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        existing = self.videos_data[idx].get("roi_4pt", "")

        dlg = TrackerRoiDialog(frame, initial_roi_str=existing, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_roi = dlg.get_roi_str()
        if not new_roi:
            return

        self.videos_data[idx]["roi_4pt"] = new_roi
        for i, v in enumerate(self.videos_data):
            if i != idx and not v.get("roi_4pt", ""):
                v["roi_4pt"] = new_roi
        self.update_indiv_roi_label(self.videos_data[idx])
        self.update_run_button_state()

    def select_rect_roi(self):
        """Open the rectangle ROI dialog to define the intrusion zone."""
        idx = self.list_widget.currentRow()
        if idx < 0:
            return
        cap = cv2.VideoCapture(self.videos_data[idx]["path"])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        existing = self.videos_data[idx].get("roi_rect", "")
        initial_roi = None
        if existing:
            parts = existing.split(",")
            if len(parts) == 4:
                initial_roi = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        initial_mode = self.videos_data[idx].get("roi_mode", "intrusion")

        dlg = RectRoiDialog(frame, initial_roi=initial_roi, initial_mode=initial_mode, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        result = dlg.get_roi()
        if result and result[2] > 0 and result[3] > 0:
            x, y, w, h = result
            new_roi = f"{x},{y},{w},{h}"
            new_mode = dlg.get_mode()
            self.videos_data[idx]["roi_rect"] = new_roi
            self.videos_data[idx]["roi_mode"] = new_mode
            for i, v in enumerate(self.videos_data):
                if i != idx and not v.get("roi_rect", ""):
                    v["roi_rect"] = new_roi
                    v["roi_mode"] = new_mode
            self.update_indiv_roi_label(self.videos_data[idx])
            self.update_run_button_state()

    def update_indiv_roi_label(self, data):
        """Update the ROI status label to reflect the current configuration state."""
        if data.get("run_tracking", True):
            if data["roi_4pt"]:
                t_ok = "Tracker: Configured"
            else:
                t_ok = "Tracker: Missing ROI"
        else:
            t_ok = "Tracking: Disabled"

        if data["run_intrusion"]:
            if data["roi_rect"]:
                mode_label = "light" if data.get("roi_mode", "intrusion") == "light" else "intrusion"
                i_ok = f"Intrusion: Configured ({mode_label})"
            else:
                i_ok = "Intrusion: Missing Rect"
        else:
            i_ok = "Intrusion: Disabled"

        self.lbl_indiv_status.setText(f"Status:\n- {t_ok}\n- {i_ok}")

    def update_run_button_state(self):
        """Enable the Run button only when all required conditions are met.

        Requires: output directory set, at least one video, tracker ROI on every video,
        and intrusion ROI on every video that has intrusion enabled.
        """
        if not self.output_directory or not self.videos_data:
            self.btn_start.setEnabled(False)
            return

        for item in self.videos_data:
            if item.get("run_tracking", True) and not item["roi_4pt"]:
                self.btn_start.setEnabled(False)
                return
            if item["run_intrusion"] and not item["roi_rect"]:
                self.btn_start.setEnabled(False)
                return

        self.btn_start.setEnabled(True)

    def handle_processor_error(self, message):
        """Display a scrollable error dialog with the full error message from the batch thread."""
        # Stop the timer immediately the pipeline is no longer running.
        self.chronometer_timer.stop()

        dialog = QDialog(self)
        dialog.setWindowTitle("Pipeline Subprocess Error")
        dialog.resize(560, 360)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("An error occurred during the batch execution sequence.\nReview the full logs below to identify the issue:"))
        log_viewer = QTextEdit()
        log_viewer.setReadOnly(True)
        log_viewer.setPlainText(message)
        layout.addWidget(log_viewer)
        btn = QPushButton("OK")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.exec()

    def update_chronometer_tick(self):
        """Update the elapsed time display in the status bar every second."""
        self.elapsed_seconds = int(time.time() - self.start_timestamp)
        mins, secs = divmod(self.elapsed_seconds, 60)
        time_string = f"{mins:02d}:{secs:02d}"
        self.lbl_status.setText(f"{self.current_status_text} (Elapsed Time: {time_string})")

    def _collect_global_settings(self):
        """Return the current UI state as a flat dict (internal key format).

        Single source of truth for all code that needs to read global parameters
        (start_batch_processing, export, etc.) to avoid duplicating widget reads.
        """
        return {
            "output_dir": self.output_directory,
            "fps": self.spin_fps.value(),
            "scale": self.spin_scale.value(),
            "calib_window": 600,
            "min_duration": self.spin_duration.value(),
            "use_gpu": self.chk_use_gpu.isChecked(),
            "skip_preprocessing": self.chk_skip_preprocessing.isChecked(),
        }

    def _apply_global_settings(self, cfg):
        """Apply a settings dict to all UI widgets.

        Accepts both internal keys (fps, scale) and JSON export keys (target_fps, scale_factor)
        so the same method works whether loading from an exported JSON or an internal snapshot.
        After applying values, preset combos are synced to show a preset name or 'Custom'.
        """
        # Output directory only apply if the path actually exists on this machine.
        out_dir = cfg.get("output_dir")
        if out_dir and os.path.isdir(out_dir):
            self.output_directory = out_dir
            self.lbl_out_dir.setText(out_dir)

        # Block spinbox signals to prevent each setValue from toggling the combo to "Custom".
        spinboxes = [self.spin_fps, self.spin_scale, self.spin_duration]
        for w in spinboxes:
            w.blockSignals(True)

        if "fps" in cfg:
            self.spin_fps.setValue(int(cfg["fps"]))
        elif "target_fps" in cfg:
            self.spin_fps.setValue(int(cfg["target_fps"]))

        if "scale" in cfg:
            self.spin_scale.setValue(float(cfg["scale"]))
        elif "scale_factor" in cfg:
            self.spin_scale.setValue(float(cfg["scale_factor"]))

        if "min_duration" in cfg:
            self.spin_duration.setValue(float(cfg["min_duration"]))
        elif "intrusion_min_duration_sec" in cfg:
            self.spin_duration.setValue(float(cfg["intrusion_min_duration_sec"]))
        elif "roi_intrusion_min_duration_sec" in cfg:
            self.spin_duration.setValue(float(cfg["roi_intrusion_min_duration_sec"]))

        for w in spinboxes:
            w.blockSignals(False)

        if "use_gpu" in cfg:
            self.chk_use_gpu.setChecked(bool(cfg["use_gpu"]))
        if "skip_preprocessing" in cfg:
            self.chk_skip_preprocessing.setChecked(bool(cfg["skip_preprocessing"]))

        # Sync video preset combo show matching preset name, or fall back to "Custom".
        self.combo_v_preset.blockSignals(True)
        video_preset_found = False
        for name, preset in self.video_presets.items():
            if preset is not None:
                fps_matches = (preset["fps"] == self.spin_fps.value())
                scale_matches = (abs(preset["scale"] - self.spin_scale.value()) < 0.001)
                if fps_matches and scale_matches:
                    self.combo_v_preset.setCurrentText(name)
                    video_preset_found = True
                    break
        if not video_preset_found:
            self.combo_v_preset.setCurrentText("Custom")
        self.combo_v_preset.blockSignals(False)


        self.update_run_button_state()

    def _import_settings_json(self):
        """Load a previously exported settings JSON and restore the full session state.

        Global settings (FPS, scale, thresholds, flags) are applied immediately.
        Videos listed in the JSON are added to the queue if their file exists on disk —
        the original full_path is tried first, then the same directory as the JSON file
        as a fallback (useful when the project folder was moved).
        If the queue already has videos, the user is asked whether to replace it or merge.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON Files (*.json)")
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Import Error", f"Could not read the JSON file:\n{e}")
            return

        # Confirm before clearing an existing queue.
        if self.videos_data:
            answer = QMessageBox.question(
                self, "Import Settings",
                "Replace the current video queue with the imported session?\n\n"
                "Yes = clear and replace   |   No = merge (skip duplicates)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.list_widget.clear()
                self.videos_data.clear()

        # Apply global settings.
        self._apply_global_settings(data.get("global_settings", {}))

        # Add videos.
        json_dir = os.path.dirname(path)
        existing_paths = set()
        for d in self.videos_data:
            existing_paths.add(d["path"])
        missing = []

        for v in data.get("videos", []):
            video_path = v.get("path") or v.get("full_path", "")
            if not os.path.isfile(video_path):
                # Fallback: look for the file next to the JSON (moved project folder).
                candidate = os.path.join(json_dir, v.get("filename", ""))
                if os.path.isfile(candidate):
                    video_path = candidate
                else:
                    missing.append(v.get("filename", "?"))
                    continue

            if video_path in existing_paths:
                continue  # already in queue skip silently
            existing_paths.add(video_path)

            if v.get("tracker_roi"):
                tracker_roi = v["tracker_roi"]
            elif v.get("roi_tracker_original"):
                tracker_roi = v["roi_tracker_original"]
            else:
                tracker_roi = ""

            if "tracking_enabled" in v:
                run_tracking = v["tracking_enabled"]
            elif "run_tracking" in v:
                run_tracking = v["run_tracking"]
            else:
                run_tracking = bool(tracker_roi)

            if v.get("intrusion_roi"):
                roi_rect_value = v["intrusion_roi"]
            elif v.get("roi_intrusion_rect_xywh"):
                roi_rect_value = v["roi_intrusion_rect_xywh"]
            else:
                roi_rect_value = ""

            roi_mode_value = v.get("roi_mode", "intrusion")
            if roi_mode_value not in ("intrusion", "light"):
                roi_mode_value = "intrusion"

            if "intrusion_enabled" in v:
                run_intrusion = v["intrusion_enabled"]
            elif "run_intrusion" in v:
                run_intrusion = v["run_intrusion"]
            else:
                run_intrusion = False

            if "live_review" in v:
                run_review = v["live_review"]
            elif "run_live_review" in v:
                run_review = v["run_live_review"]
            else:
                run_review = False

            light_on_dark = bool(v.get("light_on_dark", False))
            multi_animal  = bool(v.get("multi_animal", False))

            self.videos_data.append({
                "path": video_path,
                "roi_4pt": tracker_roi,
                "roi_rect": roi_rect_value,
                "roi_mode": roi_mode_value,
                "run_tracking": run_tracking,
                "run_intrusion": run_intrusion,
                "run_review": run_review,
                "light_on_dark": light_on_dark,
                "multi_animal": multi_animal,
            })
            self.list_widget.addItem(os.path.basename(video_path))

        if missing:
            QMessageBox.warning(
                self, "Import Warning",
                "The following videos were not found on disk and were skipped:\n\n" + "\n".join(missing)
            )

        self.update_run_button_state()
        self.update_queue_buttons_state()

    def _build_roi_for_item(self, item, scale, skip_preprocessing):
        """Compute the scaled ROI string and FFmpeg crop rect from raw tracker coordinates.

        When preprocessing is skipped, the video is not cropped or resized, so the original
        coordinates are used as-is. Otherwise, coordinates are offset to the crop bounding box
        origin and scaled to match the preprocessed video resolution.
        Returns (roi_scaled_str, crop_rect_or_none), where roi_scaled_str keeps the same
        "poly:"/"circle:" shape prefix as the input.
        """
        if skip_preprocessing:
            return item["roi_4pt"], None

        shape = self._roi_shape(item["roi_4pt"])
        coords_part = item["roi_4pt"].split(":", 1)[-1]
        raw_coords = [int(c) for c in coords_part.split(",")]

        if shape == "circle":
            cx, cy, r = raw_coords
            min_x, min_y = max(0, cx - r), max(0, cy - r)
            crop_rect = (min_x, min_y, (cx - min_x) + r, (cy - min_y) + r)
            roi_scaled_str = "circle:{},{},{}".format(
                int((cx - min_x) * scale), int((cy - min_y) * scale), int(r * scale)
            )
            return roi_scaled_str, crop_rect

        xs = raw_coords[0::2]
        ys = raw_coords[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        crop_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Re-center coordinates to crop origin, then apply scale factor.
        adjusted = []
        for i, c in enumerate(raw_coords):
            offset = min_x if i % 2 == 0 else min_y
            adjusted.append(str(int((c - offset) * scale)))
        roi_scaled_str = "poly:" + ",".join(adjusted)

        return roi_scaled_str, crop_rect

    def start_batch_processing(self):
        """Prepare video configs, launch the BatchProcessorThread, and lock the UI during processing."""
        global_config = self._collect_global_settings()
        scale = global_config["scale"]
        skip_preprocessing = global_config["skip_preprocessing"]

        # Compute scaled ROIs before handing off to the thread — spinbox values are not
        # safely readable from a secondary thread.
        for item in self.videos_data:
            if item.get("run_tracking", True) and item["roi_4pt"]:
                item["roi_4pt_scaled"], item["crop_rect"] = self._build_roi_for_item(
                    item, scale, skip_preprocessing
                )
            else:
                item["roi_4pt_scaled"] = item["roi_4pt"]
                item["crop_rect"] = None

        # Immutable snapshots for the post-batch JSON export, independent of any UI changes during the run.
        self._last_batch_config = global_config.copy()
        self._last_batch_videos = []
        for v in self.videos_data:
            self._last_batch_videos.append(dict(v))

        # Lock the UI to prevent concurrent modifications to the queue or parameters.
        self.btn_start.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_remove_item.setEnabled(False)
        self.btn_clear_all.setEnabled(False)
        self.progress_bar.setValue(0)

        self.start_timestamp = time.time()
        self.elapsed_seconds = 0
        self.current_status_text = "Status: Initializing pipeline..."
        self.chronometer_timer.start()

        # Connect signals before start() to avoid a race where the thread finishes before slots are wired.
        self.thread = BatchProcessorThread(self._last_batch_videos, global_config)
        self.thread.progress_signal.connect(self.update_progress_ui)
        self.thread.log_signal.connect(self.update_log_ui)
        self.thread.error_signal.connect(self.handle_processor_error)
        self.thread.finished_signal.connect(self.on_processing_finished)
        self.thread.live_view_signal.connect(self._on_live_view_requested)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_live_view_requested(self, video_path, csv_path, roi_str, scale_factor,
                                light_on_dark, multi_animal):
        """Run live tracking on the main thread (cv2.imshow/waitKey require the main thread)."""
        run_global_tracking(video_path, csv_path, roi_str, scale_factor,
                            lambda _: None, live_view=True, light_on_dark=light_on_dark,
                            multi_animal=multi_animal)
        self.thread.live_view_done()

    def update_progress_ui(self, value):
        """Update the progress bar. processEvents() intentionally removed to avoid UI flooding on Windows."""
        self.progress_bar.setValue(value)

    def update_log_ui(self, text):
        """Display the latest log message with elapsed time. processEvents() intentionally removed."""
        self.current_status_text = f"Status: {text.strip()}"
        mins, secs = divmod(self.elapsed_seconds, 60)
        self.lbl_status.setText(f"{self.current_status_text} (Elapsed Time: {mins:02d}:{secs:02d})")

    def _export_settings_json(self):
        """Write a timestamped JSON file with all batch settings and ROI coordinates.

        Enables experiment reproducibility by keeping a complete record of every run.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cfg = self._last_batch_config
        output_path = os.path.join(cfg["output_dir"], f"settings_{timestamp}.json")

        data = {
            "timestamp": timestamp,
            "global_settings": {
                "output_dir": cfg["output_dir"],
                "fps": cfg["fps"],
                "scale": cfg["scale"],
                "skip_preprocessing": cfg["skip_preprocessing"],
                "use_gpu": cfg["use_gpu"],
                "intrusion_calib_window": cfg["calib_window"],
                "intrusion_min_duration_sec": cfg["min_duration"],
            },
            "videos": []
        }

        for v in self._last_batch_videos:
            roi_4pt_val = v.get("roi_4pt", "")
            if roi_4pt_val:
                tracker_roi_export = roi_4pt_val
            else:
                tracker_roi_export = None

            roi_rect_val = v.get("roi_rect", "")
            if roi_rect_val:
                intrusion_roi_export = roi_rect_val
            else:
                intrusion_roi_export = None

            data["videos"].append({
                "filename": os.path.basename(v["path"]),
                "path": v["path"],
                "tracker_roi": tracker_roi_export,
                "intrusion_roi": intrusion_roi_export,
                "roi_mode": v.get("roi_mode", "intrusion"),
                "tracking_enabled": v.get("run_tracking", True),
                "intrusion_enabled": v.get("run_intrusion", False),
                "live_review": v.get("run_review", False),
                "light_on_dark": v.get("light_on_dark", False),
                "multi_animal": v.get("multi_animal", False),
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def on_processing_finished(self):
        """Unlock the UI and export settings JSON when the batch thread signals completion."""
        self.chronometer_timer.stop()
        mins, secs = divmod(self.elapsed_seconds, 60)
        self.lbl_status.setText(f"Status: Pipeline finished in {mins:02d}:{secs:02d}")

        self._export_settings_json()

        QMessageBox.information(self, "Finished", "All videos processed into the global folder!")
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.update_queue_buttons_state()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
