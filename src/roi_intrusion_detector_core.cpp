#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// Detects presence inside a rectangular ROI using a two-scale adaptive bandpass.
//
// A dark object entering the ROI (a mouse snout, a hand, ...) makes the mean pixel
// intensity drop. The detection compares the current mean against the adaptive
// background (P95):
//
//   INTRUSION mode:  P95 - GRAY_DROP_MAX  <  current_mean  <  P95 - GRAY_DROP_MIN
//   LIGHT mode:                              current_mean  <  P95 - GRAY_DROP_MIN
//
//   - Below GRAY_DROP_MIN  => gray hasn't dropped enough: no presence.
//   - Above GRAY_DROP_MAX  (INTRUSION mode only) => gray dropped too much: object is
//     fully covering the ROI (e.g. passing over the biberon), not a real entry.
//   LIGHT mode drops this upper bound: any large enough gray drop counts, including
//   full ROI coverage, at the cost of a few more false positives.
//
// Long-term background (P95 of the last N frames):
//   Robust even when the object is present up to 5% of the calibration window.
//
// Short-term baseline (mean of the last 30 frames):
//   Used ONLY at session entry to require an actual gray transition, not a slow drift.
//
// Entry:   in_gray_band  &&  recent_mean = current_mean > ENTRY_DROP
// Sustain: in_gray_band  (short-term not re-checked once session is running)
class RoiAnalyzer {
private:
    cv::Rect analyze_roi;
    bool is_roi_set = false;

    // 0 = INTRUSION (bandpass, rejects full ROI coverage), 1 = LIGHT (single-sided,
    // accepts any large enough drop including full coverage). Set via set_roi().
    int detection_mode = 0;

    // Long-term sliding window for background estimation.
    std::deque<float> gray_history;
    int calib_window = 3000;

    // Short-term sliding window, recent baseline for entry transition detection.
    // Must be shorter than a typical intrusion (a few seconds at 6-12 fps = 30 frames).
    std::deque<float> short_history;
    const int SHORT_WINDOW = 30;

    // BG_PERCENTILE: P95 so the background estimate stays robust even when the animal
    // is present up to 5% of the calibration window.
    //
    // GRAY_DROP_MIN: minimum drop below P95 to confirm any presence in the ROI.
    // GRAY_DROP_MAX: maximum drop below P95 still accepted as "museau entry".
    //   Above this threshold the animal is fully covering the ROI (passing over
    //   the biberon) this is NOT a drinking event and must be excluded.
    //
    // ENTRY_DROP: minimum drop from the short-term recent mean to confirm a genuine
    //   entry transition vs. a slow background drift.
    const int   BG_PERCENTILE  = 95;
    const float GRAY_DROP_MIN  = 15.0f;
    const float GRAY_DROP_MAX  = 55.0f;
    const float ENTRY_DROP     = 20.0f;

    // Minimum frames required before enabling detection.
    const int MIN_HISTORY_SIZE = 30;

    // presence_start_time uses -1.0 as a sentinel for "no active session".
    double presence_start_time       = -1.0;
    double current_presence_duration = 0.0;
    bool   instant_presence          = false;

    // Timestamp of the last frame where the mouse was detected. -1.0 means never.
    double last_time_inside = -1.0;

    // Debounce window: a brief absence shorter than DELAY_DURATION_SEC does not close the session.
    // Absorbs 1-2 frame gaps and accidental fast exits.
    const double DELAY_DURATION_SEC = 0.4;

public:
    RoiAnalyzer() = default;

    // Sets the ROI geometry, history window size and detection mode, then resets
    // all tracking state. mode: 0 = intrusion (bandpass), 1 = light (no upper bound).
    void set_roi(int x, int y, int width, int height, int window_size, int mode = 0) {
        analyze_roi     = cv::Rect(x, y, width, height);
        calib_window    = window_size;
        detection_mode  = mode;
        is_roi_set      = true;
        reset_tracking();
    }

    // Analyzes one frame and returns (instant_presence, mean_gray, presence_duration).
    // instant_presence is true if the mouse is in the ROI or was there less than DELAY_DURATION_SEC ago.
    py::tuple analyze_frame(py::array_t<uint8_t> input_frame,
                            double timestamp_sec,
                            double min_duration_sec)
    {
        // Zero-copy access to the NumPy buffer: cv::Mat points directly into Python's memory.
        py::buffer_info buf = input_frame.request();
        cv::Mat color_full(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);

        if (!is_roi_set) {
            analyze_roi = cv::Rect(0, 0, color_full.cols, color_full.rows);
            is_roi_set  = true;
        }

        // Clamp ROI to frame bounds to avoid out-of-bounds access if the video was resized after set_roi.
        cv::Rect safe_roi = analyze_roi & cv::Rect(0, 0, color_full.cols, color_full.rows);
        cv::Mat roi_sub = color_full(safe_roi);

        cv::Scalar mean_channels = cv::mean(roi_sub);

        // Unweighted mean (B + G + R) / 3, consistent with ImageJ Analyze > Measure.
        float current_mean = (float)(mean_channels[0] + mean_channels[1] + mean_channels[2]) / 3.0f;

        // Update both sliding windows before computing the threshold.
        gray_history.push_back(current_mean);
        if ((int)gray_history.size() > calib_window)
            gray_history.pop_front();

        short_history.push_back(current_mean);
        if ((int)short_history.size() > SHORT_WINDOW)
            short_history.pop_front();

        bool inside_interval = false;
        if ((int)gray_history.size() >= MIN_HISTORY_SIZE) {
            // Long-term P95 background estimate (O(N) via nth_element).
            std::vector<float> buf_copy(gray_history.begin(), gray_history.end());
            int p_idx = (int)((BG_PERCENTILE / 100.0f) * (float)(buf_copy.size() - 1));
            std::nth_element(buf_copy.begin(), buf_copy.begin() + p_idx, buf_copy.end());
            float bg_estimate = buf_copy[p_idx];

            // Short-term mean : representative of the ROI state in the last ~30 frames.
            float recent_mean = 0.0f;
            for (float v : short_history) recent_mean += v;
            recent_mean /= (float)short_history.size();

            // Bandpass: gray must have dropped enough to confirm presence (MIN).
            // INTRUSION mode also rejects drops beyond MAX (object fully covering
            // the ROI). LIGHT mode has no upper bound: any large drop counts.
            bool in_gray_band = (current_mean < bg_estimate - GRAY_DROP_MIN) &&
                                (detection_mode != 0 ||
                                 current_mean > bg_estimate - GRAY_DROP_MAX);

            bool session_running = (presence_start_time >= 0.0);

            if (!session_running) {
                // Entry criterion: bandpass AND a visible drop from the recent baseline.
                // Rejects slow drifts and full-body coverage events.
                inside_interval = in_gray_band &&
                                  (recent_mean - current_mean > ENTRY_DROP);
            } else {
                // Sustain criterion: bandpass only.
                // The short-term mean is already dark while the animal is inside,
                // so re-applying ENTRY_DROP would break ongoing sessions.
                inside_interval = in_gray_band;
            }
        }

        // Session management.
        if (inside_interval) {
            instant_presence = true;
            last_time_inside = timestamp_sec;

            // Start a new session or extend the current one.
            if (presence_start_time < 0.0) {
                presence_start_time       = timestamp_sec;
                current_presence_duration = 0.0;
            } else {
                current_presence_duration = timestamp_sec - presence_start_time;
            }
        } else {
            if (presence_start_time >= 0.0 &&
                (timestamp_sec - last_time_inside) <= DELAY_DURATION_SEC) {
                // Debounce: absence too short to close the session.
                instant_presence          = true;
                current_presence_duration = timestamp_sec - presence_start_time;
            } else {
                // Confirmed absence: close session and reset sentinels.
                presence_start_time       = -1.0;
                current_presence_duration = 0.0;
                instant_presence          = false;
                last_time_inside          = -1.0;
            }
        }

        return py::make_tuple(instant_presence, current_mean, current_presence_duration);
    }

    // Resets tracking state and the gray history buffer.
    // Call between videos or after changing the ROI.
    void reset_tracking() {
        presence_start_time       = -1.0;
        current_presence_duration = 0.0;
        instant_presence          = false;
        last_time_inside          = -1.0;
        gray_history.clear();
        short_history.clear();
    }
};

PYBIND11_MODULE(roi_intrusion_detector_core, m) {
    py::class_<RoiAnalyzer>(m, "RoiAnalyzer")
        .def(py::init<>())
        .def("set_roi",       &RoiAnalyzer::set_roi,
             py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"),
             py::arg("window_size"), py::arg("mode") = 0)
        .def("analyze_frame", &RoiAnalyzer::analyze_frame)
        .def("reset_history", &RoiAnalyzer::reset_tracking);
}
