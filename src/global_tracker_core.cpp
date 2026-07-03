#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>

namespace py = pybind11;

// Pre-allocated morphological structuring elements, shared across frames (allocating them
// per-frame would be prohibitively slow).
class FrameEngine {
private:
    cv::Mat k_clean;  // 3x3: remove pixel-level noise after thresholding
    cv::Mat k_merge;  // 7x7: dilate to merge nearby fragments of the same animal
    cv::Mat k_close;  // 9x9: fill internal holes in a blob

    // Fixed threshold for the dark-mouse / light-background case — a proven baseline for
    // clean (non-textured) backgrounds. Pixels below this value (dark coat) become white
    // via THRESH_BINARY_INV.
    static constexpr int MANUAL_DARK_THRESH = 72;

    // Minimum contour area, in working-resolution pixels, to be considered a mouse rather
    // than morphological noise (dust, reflections, bedding highlights).
    static constexpr double MIN_BLOB_AREA = 90.0;

    // Reusable per-frame buffers, avoid heap allocations on every call. Shared by both the
    // native batch loop (track_video_native) and the Python live-view path (process_frame_py),
    // since both funnel through process_frame().
    cv::Mat gray_, blurred_, thresh_, coat_mask_, solid_roi_blobs_;

    // ROI mask cache for process_frame_py(), which is called once per frame rather than once
    // before a loop like track_video_native — rebuilding the mask on every call would be
    // wasteful, so it's only rebuilt when the ROI string or frame size actually changes.
    std::string cached_roi_key_;
    cv::Mat cached_roi_mask_;

    // Per-frame detection output. Coordinates in `contours` / `centroids_low` are at the
    // resolution of the frame passed in (working resolution); `centroids_full` / `areas_full`
    // are remapped to original video resolution via flow_scale_inv, ready for CSV export.
    struct FrameBlobs {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Point>              centroids_low;
        std::vector<std::pair<int, int>>    centroids_full;
        std::vector<int>                    areas_full;
    };

    // Parses a shape-tagged ROI string, either "poly:x0,y0,x1,y1,x2,y2,x3,y3" or
    // "circle:cx,cy,r" (a missing "shape:" prefix defaults to "poly" for backward
    // compatibility), and rasterizes it into a filled binary mask at (H, W).
    static cv::Mat build_roi_mask(const std::string& roi_coords_str, int W, int H) {
        std::string roi_type = "poly";
        std::string coords_part = roi_coords_str;
        size_t colon_pos = roi_coords_str.find(':');
        if (colon_pos != std::string::npos) {
            roi_type = roi_coords_str.substr(0, colon_pos);
            coords_part = roi_coords_str.substr(colon_pos + 1);
        }

        std::vector<int> coords;
        std::stringstream ss(coords_part);
        std::string item;
        while (std::getline(ss, item, ',')) {
            coords.push_back(std::stoi(item));
        }

        cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
        if (roi_type == "circle") {
            cv::circle(mask, cv::Point(coords[0], coords[1]), coords[2], cv::Scalar(255), -1);
        } else {
            cv::Point roi_pts[4] = {
                cv::Point(coords[0], coords[1]), cv::Point(coords[2], coords[3]),
                cv::Point(coords[4], coords[5]), cv::Point(coords[6], coords[7])
            };
            const cv::Point* ppt[1] = { roi_pts };
            int npt[] = { 4 };
            cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255));
        }
        return mask;
    }

    // Runs the full coat-mask + blob-detection pipeline on a single frame. This is the one
    // and only implementation of the detection logic: the native batch loop and the Python
    // live-view preview both call it, so they can never drift out of sync with each other.
    //
    // light_on_dark : false (default) = dark mouse on a light background: fixed manual
    //                 threshold (MANUAL_DARK_THRESH), a proven baseline for clean backgrounds.
    //                 true = light mouse on a dark background: automatic Otsu threshold.
    // multi_animal  : true = keep every blob above MIN_BLOB_AREA (several animals may be
    //                 in the ROI at once). false = only one animal is expected, so only
    //                 the largest blob is kept and smaller ones (noise, reflections, a
    //                 second partial blob from the same animal) are discarded.
    // flow_scale_inv: 1 / scale_factor — maps working-resolution coordinates back to original.
    FrameBlobs process_frame(const cv::Mat& frame_bgr, const cv::Mat& roi_mask,
                              bool light_on_dark, bool multi_animal, double flow_scale_inv)
    {
        // Step 1: BGR → grayscale (intensity only, reduces memory and compute).
        cv::cvtColor(frame_bgr, gray_, cv::COLOR_BGR2GRAY);

        // Step 2: Gaussian blur 5x5 attenuates high-frequency noise while preserving blob shapes.
        cv::GaussianBlur(gray_, blurred_, cv::Size(5, 5), 0);

        // Step 3: coat mask.
        if (light_on_dark) {
            // Light mouse on a dark background: automatic Otsu threshold directly on the
            // blurred frame. THRESH_BINARY turns bright pixels (light coat) white.
            cv::threshold(blurred_, thresh_, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            // Dark mouse on a light background: fixed manual threshold, THRESH_BINARY_INV
            // turns pixels below MANUAL_DARK_THRESH (dark coat) white.
            cv::threshold(blurred_, thresh_, MANUAL_DARK_THRESH, 255, cv::THRESH_BINARY_INV);
        }

        // Passes morphology: OPEN removes noise, DILATE merges nearby fragments of the same
        // animal, CLOSE fills small internal holes/gaps left after merging.
        cv::morphologyEx(thresh_,    coat_mask_, cv::MORPH_OPEN,   k_clean);
        cv::morphologyEx(coat_mask_, coat_mask_, cv::MORPH_DILATE, k_merge);
        cv::morphologyEx(coat_mask_, coat_mask_, cv::MORPH_CLOSE,  k_close);

        // Step 4: intersect coat mask with the ROI — keeps only mouse-colored blobs inside the zone.
        cv::bitwise_and(coat_mask_, roi_mask, solid_roi_blobs_);

        // Step 5: find external contours (no hierarchy needed) and filter by area.
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(solid_roi_blobs_, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        FrameBlobs result;
        double best_area = -1.0;
        for (const auto& cnt : cnts) {
            double low_res_area = cv::contourArea(cnt);

            // Contours smaller than MIN_BLOB_AREA at working resolution are noise, not mice.
            if (low_res_area < MIN_BLOB_AREA) continue;

            cv::Moments M = cv::moments(cnt);
            if (M.m00 <= 0.001) continue;

            // Step 6: remap centroid and area to original resolution via flow_scale_inv.
            // Areas scale by flow_scale_inv² because area is a squared length unit.
            double cx = M.m10 / M.m00;
            double cy = M.m01 / M.m00;
            int cx_full      = static_cast<int>(cx * flow_scale_inv);
            int cy_full      = static_cast<int>(cy * flow_scale_inv);
            int full_res_area = static_cast<int>(low_res_area * (flow_scale_inv * flow_scale_inv));

            if (!multi_animal) {
                // Single-animal mode: keep only the largest blob seen so far.
                if (low_res_area <= best_area) continue;
                best_area = low_res_area;
                result.contours       = { cnt };
                result.centroids_low  = { cv::Point(static_cast<int>(cx), static_cast<int>(cy)) };
                result.centroids_full = { { cx_full, cy_full } };
                result.areas_full     = { full_res_area };
                continue;
            }

            result.contours.push_back(cnt);
            result.centroids_low.emplace_back(static_cast<int>(cx), static_cast<int>(cy));
            result.centroids_full.emplace_back(cx_full, cy_full);
            result.areas_full.push_back(full_res_area);
        }
        return result;
    }

public:
    FrameEngine() {
        k_clean = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        k_merge = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        k_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    }

    // Process a video frame-by-frame and export detected blobs (centroids + areas) to CSV.
    // Headless, no GIL, optimized for batch runs — see process_frame_py() for the interactive
    // frame-by-frame path used by the Python live-view preview.
    //
    // roi_coords_str  : shape-tagged ROI, see build_roi_mask().
    // light_on_dark, multi_animal : see process_frame().
    // flow_scale_inv  : 1 / scale_factor — maps working-resolution coordinates back to original
    void track_video_native(const std::string& video_path,
                            const std::string& output_csv,
                            const std::string& roi_coords_str,
                            bool light_on_dark,
                            bool multi_animal,
                            double flow_scale_inv)
    {
        cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
        if (!cap.isOpened())
            throw std::runtime_error("CAP_FFMPEG failed to open video (MSMF fallback prevented): " + video_path);

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 12.0; // fallback if metadata is missing

        int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::Mat roi_mask_low = build_roi_mask(roi_coords_str, W, H);

        cv::Mat frame;

        // 1 MiB userspace I/O buffer: drastically reduces syscall overhead vs. per-line flushing.
        std::vector<char> file_io_buf(1 << 20);
        std::ofstream csv_out;
        csv_out.rdbuf()->pubsetbuf(file_io_buf.data(), static_cast<std::streamsize>(file_io_buf.size()));
        csv_out.open(output_csv, std::ios::trunc);
        if (!csv_out.is_open())
            throw std::runtime_error("Failed to open output CSV: " + output_csv);
        csv_out << "frame_idx,timestamp_sec,detected_blobs_count,centroids_xy,blob_sizes_px\n";

        // Batch CSV writes every BATCH_SIZE frames to amortize disk I/O cost.
        const size_t BATCH_SIZE = 500;
        std::string csv_batch;
        csv_batch.reserve(BATCH_SIZE * 80); // ~80 chars per line estimated

        std::string centroids_str, sizes_str;
        centroids_str.reserve(256);
        sizes_str.reserve(128);

        int frame_count = 0;

        while (true) {
            if (!cap.read(frame)) break;

            int current_frame_idx = frame_count++;

            FrameBlobs blobs = process_frame(frame, roi_mask_low, light_on_dark, multi_animal, flow_scale_inv);

            centroids_str = "\"[";
            sizes_str     = "\"[";
            for (size_t i = 0; i < blobs.centroids_full.size(); ++i) {
                if (i != 0) { centroids_str += ", "; sizes_str += ", "; }
                char coord_buf[32];
                snprintf(coord_buf, sizeof(coord_buf), "(%d, %d)",
                         blobs.centroids_full[i].first, blobs.centroids_full[i].second);
                centroids_str += coord_buf;
                char size_buf[16];
                snprintf(size_buf, sizeof(size_buf), "%d", blobs.areas_full[i]);
                sizes_str += size_buf;
            }
            centroids_str += "]\"";
            sizes_str     += "]\"";

            double timestamp_sec = round((current_frame_idx / fps) * 1000.0) / 1000.0;

            char row_buf[64];
            int row_len = snprintf(row_buf, sizeof(row_buf), "%d,%.3f,%d,",
                                    current_frame_idx, timestamp_sec, (int)blobs.centroids_full.size());
            csv_batch.append(row_buf, row_len);
            csv_batch += centroids_str;
            csv_batch += ',';
            csv_batch += sizes_str;
            csv_batch += '\n';

            // Flush batch every BATCH_SIZE frames : limits memory growth while keeping I/O efficient.
            if (frame_count % BATCH_SIZE == 0) {
                csv_out << csv_batch;
                csv_batch.clear();
            }
        }

        cap.release();
        csv_out << csv_batch; // flush remaining lines (< BATCH_SIZE)
    }

    // Interactive per-frame entry point for the Python live-view preview (see
    // global_tracker.py::_run_live_tracking). Runs the exact same process_frame() pipeline as
    // track_video_native, so Python never re-implements the detection logic — it only draws
    // the returned contours/centroids and handles UI (trackbar, pause/step).
    //
    // frame_bgr_np : HxWx3 uint8 BGR frame, as read by cv2.VideoCapture.
    // Returns (contours, centroids_low, centroids_full, areas_full):
    //   contours       : list of contours, each a list of (x, y) points at working resolution
    //                     (same coordinate space as frame_bgr_np) — for overlay drawing.
    //   centroids_low  : list of (x, y) at working resolution — for overlay circle/marker.
    //   centroids_full : list of (x, y) remapped to original resolution — for CSV export.
    //   areas_full     : list of blob areas remapped to original resolution — for CSV export.
    py::tuple process_frame_py(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> frame_bgr_np,
                                const std::string& roi_coords_str,
                                bool light_on_dark,
                                bool multi_animal,
                                double flow_scale_inv)
    {
        py::buffer_info buf = frame_bgr_np.request();
        if (buf.ndim != 3 || buf.shape[2] != 3)
            throw std::runtime_error("process_frame_py expects an HxWx3 uint8 BGR frame");

        int H = static_cast<int>(buf.shape[0]);
        int W = static_cast<int>(buf.shape[1]);
        cv::Mat frame_bgr(H, W, CV_8UC3, buf.ptr);

        std::string roi_key = roi_coords_str + "|" + std::to_string(W) + "x" + std::to_string(H);
        if (roi_key != cached_roi_key_) {
            cached_roi_mask_ = build_roi_mask(roi_coords_str, W, H);
            cached_roi_key_  = roi_key;
        }

        FrameBlobs blobs = process_frame(frame_bgr, cached_roi_mask_, light_on_dark, multi_animal, flow_scale_inv);

        py::list contours_py;
        for (const auto& cnt : blobs.contours) {
            py::list pts;
            for (const auto& p : cnt) pts.append(py::make_tuple(p.x, p.y));
            contours_py.append(pts);
        }

        py::list centroids_low_py;
        for (const auto& c : blobs.centroids_low) centroids_low_py.append(py::make_tuple(c.x, c.y));

        return py::make_tuple(contours_py, centroids_low_py, blobs.centroids_full, blobs.areas_full);
    }
};

PYBIND11_MODULE(global_tracker_core, m) {
    py::class_<FrameEngine>(m, "FrameEngine")
        .def(py::init<>())
        // py::call_guard<py::gil_scoped_release>: releases the GIL for the entire duration of
        // track_video_native. This allows other Python threads (UI, callbacks) to run concurrently
        // during the long CPU-bound processing.
        // IMPORTANT: no Python objects (py::object, py::list, etc.) may be touched inside the
        // function body while the GIL is released. Only native C++ types are safe.
        .def("track_video_native", &FrameEngine::track_video_native,
             py::call_guard<py::gil_scoped_release>())
        .def("process_frame_py", &FrameEngine::process_frame_py);
}
