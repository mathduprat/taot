import os
import ast
import cv2
import numpy as np
import pandas as pd


# Shared signature: fn(hour_df, hour_sessions) -> scalar
#   hour_df       : tracking CSV rows for this hour bucket
#   hour_sessions : intrusion sessions whose start falls in this hour
#                   (empty DataFrame when no intrusion CSV is provided)
#
# Output columns (one value per hour):
#
#   mean_blob_area_px         — Average area of all detected blobs across all frames
#                               of the hour. Each frame may contain several blobs (one per
#                               animal or merged group) all individual sizes are pooled and
#                               averaged. A large value suggests merging events (two animals
#                               detected as one big blob) a small value suggests well-
#                               separated individuals.
#
#   explored_area_px²         — Area (px²) of the convex hull wrapping every centroid
#                               recorded during the hour (all animals, all frames). Reflects
#                               how much of the cage was collectively visited, regardless of
#                               movement speed.
#
#   mean_blobs_per_frame      — Average number of distinct blobs detected per frame.
#                               Drops below the real animal count when individuals merge.
#
#   immobility_total_sec      — Total time (seconds) during which all animals were
#                               considered immobile. A frame transition is immobile when the
#                               mean nearest-neighbour displacement of all centroids between
#                               two consecutive frames is below IMMOBILITY_THRESHOLD_PX.
#
#   biberon_access_count      — Number of intrusion events that started
#                               during the hour.
#
#   biberon_total_duration_sec — Cumulative duration (seconds) spent at the bib during
#                                the hour.

# col: mean_blob_area_px
# All blob sizes (px²) from every frame of the hour are collected and averaged.
def _mean_blob_area(hour_df, _sessions):
    all_sizes = []
    for raw in hour_df["blob_sizes_px"]:
        try:
            sizes = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if isinstance(sizes, (list, tuple)):
                all_sizes.extend(sizes)
        except (ValueError, SyntaxError):
            pass
    return round(float(np.mean(all_sizes)), 1) if all_sizes else 0.0


# col: explored_area_px
# Every centroid from every frame is accumulated, then a single convex hull is
# computed over all those points. Its area is the collective spatial footprint of
# all animals over the hour.
def _explored_area(hour_df, _sessions):
    points = []
    for raw in hour_df["centroids_xy"]:
        try:
            centroids = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if isinstance(centroids, (list, tuple)):
                points.extend(centroids)
        except (ValueError, SyntaxError):
            pass
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    return round(cv2.contourArea(hull), 1)


# col: mean_blobs_per_frame
# Simple mean of the detected_blobs_count column over the hour.
def _mean_blobs_per_frame(hour_df, _sessions):
    return round(hour_df["detected_blobs_count"].mean(), 2)


# col: immobility_total_sec
# For each consecutive frame pair, each current centroid is matched to its nearest
# centroid in the previous frame. If the mean of those minimum distances is below
# IMMOBILITY_THRESHOLD_PX, the transition is counted as immobile. The total number
# of immobile transitions is converted to seconds using the per-hour fps estimate.
IMMOBILITY_THRESHOLD_PX = 5

def _immobility_sec(hour_df, _sessions):
    df = hour_df.sort_values("frame_idx").reset_index(drop=True)

    valid = df[df["timestamp_sec"] > 0]
    if len(valid) < 2:
        return 0.0
    dt = valid.iloc[-1]["timestamp_sec"] - valid.iloc[0]["timestamp_sec"]
    di = valid.iloc[-1]["frame_idx"] - valid.iloc[0]["frame_idx"]
    fps = float(di) / float(dt) if dt > 0 else 6.0

    immobile_frames = 0
    prev_pts = None

    for raw in df["centroids_xy"]:
        try:
            centroids = ast.literal_eval(raw) if isinstance(raw, str) else raw
            curr_pts = np.array(centroids, dtype=np.float32) if centroids else None
        except (ValueError, SyntaxError):
            curr_pts = None

        if prev_pts is not None and curr_pts is not None and len(curr_pts) > 0:
            displacements = [np.linalg.norm(prev_pts - pt, axis=1).min() for pt in curr_pts]
            if np.mean(displacements) < IMMOBILITY_THRESHOLD_PX:
                immobile_frames += 1

        prev_pts = curr_pts

    return round(immobile_frames / fps, 2)


# col: biberon_access_count
def _biberon_count(_hour_df, hour_sessions):
    return len(hour_sessions)


# col: biberon_total_duration_sec
def _biberon_duration(_hour_df, hour_sessions):
    if hour_sessions.empty:
        return 0.0
    return round(float(hour_sessions["duration_seconds"].sum()), 2)


# Add new metrics here 
# Format: ("csv_column_name", function)
METRICS = [
    ("mean_blob_area_px",           _mean_blob_area),
    ("explored_area_px2",          _explored_area),
    ("mean_blobs_per_frame",       _mean_blobs_per_frame),
    ("immobility_total_sec",       _immobility_sec),
    ("biberon_access_count",       _biberon_count),
    ("biberon_total_duration_sec", _biberon_duration),
]


def _derive_fps(tracking_df):
    valid = tracking_df[tracking_df["timestamp_sec"] > 0]
    if valid.empty:
        return 12.0
    last = valid.iloc[-1]
    return float(last["frame_idx"]) / float(last["timestamp_sec"])


def compute_hourly_summary(tracking_csv, output_csv, sessions_csv=None):
    tracking_df = pd.read_csv(tracking_csv)
    tracking_df["hour"] = (tracking_df["timestamp_sec"] // 3600).astype(int)

    sessions_df = pd.DataFrame()
    if sessions_csv and os.path.exists(sessions_csv):
        raw = pd.read_csv(sessions_csv)
        if not raw.empty:
            fps = _derive_fps(tracking_df)
            raw = raw.copy()
            raw["hour"] = (raw["start_frame"] / fps // 3600).astype(int)
            sessions_df = raw

    rows = []
    for h in sorted(tracking_df["hour"].unique()):
        hour_df = tracking_df[tracking_df["hour"] == h]
        hour_sessions = sessions_df[sessions_df["hour"] == h] if not sessions_df.empty else pd.DataFrame()

        row = {
            "hour": h,
            "hour_label": f"{h:02d}:00-{h + 1:02d}:00",
        }
        for col_name, fn in METRICS:
            row[col_name] = fn(hour_df, hour_sessions)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
