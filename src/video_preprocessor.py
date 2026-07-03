"""
Video Preprocessor (FFmpeg)
---------------------------
Reduces FPS, crops, and/or downscales a video using FFmpeg with optional GPU acceleration.
"""

import os
import subprocess
import sys

DEFAULT_CODEC = "libx264"

# Suppress the console window that Windows spawns for each subprocess.
# On Linux/macOS creationflags=0 is a no-op.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def check_ffmpeg():
    is_windows = (sys.platform == "win32")
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        if is_windows:
            ffmpeg_binary_name = "ffmpeg.exe"
        else:
            ffmpeg_binary_name = "ffmpeg"
        ffmpeg_bin = os.path.join(bundle_dir, ffmpeg_binary_name)
        if os.path.exists(ffmpeg_bin):
            os.environ["PATH"] = bundle_dir + os.path.pathsep + os.environ["PATH"]

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, creationflags=_NO_WINDOW)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg is not installed or not found in PATH.")


def _build_cpu_filter_chain(target_fps, crop_rect, scale):
    """Build the CPU-only FFmpeg filter list: fps and crop (scale excluded — it is GPU-dependent).

    Shared by the main pipeline and the CPU fallback to avoid duplicating filter logic.
    Filter order is mandatory: fps → crop → (hwupload → scale handled by the caller).
    H.264/YUV420 require even dimensions; crop width/height are truncated down if needed.
    """
    filters = []

    if target_fps is not None:
        filters.append(f"fps=fps={target_fps}")

    if crop_rect is not None:
        cx, cy, cw, ch = crop_rect
        # H.264 and YUV420 require even dimensions; truncate by 1 pixel if needed.
        if cw % 2 != 0:
            cw = cw - 1
        if ch % 2 != 0:
            ch = ch - 1
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")

    return filters


def run_preprocessing(input_path, target_fps, scale, output_path, progress_callback, log_callback, crop_rect=None, use_gpu=False):
    check_ffmpeg()

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    is_windows = (sys.platform == "win32")

    hw_codec = None
    if use_gpu:
        if is_windows:
            hw_codec = "h264_nvenc"
            log_callback("[INFO] GPU enabled. Preparing NVIDIA 'h264_nvenc' pipeline.")
        elif sys.platform.startswith("linux") and os.path.exists("/dev/dri"):
            hw_codec = "h264_vaapi"
            log_callback("[INFO] GPU enabled. Linux VAAPI interface detected.")
    else:
        log_callback("[INFO] GPU disabled. Using CPU encoding (libx264).")

    # has_cpu_filters controls the NVENC pipeline variant.
    # With -hwaccel_output_format cuda, decoded frames stay in CUDA memory.
    # A CPU filter (fps, crop) cannot access CUDA surfaces directly and causes an FFmpeg error.
    # When CPU filters are present, we decode to RAM and use hwupload_cuda only before scale_cuda.
    has_cpu_filters = (target_fps is not None) or (crop_rect is not None)

    cmd = ["ffmpeg", "-y"]
    if hw_codec == "h264_nvenc":
        # Without CPU filters: hwaccel_output_format cuda keeps frames on GPU from decode to encode.
        # With CPU filters: decode to CPU RAM, apply fps/crop, then hwupload → scale_cuda → NVENC.
        if has_cpu_filters:
            cmd.extend(["-hwaccel", "cuda"])
        else:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    elif hw_codec == "h264_vaapi":
        cmd.extend(["-vaapi_device", "/dev/dri/renderD128"])

    cmd.extend(["-i", input_path])

    # CPU filters (fps + crop) come first; GPU scale is appended below based on the codec.
    video_filters = _build_cpu_filter_chain(target_fps, crop_rect, scale)

    if scale != 1.0:
        if hw_codec == "h264_nvenc":
            # hwupload_cuda must precede scale_cuda: transfers CPU frames to CUDA memory.
            video_filters.append(f"hwupload_cuda,scale_cuda=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2")
        elif hw_codec == "h264_vaapi":
            # NV12 conversion required by VAAPI encoder, then hwupload, then scale_vaapi.
            video_filters.append(f"format=nv12,hwupload,scale_vaapi=w=trunc(iw*{scale}/2)*2:h=trunc(ih*{scale}/2)*2")
        else:
            # CPU scale (libx264): trunc(iw*scale/2)*2 ensures even output dimensions.
            video_filters.append(f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2")
    elif hw_codec == "h264_nvenc" and not has_cpu_filters:
        # No scale but NVENC without CPU filters: frames are already in CUDA, no extra hwupload needed.
        video_filters.append("hwupload_cuda")

    if video_filters:
        cmd.extend(["-vf", ",".join(video_filters)])

    if hw_codec:
        active_codec = hw_codec
    else:
        active_codec = DEFAULT_CODEC
    cmd.extend(["-c:v", active_codec])

    if active_codec == "h264_nvenc":
        cmd.extend(["-preset", "p4", "-cq", "20"])
    elif active_codec == "libx264":
        cmd.extend(["-pix_fmt", "yuv420p", "-crf", "23", "-preset", "ultrafast"])

    cmd.extend(["-c:a", "copy", output_path])

    def _run_fallback():
        """CPU-only fallback triggered on any non-zero GPU pipeline return code.

        Rebuilds the filter chain using _build_cpu_filter_chain (shared logic) and
        encodes with libx264. No GPU dependency — works on all machines.
        """
        cmd_fallback = ["ffmpeg", "-y", "-i", input_path]
        fallback_filters = _build_cpu_filter_chain(target_fps, crop_rect, scale)
        if scale != 1.0:
            fallback_filters.append(f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2")
        if fallback_filters:
            cmd_fallback.extend(["-vf", ",".join(fallback_filters)])
        cmd_fallback.extend(["-c:v", DEFAULT_CODEC, "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "ultrafast", "-c:a", "copy", output_path])
        result = subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg CPU fallback failed:\n{result.stderr.decode(errors='replace')}")

    try:
        progress_callback(10.0)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        if result.returncode != 0:
            if hw_codec is not None:
                if result.stderr:
                    last_error_line = result.stderr.decode(errors='replace').splitlines()[-1]
                else:
                    last_error_line = 'unknown error'
                log_callback(f"[WARNING] Hardware pipeline failed: {last_error_line}. Falling back to CPU...")
                _run_fallback()
            else:
                raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode(errors='replace')}")
        progress_callback(100.0)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"FFmpeg unexpected error: {e}") from e
