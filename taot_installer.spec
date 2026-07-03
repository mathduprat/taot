# Build with:
#   pyinstaller taot_installer.spec
#
# This spec must be run natively on the target platform (Windows to produce
# MouseTracker.exe, Linux/macOS to produce a native MouseTracker binary)

import sys
import glob
import os
import shutil

# Directory containing this spec file. PyInstaller injects the `SPEC` global
# when running `pyinstaller <file>.spec`; fall back to the current working
# directory if the spec is ever imported/evaluated outside that context.
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC)) if 'SPEC' in dir() else os.path.abspath(".")

# All third-party binaries we bundle (compiled C++ extensions, FFmpeg, ...)
# are staged under lib/, never at the project/spec root.
LIB_DIR = os.path.join(SPEC_DIR, "lib")

# Detect platform: Windows compiled extensions use .pyd, Linux/macOS use .so.
if sys.platform == "win32":
    cpp_ext_pattern = "*.pyd"
else:
    cpp_ext_pattern = "*.so"

# Collect the compiled C++ extensions (global_tracker_core, roi_intrusion_detector_core)
# from src/. These are built ahead of time by src/build_cpp.py.
cpp_binaries = []
for pattern in [f"src/global_tracker_core*{cpp_ext_pattern[1:]}", f"src/roi_intrusion_detector_core*{cpp_ext_pattern[1:]}"]:
    for f in glob.glob(pattern):
        cpp_binaries.append((f, "src"))

# --- Bundle the FFmpeg binary so the built app works without a system FFmpeg install. ---
#
# At runtime, src/video_preprocessor.py:check_ffmpeg() looks for a binary named
# `ffmpeg.exe` (Windows) or `ffmpeg` (Linux/macOS) at the root of the PyInstaller
# bundle (sys._MEIPASS), so the destination below is always "." (the bundle root).
#
# The *source* copy of that binary, however, must live in lib/ next to this spec
# file — not at the project/spec root — for both platforms:
#   - lib/ffmpeg.exe : Windows FFmpeg build
#       (download from https://www.gyan.dev/ffmpeg/builds/ - ffmpeg-release-essentials.zip)
#   - lib/ffmpeg     : Linux FFmpeg build (static binary recommended)
# Both locations can be overridden with the FFMPEG_WIN_EXE / FFMPEG_LINUX_BIN
# environment variables.
if sys.platform == "win32":
    win_ffmpeg = os.environ.get("FFMPEG_WIN_EXE", os.path.join(LIB_DIR, "ffmpeg.exe"))
    if not os.path.isfile(win_ffmpeg):
        # Fall back to a system FFmpeg on PATH if lib/ffmpeg.exe wasn't provided.
        win_ffmpeg = shutil.which("ffmpeg")
    if win_ffmpeg:
        cpp_binaries.append((win_ffmpeg, "."))
    else:
        raise SystemExit(
            "ERROR: no ffmpeg.exe found. Place a Windows FFmpeg build at "
            f"'{os.path.join(LIB_DIR, 'ffmpeg.exe')}' (or set FFMPEG_WIN_EXE), "
            "or install FFmpeg and add it to PATH before building."
        )
else:
    linux_ffmpeg = os.environ.get("FFMPEG_LINUX_BIN", os.path.join(LIB_DIR, "ffmpeg"))
    if os.path.isfile(linux_ffmpeg):
        cpp_binaries.append((linux_ffmpeg, "."))
    else:
        print(
            f"WARNING: Linux ffmpeg binary not found at '{linux_ffmpeg}'. "
            "The built app will require FFmpeg installed on the target machine. "
            "To bundle it, place a Linux FFmpeg build at lib/ffmpeg or set "
            "FFMPEG_LINUX_BIN=/path/to/ffmpeg."
        )

# --- Bundle the OpenCV FFmpeg VideoIO backend plugin and its dependency DLLs. ---
#
# PyInstaller cannot auto-detect these because OpenCV loads them dynamically at
# runtime. Without this plugin, cv2.CAP_FFMPEG fails to open video files and
# OpenCV silently falls back to the MSMF backend on Windows.
# With a conda environment, the FFmpeg dependency DLLs (avcodec, avformat, ...)
# live under Library\bin\ of the conda env and must be bundled separately.
if sys.platform == "win32":
    import cv2
    cv2_dir = os.path.dirname(cv2.__file__)
    for dll in glob.glob(os.path.join(cv2_dir, "opencv_videoio_ffmpeg*.dll")):
        cpp_binaries.append((dll, "cv2"))

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        ffmpeg_dep_patterns = [
            "avcodec-*.dll", "avformat-*.dll", "avutil-*.dll",
            "swscale-*.dll", "swresample-*.dll",
        ]
        for pattern in ffmpeg_dep_patterns:
            for dll in glob.glob(os.path.join(conda_prefix, "Library", "bin", pattern)):
                cpp_binaries.append((dll, "."))

a = Analysis(
    ['app.py'],
    pathex=['src', 'ui'],
    binaries=cpp_binaries,
    datas=[('ui', 'ui'), ('src', 'src')],
    hiddenimports=[
        'cv2',
        'numpy',
        'pandas',
        'PyQt6',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='taot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='docs/taot_logo.ico',
)
