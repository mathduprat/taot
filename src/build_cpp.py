#!/usr/bin/env python3
"""
Build script for native C++ modules (Pybind11 + OpenCV).
Discovers all .cpp files in src/ and compiles them as Python extensions.
Removes the temporary build/ directory after compilation.
"""

from setuptools import setup, Extension
import subprocess
import sys
import os
import shutil
import glob
import platform

print("[BUILD] Compiling C++ modules...")

current_os = platform.system()

# Get pybind11 include paths.
try:
    pybind11_includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"]
    ).decode().split()
except subprocess.CalledProcessError:
    sys.exit("[ERROR] Pybind11 not found. Make sure the virtual environment is activated.")

includes = []
for f in pybind11_includes:
    if f.startswith("-I"):
        includes.append(f[2:])

lib_dirs = []
libs     = []
cflags   = []

# Resolve OpenCV dependencies based on the current platform.
if current_os == "Windows":
    # On Windows, rely on the OPENCV_DIR environment variable (e.g. C:\opencv\build).
    opencv_dir = os.environ.get("OPENCV_DIR")
    if not opencv_dir:
        sys.exit("[ERROR] OPENCV_DIR environment variable is not set.\n"
                 "        Point it to your OpenCV directory (e.g. C:\\opencv\\build).")

    includes.append(os.path.join(opencv_dir, "include"))

    # Conda places .lib files directly in lib/; the native Windows installer uses x64/vc16/lib/.
    windows_lib_dir = os.path.join(opencv_dir, "lib")

    if not os.path.exists(windows_lib_dir) or not glob.glob(os.path.join(windows_lib_dir, "opencv_*.lib")):
        windows_lib_dir = os.path.join(opencv_dir, "x64", "vc16", "lib")

    if not os.path.exists(windows_lib_dir):
        windows_lib_dir = os.path.join(opencv_dir, "x64", "vc15", "lib")

    if os.path.exists(windows_lib_dir):
        lib_dirs.append(windows_lib_dir)
        found_libs = glob.glob(os.path.join(windows_lib_dir, "opencv_*.lib"))
        # Exclude debug libraries (suffix "d.lib") to link the release build only.
        release_libs = []
        for l in found_libs:
            basename = os.path.basename(l)
            if not basename.endswith("d.lib"):
                release_libs.append(basename[:-4])
        libs.extend(release_libs)
    else:
        sys.exit(f"[ERROR] OpenCV directory not found at: {opencv_dir}")

    extra_compile_args = ["/O2", "/W3", "/std:c++17"]
    extra_link_args = []

else:
    # Linux / macOS
    try:
        opencv_flags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "opencv4"]
        ).decode().split()
    except subprocess.CalledProcessError:
        sys.exit("[ERROR] OpenCV4 not found via pkg-config. Install the development packages.")

    for f in opencv_flags:
        if f.startswith("-L"):
            lib_dirs.append(f[2:])
    for f in opencv_flags:
        if f.startswith("-l"):
            libs.append(f[2:])
    for f in opencv_flags:
        if not f.startswith(("-L", "-l", "-I")):
            cflags.append(f)
    includes.append("/usr/include/opencv4")

    extra_compile_args = ["-O3", "-Wall", "-std=c++17", "-fPIC"] + cflags
    extra_link_args = []

# Discover source files.
src_dir = "src"
cpp_files = glob.glob(os.path.join(src_dir, "*.cpp"))

if not cpp_files:
    sys.exit(f"[ERROR] No .cpp files found in: {src_dir}")

print(f"[BUILD] {len(cpp_files)} C++ file(s) found.")

extensions = []
for source_path in cpp_files:
    file_name = os.path.basename(source_path)
    module_name, _ = os.path.splitext(file_name)

    print(f"  -> Module: '{module_name}' (source: {source_path})")

    ext = Extension(
        f"src.{module_name}",
        sources=[source_path],
        include_dirs=includes,
        library_dirs=lib_dirs,
        libraries=libs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
    extensions.append(ext)

setup(
    name="cpp_native_modules",
    ext_modules=extensions,
    script_args=["build_ext", "--inplace"]
)

# Remove temporary build directory.
temp_build_dir = "build"
if os.path.exists(temp_build_dir):
    print(f"[CLEANUP] Removing temporary directory: '{temp_build_dir}/'")
    shutil.rmtree(temp_build_dir)

print("[BUILD] Done.")
