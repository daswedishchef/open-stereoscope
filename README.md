# open-stereoscope

Open Stereoscope is a Windows desktop app for creating geological wiggle GIFs
from two overlapping stereo images.

The app lets a user:

- choose two image files;
- automatically register the second image to the first with OpenCV;
- choose ORB or SIFT feature registration;
- ignore black scan borders while registering and cropping;
- crop both images to the detected overlap;
- preview the two cropped frames and the animated wiggle or smooth transition;
- choose the animation frame speed;
- toggle between two-frame wiggle animation and smooth interpolation;
- adjust brightness and contrast independently for each cropped frame;
- auto-adjust the second cropped frame to match the first frame's brightness
  and contrast;
- export a looping GIF, with optional MP4 export.

## Tech stack

- Python 3.11+
- PySide6 for the desktop GUI
- OpenCV for registration, warping, cropping, and frame generation
- Pillow for GIF export with millisecond frame timing
- imageio for MP4 export
- PyInstaller for Windows packaging

## Run from source

Install Python 3.11 or newer, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m open_stereoscope
```

If you prefer not to install the package in editable mode, install the
requirements and set `PYTHONPATH` first:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python -m open_stereoscope
```

## Build a Windows executable

```powershell
.\scripts\build_windows.ps1
```

The executable is written to:

```text
dist\open-stereoscope\open-stereoscope.exe
```

## Registration approach

Open Stereoscope treats the first image as the reference. It detects ORB or
SIFT features in both images, matches them, estimates an affine transform with
RANSAC, warps the second image into the first image's coordinate space, and
crops both frames to the shared valid overlap. If feature matching is not
available, it falls back to phase-correlation translation. Black regions
connected to the image border are masked out before feature detection and are
excluded from the final overlap crop. Before registration, the second image is
temporarily brightness/contrast normalized toward the first image using
non-border content. After overlap detection, those same brightness and
contrast values are applied through the second image sliders, so the preview
and export start from the correction used for registration.

After registration, the brightness and contrast sliders adjust each cropped
frame independently. This helps when stereo images were captured with slightly
different exposure, lighting, or scan contrast. The Auto Adjust button uses
the cropped overlap to set the second frame's brightness and contrast toward
the first frame.

Smooth mode uses feature-based dense optical-flow interpolation between the
registered crops. It generates intermediate frames in both directions so the
GIF or MP4 moves smoothly from one perspective to the other and back.
