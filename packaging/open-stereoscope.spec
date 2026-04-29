# Run from the repository root:
#   pyinstaller packaging/open-stereoscope.spec

from PyInstaller.utils.hooks import collect_submodules


block_cipher = None

hiddenimports = (
    collect_submodules("cv2")
    + collect_submodules("imageio")
    + collect_submodules("imageio_ffmpeg")
    + collect_submodules("PIL")
)

a = Analysis(
    ["scripts/run_open_stereoscope.py"],
    pathex=["src"],
    binaries=[],
    datas=[("src/open_stereoscope/assets/open-stereo.png", "open_stereoscope/assets")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="open-stereoscope",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/open_stereoscope/assets/open-stereo.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="open-stereoscope",
)
