from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .processing import (
    ImageAdjustments,
    RegistrationError,
    RegistrationResult,
    apply_adjustments,
    build_animation_frames,
    estimate_adjustments_to_match,
    export_gif,
    export_mp4,
    load_image,
    register_pair,
)


IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All files (*.*)"
APP_ICON_RESOURCE = "assets/open-stereo.png"


def app_icon() -> QIcon:
    return QIcon(str(files("open_stereoscope").joinpath(APP_ICON_RESOURCE)))


class ImagePreview(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.setObjectName("imagePreview")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(240, 190)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("previewTitle")

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(220, 150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None:
            self.image_label.setText("No image")
            self.image_label.setPixmap(QPixmap())
            return

        available = self.image_label.size()
        scaled = self._pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("open-stereoscope")
        self.setWindowIcon(app_icon())
        self.resize(1180, 760)

        self.fixed_path: Path | None = None
        self.moving_path: Path | None = None
        self.result: RegistrationResult | None = None
        self.wiggle_frames: list[QPixmap] = []
        self.wiggle_index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_wiggle)
        self.preview_refresh_timer = QTimer(self)
        self.preview_refresh_timer.setSingleShot(True)
        self.preview_refresh_timer.timeout.connect(self._refresh_result_previews)
        self.adjustment_drag_depth = 0

        self._build_actions()
        self._build_ui()
        self._apply_style()
        self._sync_export_state()

    def _build_actions(self) -> None:
        open_fixed = QAction("Open first image", self)
        open_fixed.triggered.connect(self._choose_fixed)
        open_moving = QAction("Open second image", self)
        open_moving.triggered.connect(self._choose_moving)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.quit)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_fixed)
        file_menu.addAction(open_moving)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 14, 16, 14)
        root_layout.setSpacing(12)

        controls = QGroupBox("Image pair")
        controls_layout = QGridLayout(controls)
        controls_layout.setColumnStretch(1, 1)

        self.fixed_label = self._make_file_field("No first image selected")
        self.moving_label = self._make_file_field("No second image selected")
        self.fixed_button = QPushButton("Choose First Image")
        self.moving_button = QPushButton("Choose Second Image")
        self.process_button = QPushButton("Find Overlap")
        self.auto_adjust_button = QPushButton("Auto Adjust")
        self.export_gif_button = QPushButton("Export GIF")
        self.export_mp4_button = QPushButton("Export MP4")

        self.fixed_button.clicked.connect(self._choose_fixed)
        self.moving_button.clicked.connect(self._choose_moving)
        self.process_button.clicked.connect(self._process_pair)
        self.auto_adjust_button.clicked.connect(self._auto_adjust)
        self.export_gif_button.clicked.connect(self._export_gif)
        self.export_mp4_button.clicked.connect(self._export_mp4)

        detector_box = QWidget()
        detector_layout = QHBoxLayout(detector_box)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.setSpacing(8)
        self.orb_button = QPushButton("ORB")
        self.sift_button = QPushButton("SIFT")
        for button in (self.orb_button, self.sift_button):
            button.setCheckable(True)
            button.clicked.connect(self._registration_method_changed)
            detector_layout.addWidget(button)
        self.orb_button.setChecked(True)
        detector_layout.addStretch(1)

        animation_box = QWidget()
        animation_layout = QHBoxLayout(animation_box)
        animation_layout.setContentsMargins(0, 0, 0, 0)
        animation_layout.setSpacing(8)
        self.wiggle_button = QPushButton("Wiggle")
        self.smooth_button = QPushButton("Smooth")
        for button in (self.wiggle_button, self.smooth_button):
            button.setCheckable(True)
            button.clicked.connect(self._animation_mode_changed)
            animation_layout.addWidget(button)
        self.wiggle_button.setChecked(True)
        animation_layout.addStretch(1)

        speed_box = QWidget()
        speed_layout = QHBoxLayout(speed_box)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(8)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(80, 1000)
        self.speed_slider.setSingleStep(20)
        self.speed_slider.setPageStep(100)
        self.speed_slider.setValue(260)
        self.speed_value_label = self._make_value_label(self._format_ms(260))
        self.speed_slider.valueChanged.connect(
            lambda value: self.speed_value_label.setText(self._format_ms(value))
        )
        self.speed_slider.valueChanged.connect(self._speed_changed)
        speed_layout.addWidget(self.speed_slider, 1)
        speed_layout.addWidget(self.speed_value_label)

        self.fixed_brightness_slider, fixed_brightness_box = (
            self._make_adjustment_control(-100, 100, 0, "", self._adjustments_changed)
        )
        self.fixed_contrast_slider, fixed_contrast_box = (
            self._make_adjustment_control(0, 200, 100, "%", self._adjustments_changed)
        )
        self.moving_brightness_slider, moving_brightness_box = (
            self._make_adjustment_control(-100, 100, 0, "", self._adjustments_changed)
        )
        self.moving_contrast_slider, moving_contrast_box = (
            self._make_adjustment_control(0, 200, 100, "%", self._adjustments_changed)
        )

        controls_layout.addWidget(self.fixed_button, 0, 0)
        controls_layout.addWidget(self.fixed_label, 0, 1)
        controls_layout.addWidget(self.moving_button, 1, 0)
        controls_layout.addWidget(self.moving_label, 1, 1)
        controls_layout.addWidget(QLabel("Registration"), 2, 0)
        controls_layout.addWidget(detector_box, 2, 1)
        controls_layout.addWidget(QLabel("Animation"), 3, 0)
        controls_layout.addWidget(animation_box, 3, 1)
        controls_layout.addWidget(QLabel("Frame speed"), 4, 0)
        controls_layout.addWidget(speed_box, 4, 1)
        controls_layout.addWidget(QLabel("First brightness"), 5, 0)
        controls_layout.addWidget(fixed_brightness_box, 5, 1)
        controls_layout.addWidget(QLabel("First contrast"), 6, 0)
        controls_layout.addWidget(fixed_contrast_box, 6, 1)
        controls_layout.addWidget(QLabel("Second brightness"), 7, 0)
        controls_layout.addWidget(moving_brightness_box, 7, 1)
        controls_layout.addWidget(QLabel("Second contrast"), 8, 0)
        controls_layout.addWidget(moving_contrast_box, 8, 1)
        controls_layout.addWidget(self.process_button, 0, 2)
        controls_layout.addWidget(self.export_gif_button, 1, 2)
        controls_layout.addWidget(self.export_mp4_button, 2, 2)
        controls_layout.addWidget(self.auto_adjust_button, 3, 2)

        self.first_preview = ImagePreview("Cropped first image")
        self.second_preview = ImagePreview("Cropped registered second image")
        self.wiggle_preview = ImagePreview("Wiggle preview")

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(12)
        preview_layout.addWidget(self.first_preview, 1)
        preview_layout.addWidget(self.second_preview, 1)
        preview_layout.addWidget(self.wiggle_preview, 1)

        self.details_label = QLabel("Select two images to begin.")
        self.details_label.setObjectName("detailsLabel")

        root_layout.addWidget(controls)
        root_layout.addLayout(preview_layout, 1)
        root_layout.addWidget(self.details_label)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 10pt;
                color: #f5f5f5;
                background: #202124;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #4b5563;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel {
                color: #f5f5f5;
                background: transparent;
                border: 0;
            }
            QPushButton {
                min-height: 30px;
                padding: 4px 10px;
                color: #ffffff;
                background: #6b7280;
                border: 1px solid #9ca3af;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #7b8493;
            }
            QPushButton:pressed {
                background: #596273;
            }
            QPushButton:checked {
                color: #ffffff;
                background: #2563eb;
                border-color: #93c5fd;
            }
            QPushButton:disabled {
                color: #d1d5db;
                background: #4b5563;
                border-color: #6b7280;
            }
            QLineEdit {
                min-height: 28px;
                color: #ffffff;
                background: #2b2d31;
                border: 1px solid #6b7280;
                border-radius: 4px;
                padding: 2px 6px;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QFrame#imagePreview {
                background: #2b2d31;
                border: 1px solid #4b5563;
                border-radius: 6px;
            }
            QLabel#valueLabel {
                min-width: 58px;
                color: #ffffff;
                background: #2b2d31;
                border: 1px solid #6b7280;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QLabel#previewTitle {
                font-weight: 600;
                border: 0;
                background: transparent;
            }
            QLabel#detailsLabel {
                color: #f5f5f5;
            }
            QMenuBar, QStatusBar {
                color: #f5f5f5;
                background: #202124;
            }
            QMenuBar::item:selected, QMenu {
                color: #f5f5f5;
                background: #2b2d31;
            }
            QMenu::item:selected {
                background: #4b5563;
            }
            """
        )

    def _make_file_field(self, text: str) -> QLineEdit:
        field = QLineEdit(text)
        field.setReadOnly(True)
        field.setCursorPosition(0)
        return field

    def _make_value_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("valueLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _make_adjustment_control(
        self,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str,
        callback,
    ) -> tuple[QSlider, QWidget]:
        control = QWidget()
        layout = QHBoxLayout(control)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(value)

        value_label = self._make_value_label(self._format_value(value, suffix))
        slider.valueChanged.connect(
            lambda current: value_label.setText(self._format_value(current, suffix))
        )
        slider.valueChanged.connect(callback)
        slider.sliderPressed.connect(self._begin_adjustment_interaction)
        slider.sliderReleased.connect(self._end_adjustment_interaction)

        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return slider, control

    def _format_value(self, value: int, suffix: str) -> str:
        return f"{value}{suffix}" if suffix else str(value)

    def _format_ms(self, value: int) -> str:
        return f"{value} ms"

    def _choose_fixed(self) -> None:
        path = self._choose_image_file("Choose first image")
        if path is None:
            return
        self.fixed_path = path
        self.fixed_label.setText(str(path))
        self.fixed_label.setCursorPosition(0)
        self._clear_result()

    def _choose_moving(self) -> None:
        path = self._choose_image_file("Choose second image")
        if path is None:
            return
        self.moving_path = path
        self.moving_label.setText(str(path))
        self.moving_label.setCursorPosition(0)
        self._clear_result()

    def _choose_image_file(self, title: str) -> Path | None:
        selected, _ = QFileDialog.getOpenFileName(self, title, "", IMAGE_FILTER)
        if not selected:
            return None
        return Path(selected)

    def _process_pair(self) -> None:
        if self.fixed_path is None or self.moving_path is None:
            QMessageBox.warning(self, "Missing images", "Choose both source images first.")
            return

        self.statusBar().showMessage("Registering images...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fixed = load_image(self.fixed_path)
            moving = load_image(self.moving_path)
            self.result = register_pair(fixed, moving, self._registration_method())
            self._show_result()
        except RegistrationError as exc:
            self._clear_result()
            QMessageBox.critical(self, "Registration failed", str(exc))
            self.statusBar().showMessage("Registration failed", 5000)
        except Exception as exc:  # pragma: no cover - GUI boundary
            self._clear_result()
            QMessageBox.critical(self, "Unexpected error", str(exc))
            self.statusBar().showMessage("Unexpected error", 5000)
        finally:
            QApplication.restoreOverrideCursor()

    def _show_result(self) -> None:
        if self.result is None:
            return

        self.wiggle_index = 0
        self.fixed_brightness_slider.setValue(0)
        self.fixed_contrast_slider.setValue(100)
        self.moving_brightness_slider.setValue(
            self.result.registration_adjustments.brightness
        )
        self.moving_contrast_slider.setValue(
            int(round(self.result.registration_adjustments.contrast * 100))
        )
        self._refresh_result_previews()
        self.timer.start(self._wiggle_delay_ms())

        x, y, width, height = self.result.overlap_box
        self.details_label.setText(
            f"Overlap: {width} x {height}px at ({x}, {y}) | "
            f"Method: {self.result.method} | "
            f"Matches: {self.result.match_count} | "
            f"Confidence: {self._confidence_label(self.result.confidence)}"
        )
        self.statusBar().showMessage("Overlap found", 5000)
        self._sync_export_state()

    def _clear_result(self) -> None:
        self.result = None
        self.timer.stop()
        self.preview_refresh_timer.stop()
        self.adjustment_drag_depth = 0
        self.wiggle_frames = []
        self.first_preview.set_pixmap(None)
        self.second_preview.set_pixmap(None)
        self.wiggle_preview.set_pixmap(None)
        self.details_label.setText("Find the overlap to preview and export a wiggle.")
        self._sync_export_state()

    def _sync_export_state(self) -> None:
        has_result = self.result is not None
        self.auto_adjust_button.setEnabled(has_result)
        self.export_gif_button.setEnabled(has_result)
        self.export_mp4_button.setEnabled(has_result)

    def _speed_changed(self, value: int) -> None:
        if self.timer.isActive():
            self.timer.setInterval(value)

    def _wiggle_delay_ms(self) -> int:
        return self.speed_slider.value()

    def _registration_method(self) -> str:
        return "SIFT" if self.sift_button.isChecked() else "ORB"

    def _registration_method_changed(self, _checked: bool = False) -> None:
        sender = self.sender()
        if sender is self.sift_button:
            self.orb_button.setChecked(False)
            self.sift_button.setChecked(True)
        else:
            self.orb_button.setChecked(True)
            self.sift_button.setChecked(False)
        self._clear_result()

    def _animation_mode(self) -> str:
        return "smooth" if self.smooth_button.isChecked() else "wiggle"

    def _animation_mode_changed(self, _checked: bool = False) -> None:
        sender = self.sender()
        if sender is self.smooth_button:
            self.wiggle_button.setChecked(False)
            self.smooth_button.setChecked(True)
            self.wiggle_preview.title_label.setText("Smooth preview")
        else:
            self.wiggle_button.setChecked(True)
            self.smooth_button.setChecked(False)
            self.wiggle_preview.title_label.setText("Wiggle preview")

        if self.result is not None:
            self._refresh_result_previews()

    def _confidence_label(self, confidence: float) -> str:
        if confidence >= 0.65:
            return "high"
        if confidence >= 0.35:
            return "medium"
        return "low"

    def _adjustments_changed(self, _value: int) -> None:
        if self.result is None:
            return
        self._show_adjusted_still_previews()
        if self.adjustment_drag_depth > 0:
            return
        self._schedule_preview_refresh()

    def _begin_adjustment_interaction(self) -> None:
        self.adjustment_drag_depth += 1
        self.preview_refresh_timer.stop()
        if self.timer.isActive():
            self.timer.stop()
        self.statusBar().showMessage("Preview paused while adjusting", 2000)

    def _end_adjustment_interaction(self) -> None:
        self.adjustment_drag_depth = max(0, self.adjustment_drag_depth - 1)
        if self.result is not None and self.adjustment_drag_depth == 0:
            self._schedule_preview_refresh(delay_ms=50)

    def _schedule_preview_refresh(self, delay_ms: int = 250) -> None:
        self.preview_refresh_timer.start(delay_ms)

    def _auto_adjust(self) -> None:
        if self.result is None:
            return

        adjustments = estimate_adjustments_to_match(
            self.result.fixed_crop,
            self.result.moving_crop,
            min_brightness=self.moving_brightness_slider.minimum(),
            max_brightness=self.moving_brightness_slider.maximum(),
            min_contrast=self.moving_contrast_slider.minimum() / 100.0,
            max_contrast=self.moving_contrast_slider.maximum() / 100.0,
        )

        self.fixed_brightness_slider.setValue(0)
        self.fixed_contrast_slider.setValue(100)
        self.moving_brightness_slider.setValue(adjustments.brightness)
        self.moving_contrast_slider.setValue(int(round(adjustments.contrast * 100)))
        self._refresh_result_previews()
        self.statusBar().showMessage("Auto-adjusted second image to match first", 5000)

    def _fixed_adjustments(self) -> ImageAdjustments:
        return ImageAdjustments(
            brightness=self.fixed_brightness_slider.value(),
            contrast=self.fixed_contrast_slider.value() / 100.0,
        )

    def _moving_adjustments(self) -> ImageAdjustments:
        return ImageAdjustments(
            brightness=self.moving_brightness_slider.value(),
            contrast=self.moving_contrast_slider.value() / 100.0,
        )

    def _refresh_result_previews(self) -> None:
        if self.result is None:
            return

        self.preview_refresh_timer.stop()
        if self.adjustment_drag_depth > 0:
            return

        self.statusBar().showMessage("Updating preview...", 2000)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._show_adjusted_still_previews()
            self.wiggle_frames = [
                _rgb_array_to_pixmap(frame)
                for frame in build_animation_frames(
                    self.result,
                    self._fixed_adjustments(),
                    self._moving_adjustments(),
                    self._animation_mode(),
                )
            ]
            self.wiggle_index %= len(self.wiggle_frames)
            self.wiggle_preview.set_pixmap(self.wiggle_frames[self.wiggle_index])
            self.timer.start(self._wiggle_delay_ms())
            self.statusBar().showMessage("Preview updated", 2000)
        finally:
            QApplication.restoreOverrideCursor()

    def _show_adjusted_still_previews(self) -> None:
        if self.result is None:
            return

        adjusted_fixed = apply_adjustments(
            self.result.fixed_crop,
            self._fixed_adjustments(),
        )
        adjusted_moving = apply_adjustments(
            self.result.moving_crop,
            self._moving_adjustments(),
        )
        self.first_preview.set_pixmap(_array_to_pixmap(adjusted_fixed))
        self.second_preview.set_pixmap(_array_to_pixmap(adjusted_moving))

    def _advance_wiggle(self) -> None:
        if not self.wiggle_frames:
            return
        self.wiggle_index = (self.wiggle_index + 1) % len(self.wiggle_frames)
        self.wiggle_preview.set_pixmap(self.wiggle_frames[self.wiggle_index])

    def _export_gif(self) -> None:
        if self.result is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export wiggle GIF", "wiggle.gif", "GIF files (*.gif)"
        )
        if not selected:
            return
        try:
            export_gif(
                self.result,
                selected,
                self._wiggle_delay_ms(),
                self._fixed_adjustments(),
                self._moving_adjustments(),
                self._animation_mode(),
            )
        except Exception as exc:  # pragma: no cover - GUI boundary
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved GIF: {selected}", 7000)

    def _export_mp4(self) -> None:
        if self.result is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export wiggle MP4", "wiggle.mp4", "MP4 files (*.mp4)"
        )
        if not selected:
            return
        try:
            export_mp4(
                self.result,
                selected,
                self._wiggle_delay_ms(),
                self._fixed_adjustments(),
                self._moving_adjustments(),
                self._animation_mode(),
            )
        except Exception as exc:  # pragma: no cover - GUI boundary
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved MP4: {selected}", 7000)


def _array_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = image_bgr[:, :, ::-1].copy()
    return _rgb_array_to_pixmap(rgb)


def _rgb_array_to_pixmap(image_rgb: np.ndarray) -> QPixmap:
    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width
    qimage = QImage(
        image_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)
