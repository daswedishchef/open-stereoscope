from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .gui import MainWindow, app_icon


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("open-stereoscope")
    app.setOrganizationName("open-stereoscope")
    app.setWindowIcon(app_icon())

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
