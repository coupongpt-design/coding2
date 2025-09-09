import sys
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox

from gui.main_window import MainWindow


def _excepthook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    try:
        QMessageBox.critical(None, "Unhandled Error", f"{value}")
    except Exception:
        pass


def main():
    app = QApplication(sys.argv)
    sys.excepthook = _excepthook
    w = MainWindow(); w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
