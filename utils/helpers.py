from typing import Optional, Tuple
from PyQt5.QtCore import Qt, QEventLoop, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon
from PyQt5.QtWidgets import QApplication, QDialog


def info(msg: str):
    print(f"[INFO] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

def err(msg: str):
    print(f"[ERROR] {msg}")


def _normalize_point_result(res):
    if not res:
        return None
    if isinstance(res, (tuple, list)):
        first = res[0] if len(res) >= 1 else None
        if hasattr(first, "x") and hasattr(first, "y"):
            return int(first.x()), int(first.y())
        if len(res) >= 2 and all(isinstance(v, (int, float)) for v in res[:2]):
            return int(res[0]), int(res[1])
        if isinstance(first, dict) and "x" in first and "y" in first:
            return int(first["x"]), int(first["y"])
    if hasattr(res, "x") and hasattr(res, "y"):
        return int(res.x()), int(res.y())
    if isinstance(res, dict) and "x" in res and "y" in res:
        return int(res["x"]), int(res["y"])
    return None


class _InlinePointOverlay(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setWindowOpacity(0.01)
        self.setModal(True)
        self._pt = None
        virt = QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(virt)
        self.raise_(); self.activateWindow()
        self.setMouseTracking(False)

    @staticmethod
    def pick(parent=None):
        dlg = _InlinePointOverlay(parent)
        res = dlg.exec_()
        if res == QDialog.Accepted and dlg._pt is not None:
            return dlg._pt
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pt = e.globalPos(); self.accept()
        elif e.button() == Qt.RightButton:
            self.reject()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.reject()


def safe_select_point(parent=None):
    app = QApplication.instance()
    if app is not None:
        try:
            setattr(app, "isPickingPoint", True)
        except Exception:
            pass
    try:
        pt = None
        try:
            if 'PointSelector' in globals() and hasattr(PointSelector, 'select_point'):
                pt = PointSelector.select_point(None)
        except Exception as ex:
            err(f"PointSelector.select_point failed: {ex}. Falling back to inline overlay.")
            pt = None
        if not pt:
            try:
                dlg = _InlinePointOverlay(parent)
                res = dlg.exec_()
                if res == QDialog.Accepted and getattr(dlg, "_pt", None) is not None:
                    pt = dlg._pt
                try:
                    if dlg.isVisible():
                        dlg.close()
                    dlg.deleteLater()
                except Exception:
                    pass
            except Exception as ex:
                err(f"_InlinePointOverlay failed: {ex}")
                pt = None
        return pt
    finally:
        try:
            QApplication.processEvents(QEventLoop.AllEvents, 50)
        except Exception:
            pass
        try:
            QTimer.singleShot(0, lambda: QApplication.processEvents(QEventLoop.AllEvents, 50))
        except Exception:
            pass
        if parent is not None:
            try:
                st = parent.windowState()
                if st & Qt.WindowMinimized:
                    parent.setWindowState(st & ~Qt.WindowMinimized)
                if not parent.isVisible():
                    try:
                        parent.showNormal()
                    except Exception:
                        parent.show()
                parent.setEnabled(True)
                try:
                    if parent.windowOpacity() < 0.99:
                        parent.setWindowOpacity(1.0)
                except Exception:
                    pass
                parent.show()
                parent.raise_()
                parent.activateWindow()
                try:
                    QApplication.setActiveWindow(parent)
                except Exception:
                    pass
                QTimer.singleShot(0, parent.raise_)
                QTimer.singleShot(0, parent.activateWindow)
            except Exception:
                pass
        if app is not None:
            try:
                setattr(app, "isPickingPoint", False)
            except Exception:
                pass


def make_letter_icon(ch: str, bg="#333", fg="#eee") -> QIcon:
    pm = QPixmap(64, 64); pm.fill(Qt.transparent)
    p = QPainter(pm); p.fillRect(0,0,64,64, QColor(bg))
    p.setPen(QColor(fg)); p.drawText(0,0,64,64, Qt.AlignCenter, ch)
    p.end()
    return QIcon(pm)


def hk_normalize(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().lower().replace(" ", "")
    s = s.replace("ctrl+", "ctrl+").replace("shift+", "shift+").replace("alt+", "alt+").replace("win+", "win+")
    return s


def hk_pretty(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    parts = s.split("+")
    parts = [p.capitalize() if p not in ("ctrl","shift","alt","win") else {"ctrl":"Ctrl","shift":"Shift","alt":"Alt","win":"Win"}[p] for p in parts]
    return "+".join(parts)


def hk_to_tuple(combo: Optional[str]) -> Tuple[frozenset, Optional[str]]:
    if not combo:
        return (frozenset(), None)
    parts = combo.split("+")
    mods = set([p for p in parts[:-1] if p in ("ctrl","shift","alt","win")])
    base = parts[-1] if parts else None
    if base in ("ctrl","shift","alt","win"):
        return (frozenset(), None)
    return (frozenset(mods), base)
