# macro_fixed_roi_final.py
# - Crosshair overlay on click/drag targets
# - HiDPI-safe ROI/Point selectors (Qt.Window|Tool, parented to editor, ESC-safe, focus/raise)
# - Image steps + Not-Image steps (key, key_down, key_up, key_hold, click_point L/R/M, drag, scroll)
# - Recorder (toggle), Pause/Resume (F8), save/load .macro (v6 + optional search_roi)
# - "Hold Until Next Image" (keep clicking A until next image appears)
# - Click reliability: move→pre-dwell→mouseDown→hold→mouseUp→post-dwell, overlay after click
# - GLOBAL HOTKEYS (configurable): Run/Stop/Record (defaults: End/Home/F9)
# - Hotkey Settings dialog + labels "Run (End)" / "Stop (Home)" / "RECORD (F9)"
# - NEW: Image step ROI (search range) editor + runner-side ROI-aware matching
import sys, json, time, uuid, zipfile, traceback
from dataclasses import dataclass, asdict, field, fields as dataclass_fields
from typing import List, Optional, Tuple
import numpy as np
import cv2
import mss
import pyautogui
from pynput import keyboard, mouse
from PyQt5.QtCore import (
    Qt, QRect, QPoint, QSize, QThread, pyqtSignal,
    QSettings, QTimer, QEventLoop, QObject
)
from PyQt5.QtGui import (
    QGuiApplication, QPixmap, QImage, QPainter,
    QColor, QIcon, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QPlainTextEdit, QListWidget, QListWidgetItem, QMainWindow, QAction,
    QCheckBox, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit,
    QMessageBox, QShortcut, QDialogButtonBox, QGroupBox, QComboBox, QMenu, QInputDialog
)

if 'info' not in globals():
    def info(msg: str): print(f"[INFO] {msg}")
if 'warn' not in globals():
    def warn(msg: str): print(f"[WARN] {msg}")
if 'err' not in globals():
    def err(msg: str): print(f"[ERROR] {msg}")

if '_normalize_point_result' not in globals():
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

if '_InlinePointOverlay' not in globals():
    class _InlinePointOverlay(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
            self.setWindowOpacity(0.01)  # 완전투명은 이벤트 미수신 → 극소 불투명
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

if '_safe_select_point' not in globals():
    def _safe_select_point(parent=None):
        """
        안전한 포인트 선택 래퍼:
        - PointSelector.select_point(None) 우선 시도 (부모 비모달/스택킹 충돌 최소화)
        - 실패/취소 시 인라인 백업 오버레이 사용
        - 종료 후 이벤트 큐 비우기 + (가능 시) 부모 복귀까지 처리
        """
        app = QApplication.instance()
        # 전역 플래그(선택): 다른 모듈에서 선택중 상태를 참고할 수 있음
        if app is not None:
            try:
                setattr(app, "isPickingPoint", True)
            except Exception:
                pass

        try:
            pt = None

            # 1) 정식 픽커 우선
            try:
                if 'PointSelector' in globals() and hasattr(PointSelector, 'select_point'):
                    # parent=None로 올려서 모달/스택킹 간섭 줄임
                    pt = PointSelector.select_point(None)
            except Exception as ex:
                err(f"PointSelector.select_point failed: {ex}. Falling back to inline overlay.")
                pt = None

            # 2) 실패·취소 시 백업 오버레이
            if not pt:
                try:
                    dlg = _InlinePointOverlay(parent)
                    res = dlg.exec_()
                    if res == QDialog.Accepted and getattr(dlg, "_pt", None) is not None:
                        pt = dlg._pt
                    # 방금 띄운 백업 오버레이 즉시 정리
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
            # ===== 오버레이 쪽 정리(메모리/포커스 잔재 방지) =====
            # 1) 이벤트 큐를 한 틱 비워서 중첩 루프 여파 제거
            try:
                QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception:
                pass
            # 2) 다음 틱에서도 한 번 더 비워 승격/포커스 지연 보정
            try:
                QTimer.singleShot(0, lambda: QApplication.processEvents(QEventLoop.AllEvents, 50))
            except Exception:
                pass
            # 3) 가능하면 부모 다이얼로그를 전면/활성으로 복귀
            if parent is not None:
                try:
                    # 최소화 복구
                    st = parent.windowState()
                    if st & Qt.WindowMinimized:
                        parent.setWindowState(st & ~Qt.WindowMinimized)
                    # 보이기/활성/승격
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
                    # 다음 틱 재승격 (일부 WM에서 즉시 승격 무시되는 경우 대비)
                    QTimer.singleShot(0, parent.raise_)
                    QTimer.singleShot(0, parent.activateWindow)
                except Exception:
                    pass
            # 4) 전역 플래그 해제
            if app is not None:
                try:
                    setattr(app, "isPickingPoint", False)
                except Exception:
                    pass

    
# ---------------- Utilities ----------------
def cvimg_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def encode_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return bytes(buf) if ok else b""

# === Matcher helpers (module-level; duplicates OK) ===
def _apply_preprocess(img_bgr, step):
    img = img_bgr
    if getattr(step, "pre_gray", False) or getattr(step, "pre_edge", False) or getattr(step, "pre_clahe", False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if getattr(step, "pre_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
    ksize = int(getattr(step, "pre_blur_ksize", 1))
    if ksize and ksize >= 3 and ksize % 2 == 1:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    if getattr(step, "pre_edge", False):
        try:
            img = cv2.Canny(img, 80, 160)
        except Exception:
            pass
    if getattr(step, "pre_sharpen", False) and len(getattr(img, "shape", [])) == 2:
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, k)
    return img

def _resize_keep(template, scale):
    nh, nw = max(1, int(template.shape[0]*scale)), max(1, int(template.shape[1]*scale))
    if nh < 2 or nw < 2:
        return None
    return cv2.resize(template, (nw, nh), interpolation=cv2.INTER_LINEAR)

def _rotate_template(tpl, angle_deg):
    h, w = tpl.shape[:2]
    center = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(tpl, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rot

def _extract_topk(resmap, k=1, tpl_w=0, tpl_h=0):
    cands = []
    res = resmap.copy()
    for _ in range(max(1, int(k))):
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        cands.append((maxv, maxloc[0], maxloc[1]))
        x0, y0 = maxloc[0], maxloc[1]
        x1 = max(0, x0 - tpl_w//3); y1 = max(0, y0 - tpl_h//3)
        x2 = min(res.shape[1]-1, x0 + tpl_w//3); y2 = min(res.shape[0]-1, y0 + tpl_h//3)
        res[y1:y2+1, x1:x2+1] = -1.0
    return cands

# ---------------- Advanced Matcher ----------------
class MatchResult:
    def __init__(self, ok: bool, x: int = None, y: int = None, score: float = 0.0, w: int = 0, h: int = 0, stage: str = "single", scale: float = 1.0, angle: float = 0.0):
        self.ok = ok; self.x = x; self.y = y; self.score = score; self.w = w; self.h = h; self.stage = stage; self.scale = scale; self.angle = angle

class Matcher:
    def __init__(self):
        pass

    # ====== 내부 헬퍼: 전처리/마스크/회전/스케일 캐시 ======
    def _pp_key(self, step):
        # 전처리 파라미터가 바뀌면 키도 바뀌도록 구성
        return (
            bool(getattr(step, "pre_gray", True)),
            int(getattr(step, "pre_blur_ksize", 1)),
            bool(getattr(step, "pre_clahe", False)),
            bool(getattr(step, "pre_edge", False)),
            bool(getattr(step, "pre_sharpen", False)),
        )

    def _get_pp_tpl(self, tpl_bgr, step):
        cache = getattr(step, "_tpl_cache", None)
        if cache is None:
            step._tpl_cache = cache = {}
        key = ("pp", self._pp_key(step))
        hit = cache.get(key)
        if hit is not None:
            return hit
        pp = _apply_preprocess(tpl_bgr, step)
        cache[key] = pp
        return pp

    def _get_rot_tpl(self, tpl_pp, step, ang):
        cache = getattr(step, "_tpl_cache", None)
        if cache is None:
            step._tpl_cache = cache = {}
        key = ("rot", self._pp_key(step), float(ang))
        hit = cache.get(key)
        if hit is not None:
            return hit
        rot = _rotate_template(tpl_pp, float(ang))
        cache[key] = rot
        return rot

    def _get_scale_tpl(self, tpl_pp, step, sc):
        cache = getattr(step, "_tpl_cache", None)
        if cache is None:
            step._tpl_cache = cache = {}
        key = ("scale", self._pp_key(step), float(sc))
        hit = cache.get(key)
        if hit is not None:
            return hit
        tpl_s = _resize_keep(tpl_pp, float(sc))
        cache[key] = tpl_s
        return tpl_s

    def _get_mask(self, tpl_like, step, mode: str):
        """
        mode: 'edge'|'thresh'|'none'
        템플릿 전처리 키 + (scale/rot 포함되었으면 템플릿 자체 id) 로 캐시.
        """
        cache = getattr(step, "_tpl_cache", None)
        if cache is None:
            step._tpl_cache = cache = {}
        ident = int(tpl_like.__array_interface__['data'][0])  # ndarray 포인터로 간단 식별
        key = ("mask", self._pp_key(step), mode, ident)
        hit = cache.get(key)
        if hit is not None:
            return hit

        mask = None
        try:
            if len(tpl_like.shape) == 2:
                if mode == "edge":
                    m = cv2.Canny(tpl_like, 50, 100)
                    mask = (m > 0).astype("uint8") * 255
                elif mode == "thresh" and int(getattr(step, "mask_thresh", 0)) > 0:
                    _, m = cv2.threshold(tpl_like, int(step.mask_thresh), 255, cv2.THRESH_BINARY)
                    mask = m.astype("uint8")
        except Exception:
            mask = None
        cache[key] = mask
        return mask

    # ====== 메인 엔트리: 예산 엄수 + 단계별 조기종료 ======
    def find_best(self, frame_bgr: np.ndarray, step) -> MatchResult:
        tpl_bgr = step.ensure_tpl()
        if tpl_bgr is None:
            return MatchResult(False)

        # 공통 파라미터
        th = float(getattr(step, "min_confidence", getattr(step, "threshold", 0.85)))
        delta = 0.03  # 충분히 큰 점수면 조기종료 허용치
        budget_ms = int(getattr(step, "budget_ms", 0))
        t0 = time.time()

        def over_budget():
            if budget_ms <= 0:
                return False
            return (time.time() - t0) * 1000.0 > budget_ms

        # 템플릿 전처리(캐시)
        tpl_pp = self._get_pp_tpl(tpl_bgr, step)

        # 1) Single-scale
        if over_budget():
            return MatchResult(False, stage="budget")
        res = self._single_scale(_apply_preprocess(frame_bgr, step), tpl_pp, step)
        if res and res[0][0] >= th:
            sc, x, y = res[0]
            return MatchResult(True, x + tpl_pp.shape[1]//2, y + tpl_pp.shape[0]//2, sc,
                               tpl_pp.shape[1], tpl_pp.shape[0], stage="single")
        # 점수가 충분히 높으면 다음 단계 스킵
        if res and res[0][0] >= (th + delta):
            sc, x, y = res[0]
            return MatchResult(True, x + tpl_pp.shape[1]//2, y + tpl_pp.shape[0]//2, sc,
                               tpl_pp.shape[1], tpl_pp.shape[0], stage="single-fast")

        # 2) Multiscale
        if getattr(step, "ms_enable", False) and not over_budget():
            best_ms, _ = self._multiscale(_apply_preprocess(frame_bgr, step), tpl_pp, step,
                                          th, delta, over_budget)
            if best_ms and best_ms[0][0] >= th:
                sc, x, y, s = best_ms[0]
                tw, thh = int(tpl_pp.shape[1]*s), int(tpl_pp.shape[0]*s)
                return MatchResult(True, x + tw//2, y + thh//2, sc, tw, thh, stage="multiscale", scale=s)

        # 3) Rotation
        if getattr(step, "rot_enable", False) and not over_budget():
            best_rot, _ = self._rotation(_apply_preprocess(frame_bgr, step), tpl_pp, step,
                                         th, delta, over_budget)
            if best_rot and best_rot[0][0] >= th:
                sc, x, y, ang = best_rot[0]
                return MatchResult(True, x + tpl_pp.shape[1]//2, y + tpl_pp.shape[0]//2, sc,
                                   tpl_pp.shape[1], tpl_pp.shape[0], stage="rotation", angle=ang)

        # 4) ORB fallback (비용 상한 + 축소/ROI 우선)
        if getattr(step, "feat_fallback_enable", False) and not over_budget():
            mr = self._orb_fallback(frame_bgr, tpl_bgr, step, over_budget)
            if mr and mr.ok and mr.score >= th * 0.9:
                mr.stage = "feature"
                return mr

        return MatchResult(False)

    def _single_scale(self, img_pp, tpl_pp, step):
        try:
            res = cv2.matchTemplate(img_pp, tpl_pp, cv2.TM_CCOEFF_NORMED)
            topk = _extract_topk(res, int(getattr(step, "top_k", 1)), tpl_pp.shape[1], tpl_pp.shape[0])
            return topk
        except Exception:
            return None

    def _multiscale(self, img_pp, tpl_pp, step, th, delta, over_budget_cb):
        smin = max(0.3, float(getattr(step, "ms_min_scale", 0.9)))
        smax = min(3.0, float(getattr(step, "ms_max_scale", 1.1)))
        sstep = max(1.01, float(getattr(step, "ms_step", 1.05)))
        scales = [1.0]
        s = 1.0
        while s / sstep >= smin:
            s /= sstep; scales.append(s)
        s = 1.0
        while s * sstep <= smax:
            s *= sstep; scales.append(s)
        scales = sorted(set([round(x,4) for x in scales]), key=lambda v: abs(v-1.0))

        best = []
        best_seen = -1.0
        for sc in scales:
            if over_budget_cb():
                break
            tpl_s = self._get_scale_tpl(tpl_pp, step, sc)
            if tpl_s is None or min(tpl_s.shape[:2]) < 2:
                continue

            # 마스크 준비(옵션)
            mask = None
            if getattr(step, 'mask_enable', False) and len(tpl_s.shape) == 2:
                mode = "edge" if getattr(step, "mask_auto_edge", False) else ("thresh" if getattr(step, "mask_thresh", 0) > 0 else "none")
                if mode != "none":
                    mask = self._get_mask(tpl_s, step, mode)

            try:
                if mask is not None:
                    res = cv2.matchTemplate(img_pp, tpl_s, cv2.TM_CCOEFF_NORMED, mask=mask)
                else:
                    res = cv2.matchTemplate(img_pp, tpl_s, cv2.TM_CCOEFF_NORMED)
                cands = _extract_topk(res, int(getattr(step, "top_k", 1)), tpl_s.shape[1], tpl_s.shape[0])
                for (score, x, y) in cands:
                    best.append((score, x, y, sc))
                    if score > best_seen:
                        best_seen = score
            except Exception:
                continue

            # 충분히 높은 점수면 조기 종료
            if best_seen >= (th + delta):
                break

        best.sort(key=lambda t: t[0], reverse=True)
        return best, {"scales": scales}

    def _rotation(self, img_pp, tpl_pp, step, th, delta, over_budget_cb):
        rmin = float(getattr(step, "rot_min_deg", -10.0))
        rmax = float(getattr(step, "rot_max_deg", 10.0))
        rstep = max(1.0, float(getattr(step, "rot_step_deg", 5.0)))
        angles = []
        a = 0.0
        while a - rstep >= rmin:
            a -= rstep; angles.append(round(a,2))
        a = 0.0
        while a + rstep <= rmax:
            a += rstep; angles.append(round(a,2))
        angles = sorted(set([round(x,2) for x in angles]), key=lambda v: abs(v))

        best = []
        best_seen = -1.0
        for ang in angles:
            if over_budget_cb():
                break
            tpl_r = self._get_rot_tpl(tpl_pp, step, ang)
            mask = None
            if getattr(step, 'mask_enable', False) and len(tpl_r.shape) == 2:
                mode = "edge" if getattr(step, "mask_auto_edge", False) else ("thresh" if getattr(step, "mask_thresh", 0) > 0 else "none")
                if mode != "none":
                    mask = self._get_mask(tpl_r, step, mode)
            try:
                if mask is not None:
                    res = cv2.matchTemplate(img_pp, tpl_r, cv2.TM_CCOEFF_NORMED, mask=mask)
                else:
                    res = cv2.matchTemplate(img_pp, tpl_r, cv2.TM_CCOEFF_NORMED)
                cands = _extract_topk(res, int(getattr(step, "top_k", 1)), tpl_r.shape[1], tpl_r.shape[0])
                for (score, x, y) in cands:
                    best.append((score, x, y, ang))
                    if score > best_seen:
                        best_seen = score
            except Exception:
                continue
            if best_seen >= (th + delta):
                break
        best.sort(key=lambda t: t[0], reverse=True)
        return best, {"angles": angles}

    def _orb_fallback(self, img_bgr, tpl_bgr, step, over_budget_cb=None):
        try:
            if over_budget_cb and over_budget_cb():
                return None
            n = int(getattr(step, "feat_nfeatures", 500))
            orb = cv2.ORB_create(nfeatures=max(100, n))

            # ROI가 있으면 ROI만, 없으면 프레임 축소본 사용(비용 상한)
            frame = img_bgr
            if getattr(step, "search_roi_enabled", False) and int(getattr(step, "search_roi_width", 0)) > 0 and int(getattr(step, "search_roi_height", 0)) > 0:
                # 호출부(러너)가 이미 ROI 캡처했을 가능성 높지만, 안전하게 그대로 진행
                pass
            else:
                # 다운스케일 0.75
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (max(1,int(w*0.75)), max(1,int(h*0.75))), interpolation=cv2.INTER_AREA)

            kp1, des1 = orb.detectAndCompute(tpl_bgr, None)
            kp2, des2 = orb.detectAndCompute(frame, None)
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            ratio = float(getattr(step, "feat_match_ratio", 0.75))
            good = []
            # 비용 상한: 상위 200 페어만 평가
            head = matches[:min(200, len(matches))]
            for m, n in head:
                if m.distance < ratio * n.distance:
                    good.append(m)
            if len(good) < 4:
                return None

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            thr = float(getattr(step, "feat_ransac_reproj_thresh", 3.0))
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, thr)
            if M is None:
                return None
            h, w = tpl_bgr.shape[:2]
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            trans = cv2.perspectiveTransform(corners, M)
            cx = int(np.mean(trans[:,0,0])); cy = int(np.mean(trans[:,0,1]))
            inliers = int(mask.sum()) if mask is not None else 0
            min_inliers = 12  # 실전 안전핀
            if inliers < min_inliers:
                return None
            score = min(1.0, inliers / max(10.0, len(good)))
            return MatchResult(True, cx, cy, score, w, h, stage="feature")
        except Exception:
            return None

def decode_png_bytes(png: Optional[bytes]) -> Optional[np.ndarray]:
    if not png: return None
    data = np.frombuffer(png, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def make_letter_icon(ch: str, bg="#333", fg="#eee") -> QIcon:
    pm = QPixmap(64, 64); pm.fill(Qt.transparent)
    p = QPainter(pm); p.fillRect(0,0,64,64, QColor(bg))
    p.setPen(QColor(fg)); p.drawText(0,0,64,64, Qt.AlignCenter, ch)
    p.end()
    return QIcon(pm)

# --- Hotkey helpers ---
def hk_normalize(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().lower().replace(" ", "")
    s = s.replace("ctrl+", "ctrl+").replace("shift+", "shift+").replace("alt+", "alt+").replace("win+", "win+")
    return s

def hk_pretty(s: Optional[str]) -> Optional[str]:
    if not s: return None
    parts = s.split("+")
    parts = [p.capitalize() if p not in ("ctrl","shift","alt","win") else {"ctrl":"Ctrl","shift":"Shift","alt":"Alt","win":"Win"}[p] for p in parts]
    return "+".join(parts)

def hk_to_tuple(combo: Optional[str]) -> Tuple[frozenset, Optional[str]]:
    if not combo: return (frozenset(), None)
    parts = combo.split("+")
    mods = set([p for p in parts[:-1] if p in ("ctrl","shift","alt","win")])
    base = parts[-1] if parts else None
    if base in ("ctrl","shift","alt","win"):
        return (frozenset(), None)
    return (frozenset(mods), base)

# ---------------- Crosshair Overlay ----------------
class CrosshairOverlay(QWidget):
    def __init__(self, virt_left, virt_top, virt_w, virt_h, x, y, duration_ms=300):
        super().__init__(parent=None, flags=Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)  # ← 포커스 안 뺏기
        self.setGeometry(virt_left, virt_top, virt_w, virt_h)

        self._virt_left = virt_left; self._virt_top = virt_top
        self._x = x; self._y = y
        self._timer = QTimer(self); self._timer.setInterval(max(1, int(duration_ms))); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close)

        self.show()
        self.raise_()
        QTimer.singleShot(0, self.raise_)  # ← 여기! 다음 이벤트 틱에서 한 번 더 올려줌
        # self.activateWindow()는 굳이 필요 없음(포커스 뺏을 수 있어 생략 권장)

        self._timer.start()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QColor(0, 255, 0, 220))
        lx = self._x - self._virt_left
        ly = self._y - self._virt_top
        p.drawLine(lx - 10, ly, lx + 10, ly)
        p.drawLine(lx, ly - 10, lx, ly + 10)
        p.end()

# ---------------- ROI / Point Selector ----------------
class ROISelector(QWidget):
    done = pyqtSignal(QRect, np.ndarray, tuple)  # rect, crop_bgr, (virt_left, virt_top, virt_w, virt_h)

    def __init__(self, frame_bgr: np.ndarray, virt_left: int, virt_top: int, virt_size: QSize, parent=None):
        super().__init__(parent, flags=Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._frame_bgr = frame_bgr
        self._virt_left = virt_left
        self._virt_top = virt_top
        self._virt_size = virt_size

        self._drag = False
        self._start = QPoint()
        self._end = QPoint()

        # 물리↔DIP 스케일 (프레임은 물리 픽셀, 위젯은 DIP)
        fh, fw = frame_bgr.shape[:2]
        vw, vh = max(1, virt_size.width()), max(1, virt_size.height())
        self._fx = fw / float(vw)
        self._fy = fh / float(vh)

        # 전체 가상화면 덮기
        self.setGeometry(virt_left, virt_top, virt_size.width(), virt_size.height())
        self.show()
        self.raise_()
        self.activateWindow()

    def keyPressEvent(self, e):
        # ESC로 취소
        if e.key() == Qt.Key_Escape:
            self.done.emit(QRect(), None, (self._virt_left, self._virt_top, self._virt_size.width(), self._virt_size.height()))
            self.close()

    def mousePressEvent(self, e):
        self._drag = True
        self._start = e.pos()
        self._end = e.pos()
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag:
            self._end = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = False
        self._end = e.pos()
        self.update()

        x1w, y1w = min(self._start.x(), self._end.x()), min(self._start.y(), self._end.y())
        x2w, y2w = max(self._start.x(), self._end.x()), max(self._start.y(), self._end.y())

        # 위젯좌표(DIP) → 물리 픽셀
        x1 = int(round(x1w * self._fx)); y1 = int(round(y1w * self._fy))
        x2 = int(round(x2w * self._fx)); y2 = int(round(y2w * self._fy))
        w = max(0, x2 - x1); h = max(0, y2 - y1)

        if w < 3 or h < 3:
            # 너무 작으면 취소 (물리 좌표 기준)
            self.done.emit(QRect(), None, (self._virt_left, self._virt_top, self._virt_size.width(), self._virt_size.height()))
        else:
            crop = self._frame_bgr[y1:y1+h, x1:x1+w].copy()
            # 사각형은 "가상 데스크톱 절대 좌표계 대비 상대(좌상단 기준) 물리 픽셀"로 전달
            rect_phys = QRect(x1, y1, w, h)
            self.done.emit(rect_phys, crop, (self._virt_left, self._virt_top, self._virt_size.width(), self._virt_size.height()))
        self.close()

    def paintEvent(self, e):
        p = QPainter(self)
        rgb = cv2.cvtColor(self._frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3*rgb.shape[1], QImage.Format_RGB888)
        # 이미지 전체를 위젯 크기에 맞춰 스케일링 렌더
        p.drawImage(self.rect(), qimg)

        # 반투명 어둡게
        p.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self._drag:
            rect = QRect(self._start, self._end).normalized()
            # 선택 영역은 밝게
            p.fillRect(rect, QColor(255, 255, 255, 40))
            p.setPen(QColor(0, 255, 0, 255))
            p.drawRect(rect)
        p.end()

    @staticmethod
    def _grab_virtual_screen_bgr():
        """mss 우선, 실패 시 QScreen.virtualGeometry()로 폴백(음수 좌표 포함)."""
        try:
            with mss.mss() as sct:
                mon = sct.monitors[0]  # virtual desktop
                frame = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3].copy()  # BGR
                return frame, (int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"]))
        except Exception:
            screen = QApplication.primaryScreen()
            vg = screen.virtualGeometry()  # ← 가상 전체 데스크톱의 실제 (left, top, w, h)
            pm = screen.grabWindow(0)      # 전체 데스크톱 이미지
            qi = pm.toImage().convertToFormat(QImage.Format_RGB888)
            w, h = qi.width(), qi.height()
            ptr = qi.bits(); ptr.setsize(qi.byteCount())
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return bgr, (vg.x(), vg.y(), vg.width(), vg.height())

    @staticmethod
    def select_from_screen(parent=None):
        frame_bgr, virt = ROISelector._grab_virtual_screen_bgr()
        virt_left, virt_top, virt_w, virt_h = virt

        dlg = ROISelector(frame_bgr, virt_left, virt_top, QSize(virt_w, virt_h), parent)
        result = {}

        loop = QEventLoop()

        def on_done(rect, crop, v):
            result["rect"] = rect
            result["crop"] = crop
            result["virt"] = v
            loop.quit()

        dlg.done.connect(on_done)
        # 사용 안내가 필요하면 여기에서 status bar나 로그로 출력
        loop.exec_()

        if "rect" not in result:
            return QRect(), None, (virt_left, virt_top, virt_w, virt_h)
        return result["rect"], result["crop"], result["virt"]


class PointSelector(QWidget):
    done = pyqtSignal(QPoint, tuple)  # pos, virtual bbox

    def __init__(self, frame_bgr: np.ndarray, virt_left: int, virt_top: int, virt_size: QSize, parent=None):
        super().__init__(parent, flags=Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowOpacity(0.4)
        self._frame = frame_bgr
        self._virt_left = virt_left; self._virt_top = virt_top; self._virt_size = virt_size

        fh, fw = frame_bgr.shape[:2]
        vw, vh = max(1, virt_size.width()), max(1, virt_size.height())
        self._fx = fw / float(vw)
        self._fy = fh / float(vh)

        self.setGeometry(virt_left, virt_top, virt_size.width(), virt_size.height())
        self.show(); self.raise_(); self.activateWindow()

    def keyPressEvent(self, e):
        # ESC 취소 지원
        if e.key() == Qt.Key_Escape:
            self.done.emit(QPoint(), (self._virt_left, self._virt_top, self._virt_size.width(), self._virt_size.height()))
            self.close()

    def mousePressEvent(self, e):
        # 위젯좌표(DIP) → 물리 픽셀
        px = int(round(e.pos().x() * self._fx))
        py = int(round(e.pos().y() * self._fy))
        self.done.emit(QPoint(px, py), (self._virt_left, self._virt_top, self._virt_size.width(), self._virt_size.height()))
        self.close()

    def paintEvent(self, e):
        p = QPainter(self)
        # BGR → RGB 보정
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3*rgb.shape[1], QImage.Format_RGB888)
        p.drawImage(self.rect(), qimg)  # 위젯 크기에 맞춰 스케일
        p.end()

    @staticmethod
    def select_point(parent=None, timeout_ms: int = 15000, grid: bool = False, grid_step: int = 8):
        """
        Full-virtual-desktop point picker (safe, multi-monitor, mixed-DPI ready).
        - 좌클릭: 좌표 확정 / 우클릭·ESC·타임아웃: 취소
        - 혼합 DPI/멀티모니터 안전: 해당 스크린 DPR 반영 → '가상 데스크톱 물리 px'로 계산
        - 반환: QPoint(x, y) in virtual-physical px; 취소 시 None
        """
        app = QApplication.instance()
        # 전역 충돌(전역 핫키/레코더) 방지를 위해, 선택 중 플래그 세팅(있으면 다른 곳에서 가드 가능)
        if app is not None:
            try:
                setattr(app, "isPickingPoint", True)
            except Exception:
                pass

        class _Overlay(QWidget):
            picked = pyqtSignal(int, int)
            canceled = pyqtSignal()

            def __init__(self, timeout_ms=15000, parent=None):
                super().__init__(parent)
                self.setWindowFlags(
                    Qt.Window |
                    Qt.FramelessWindowHint |
                    Qt.WindowStaysOnTopHint |
                    Qt.NoDropShadowWindowHint
                )
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                # self.setAttribute(Qt.WA_ShowWithoutActivating, True)  # ← 제거
                self.setCursor(Qt.CrossCursor)
                self.setFocusPolicy(Qt.StrongFocus)
                # 모달 지정 절대 금지 (문제 원인)
                # self.setWindowModality(Qt.ApplicationModal)

                vrect = QApplication.primaryScreen().virtualGeometry()
                self.setGeometry(vrect)

                self._hover = None
                self._grid = bool(grid)
                self._grid_step = max(1, int(grid_step))

                self._timer = QTimer(self); self._timer.setSingleShot(True)
                self._timer.timeout.connect(self._on_timeout)
                self._timeout_ms = int(timeout_ms)

                # 보이자마자 가이드
                c = self.rect().center()
                self._hover = QPoint(c.x(), c.y())
                self._prime_hint = True

            def start(self):
                self.show()
                self.raise_()
                QTimer.singleShot(0, self.raise_)
                # 확실한 활성화/포커스
                QTimer.singleShot(0, lambda: (self.activateWindow(),
                                              self.setFocus(Qt.ActiveWindowFocusReason)))
                self._timer.start(self._timeout_ms)

            def paintEvent(self, _):
                p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
                p.fillRect(self.rect(), QColor(0, 0, 0, 60))
                if self._hover is not None:
                    p.setPen(QColor(0, 220, 120, 230))
                    x, y = self._hover.x(), self._hover.y()
                    p.drawLine(x, self.rect().top(), x, self.rect().bottom())
                    p.drawLine(self.rect().left(), y, self.rect().right(), y)
                    p.drawEllipse(self._hover, 6, 6)
                if self._prime_hint:
                    self._prime_hint = False

            def mouseMoveEvent(self, e):
                if self._grid:
                    gx = round(e.pos().x()/self._grid_step)*self._grid_step
                    gy = round(e.pos().y()/self._grid_step)*self._grid_step
                    self._hover = QPoint(gx, gy)
                else:
                    self._hover = e.pos()
                self.update()

            def mousePressEvent(self, e):
                if e.button() == Qt.RightButton:
                    self._cancel()

            def mouseReleaseEvent(self, e):
                if e.button() != Qt.LeftButton:
                    return
                gpos = e.globalPos()
                screen = QApplication.screenAt(gpos) or QApplication.primaryScreen()
                try:
                    dpr = float(screen.devicePixelRatio())
                except Exception:
                    dpr = 1.0
                self._finish(int(round(gpos.x()*dpr)), int(round(gpos.y()*dpr)))

            def keyPressEvent(self, e):
                if e.key() == Qt.Key_Escape:
                    self._cancel()
                else:
                    super().keyPressEvent(e)

            def _on_timeout(self):
                self._cancel()

            # ▼▼▼ 핵심: 확실한 해제 + 지연 emit ▼▼▼
            def _release_all(self):
                try: self.releaseMouse()
                except Exception: pass
                try: self.releaseKeyboard()
                except Exception: pass
                try: QGuiApplication.restoreOverrideCursor()
                except Exception: pass

            def _cancel(self):
                try: self._timer.stop()
                except Exception: pass
                self._release_all()
                self.close()
                QTimer.singleShot(0, self.canceled.emit)
                QTimer.singleShot(0, self.deleteLater)

            def _finish(self, x, y):
                try: self._timer.stop()
                except Exception: pass
                self._release_all()
                self.close()
                QTimer.singleShot(0, lambda: self.picked.emit(x, y))
                QTimer.singleShot(0, self.deleteLater)

            # ---------- lifecycle ----------
            def _on_timeout(self):
                self._cancel()

            def _cancel(self):
                try: self._timer.stop()
                except Exception: pass
                self.hide()
                self.canceled.emit()
                self.deleteLater()

            def _finish(self, x, y):
                try: self._timer.stop()
                except Exception: pass
                self.hide()
                self.picked.emit(x, y)
                self.deleteLater()


        prev = QApplication.activeWindow()

        overlay = _Overlay(timeout_ms=timeout_ms, parent=parent)
        result_holder = {"pt": None}
        loop = QEventLoop()

        def _picked(x, y):
            result_holder["pt"] = QPoint(int(x), int(y))
            loop.quit()

        def _canceled():
            result_holder["pt"] = None
            loop.quit()

        overlay.picked.connect(_picked)
        overlay.canceled.connect(_canceled)
        overlay.start()
        loop.exec_()

        # ✅ 안전 정리
        try:
            if overlay.isVisible():
                overlay.close()
            overlay.deleteLater()
        except Exception:
            pass
        QApplication.processEvents(QEventLoop.AllEvents, 50)

        # ✅ 포커스/활성 복구
        try:
            if prev is not None:
                prev.setEnabled(True)
                prev.raise_()
                prev.activateWindow()
                prev.setFocus(Qt.ActiveWindowFocusReason)
        except Exception:
            pass

        # 플래그 해제(기존 유지)
        if app is not None:
            try:
                setattr(app, "isPickingPoint", False)
            except Exception:
                pass

        return result_holder["pt"]

# ---------------- Step Model ----------------
@dataclass
class StepData:
    id: str
    name: str
    type: str  # "image_click", "key", "key_down", "key_up", "key_hold", "click_point", "drag", "scroll"
    # image matching
    png_bytes: Optional[bytes] = None
    threshold: float = 0.85
    timeout_ms: int = 5000
    poll_ms: int = 100
    jitter: int = 2
    pre_move_sleep_ms: int = 30
    press_duration_ms: int = 70
    post_click_sleep_ms: int = 80
    # click_point / drag / scroll / key
    click_button: str = "left"
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    drag_from_x: Optional[int] = None
    drag_from_y: Optional[int] = None
    drag_to_x: Optional[int] = None
    drag_to_y: Optional[int] = None
    drag_duration_ms: int = 200
    scroll_dx: int = 0
    scroll_dy: int = 0
    scroll_times: int = 1
    scroll_interval_ms: int = 0
    key_string: Optional[str] = None
    key_times: int = 1
    hold_ms: int = 0
    # hold until next image
    hold_until_next: bool = False
    hold_timeout_ms: int = 5000
    hold_reclick_interval_ms: int = 500
    hold_reacquire_each_time: bool = False
    hold_release_consecutive: int = 1


    # --- Advanced matching params ---
    pre_gray: bool = True
    pre_blur_ksize: int = 1
    pre_clahe: bool = False
    pre_edge: bool = False
    pre_sharpen: bool = False

    ms_enable: bool = False
    ms_min_scale: float = 0.80
    ms_max_scale: float = 1.20
    ms_step: float = 1.06

    rot_enable: bool = False
    rot_min_deg: float = -10.0
    rot_max_deg: float = 10.0
    rot_step_deg: float = 5.0

    feat_fallback_enable: bool = False
    feat_nfeatures: int = 500
    feat_match_ratio: float = 0.75
    feat_ransac_reproj_thresh: float = 3.0

    mask_enable: bool = False
    mask_auto_edge: bool = False
    mask_thresh: int = 15

    top_k: int = 1
    nms_enable: bool = False
    nms_iou: float = 0.3
    detect_consecutive: int = 1
    adaptive_threshold: bool = False
    min_confidence: float = 0.85

    click_anchor: str = "center"
    click_offset_x: int = 0
    click_offset_y: int = 0

    budget_ms: int = 0
    max_rotations: int = 0
    max_scales: int = 0
    # search ROI (absolute virtual desktop coords)
    search_roi_enabled: bool = False
    search_roi_left: int = 0
    search_roi_top: int = 0
    search_roi_width: int = 0
    search_roi_height: int = 0

    # --- Step chaining (on-fail) ---
    on_fail_action: str = "default"      # "default" | "continue" | "goto" | "abort"
    on_fail_target_id: Optional[str] = None # target step.id when action is "goto"

    # runtime only
    _tpl_bgr: Optional[np.ndarray] = None
    _last_match_xy: Optional[Tuple[int,int]] = None
    _tpl_cache: dict = field(default_factory=dict)

    def ensure_tpl(self) -> Optional[np.ndarray]:
        if self._tpl_bgr is not None: return self._tpl_bgr
        if not self.png_bytes: return None
        self._tpl_bgr = decode_png_bytes(self.png_bytes)
        return self._tpl_bgr

    def to_serializable(self) -> dict:
        d = asdict(self)
        d.pop('_tpl_cache', None)
        d.pop("_tpl_bgr", None)
        d.pop("_last_match_xy", None)
        if self.type == "image_click" and self.png_bytes:
            d["image_path"] = f"images/{self.id}.png"
        d.pop("png_bytes", None)
        # attach ROI block if enabled
        if self.type == 'image_click' and self.search_roi_enabled and self.search_roi_width > 0 and self.search_roi_height > 0:
            d['search_roi'] = {
                'left': int(self.search_roi_left),
                'top': int(self.search_roi_top),
                'width': int(self.search_roi_width),
                'height': int(self.search_roi_height)
            }

        # attach MATCH block (advanced params)
        if self.type == 'image_click':
            match = {
                'pre_gray': self.pre_gray,
                'pre_blur_ksize': int(self.pre_blur_ksize),
                'pre_clahe': self.pre_clahe,
                'pre_edge': self.pre_edge,
                'pre_sharpen': self.pre_sharpen,
                'ms_enable': self.ms_enable,
                'ms_min_scale': float(self.ms_min_scale),
                'ms_max_scale': float(self.ms_max_scale),
                'ms_step': float(self.ms_step),
                'rot_enable': self.rot_enable,
                'rot_min_deg': float(self.rot_min_deg),
                'rot_max_deg': float(self.rot_max_deg),
                'rot_step_deg': float(self.rot_step_deg),
                'feat_fallback_enable': self.feat_fallback_enable,
                'feat_nfeatures': int(self.feat_nfeatures),
                'feat_match_ratio': float(self.feat_match_ratio),
                'feat_ransac_reproj_thresh': float(self.feat_ransac_reproj_thresh),
                'mask_enable': self.mask_enable,
                'mask_auto_edge': self.mask_auto_edge,
                'mask_thresh': int(self.mask_thresh),
                'top_k': int(self.top_k),
                'nms_enable': self.nms_enable,
                'nms_iou': float(self.nms_iou),
                'detect_consecutive': int(self.detect_consecutive),
                'adaptive_threshold': self.adaptive_threshold,
                'min_confidence': float(self.min_confidence),
                'click_anchor': self.click_anchor,
                'click_offset_x': int(self.click_offset_x),
                'click_offset_y': int(self.click_offset_y),
                'budget_ms': int(self.budget_ms),
                'max_rotations': int(self.max_rotations),
                'max_scales': int(self.max_scales),
            }
            d['match'] = match
        return d


# ---------------- Repeat Config (Scenario-level) ----------------
@dataclass
class RepeatConfig:
    repeat_count: int = 1       # 0 = infinite
    repeat_cooldown_ms: int = 500 # cooldown between loops
    stop_on_fail: bool = True   # stop all on first step failure
    max_duration_ms: int = 0    # 0 = no limit

    def to_json(self) -> dict:
        return {
            "repeat_count": int(self.repeat_count),
            "repeat_cooldown_ms": int(self.repeat_cooldown_ms),
            "stop_on_fail": bool(self.stop_on_fail),
            "max_duration_ms": int(self.max_duration_ms),
        }

    @staticmethod
    def from_json(d: dict) -> "RepeatConfig":
        rc = RepeatConfig()
        if not isinstance(d, dict):
            return rc
        rc.repeat_count = int(d.get("repeat_count", rc.repeat_count))
        rc.repeat_cooldown_ms = int(d.get("repeat_cooldown_ms", rc.repeat_cooldown_ms))
        rc.stop_on_fail = bool(d.get("stop_on_fail", rc.stop_on_fail))
        rc.max_duration_ms = int(d.get("max_duration_ms", rc.max_duration_ms))
        return rc

# ---------------- Runner ----------------
class MacroRunner(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool)
    requestCrosshair = pyqtSignal(int, int, int)  # x, y, duration_ms

    def __init__(self, steps: List[StepData], repeat: RepeatConfig = None, dry_run: bool = False, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.repeat = repeat or RepeatConfig()
        self.dry_run = dry_run
        self._stop = False
        self._matcher = Matcher()

    def run(self):
        self._stop = False
        try:
            with mss.mss() as sct:
                mon = sct.monitors[0]
                loop_idx = 0
                t_start = time.time()
                # scenario-level repeat loop
                while (not self._stop) and (self.repeat.repeat_count == 0 or loop_idx < self.repeat.repeat_count):
                    self.log.emit(f"[Loop {loop_idx+1}/{self.repeat.repeat_count or '∞'}] start")
                    loop_idx += 1

                    # id -> index map for this loop
                    id2idx = {st.id: ix for ix, st in enumerate(self.steps)}
                    i = 0
                    hops = 0
                    max_hops = max(100, len(self.steps) * 10)
                    while i < len(self.steps):
                        if self._stop:
                            self.finished.emit(False); return

                        s = self.steps[i]
                        self.log.emit(f"[{i+1}/{len(self.steps)}] {s.type} :: {s.name}")
                        ok = self._exec_step(sct, mon, s, i)

                        if ok:
                            i += 1
                            continue

                        # ---- failure handling ----
                        self.log.emit("  !! failed")
                        act = s.on_fail_action
                        if act == "continue":
                            self.log.emit("  on-fail: continue → skip to next")
                            i += 1
                            continue

                        if act == "goto":
                            tgt = s.on_fail_target_id
                            j = id2idx.get(tgt) if tgt else None
                            if j is None:
                                self.log.emit(f"  on-fail: goto → target not found ({tgt}); fallback to next")
                                i += 1
                            else:
                                self.log.emit(f"  on-fail: goto → jump to step {j+1} [{self.steps[j].name}]")
                                i = j
                                hops += 1
                                if hops > max_hops:
                                    self.log.emit("  !! too many jumps; possible loop → abort")
                                    self.finished.emit(False); return
                            continue

                        if act == "abort":
                            self.log.emit("  on-fail: abort → stop runner")
                            self.finished.emit(False); return

                        # scenario policy fallback
                        if self.repeat.stop_on_fail:
                            self.finished.emit(False); return
                        else:
                            self.log.emit("  stop_on_fail=False → end this loop")
                            break

                    if self.repeat.repeat_cooldown_ms > 0 and (self.repeat.repeat_count == 0 or loop_idx < self.repeat.repeat_count):
                        time.sleep(self.repeat.repeat_cooldown_ms / 1000.0)

            self.finished.emit(True)
        except Exception as e:
            self.log.emit(f"!! runner exception: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False)

    def stop(self): self._stop = True

    def _exec_step(self, sct, mon, s: StepData, idx: int) -> bool:
        if s.type == "image_click": return self._image_click(sct, mon, s, idx)
        if s.type == "key": return self._key(s)
        if s.type == "key_down": return self._key_down(s)
        if s.type == "key_up": return self._key_up(s)
        if s.type == "key_hold": return self._key_hold(s)
        if s.type == "click_point": return self._click_point(s)
        if s.type == "drag": return self._drag(s)
        if s.type == "scroll": return self._scroll(s)
        self.log.emit(f"  !! unknown step type: {s.type}")
        return False

    # ---- common click performer ----
    def _perform_click(self, x: int, y: int, step: StepData, btn: str = "left"):
        jx = np.random.randint(-step.jitter, step.jitter + 1) if step.jitter > 0 else 0
        jy = np.random.randint(-step.jitter, step.jitter + 1) if step.jitter > 0 else 0
        px, py = int(x + jx), int(y + jy)
        pyautogui.moveTo(px, py)
        time.sleep(max(0, int(step.pre_move_sleep_ms)) / 1000.0)
        pyautogui.mouseDown(button=btn)
        time.sleep(max(0, int(step.press_duration_ms)) / 1000.0)
        pyautogui.mouseUp(button=btn)
        time.sleep(max(0, int(step.post_click_sleep_ms)) / 1000.0)

    # ---- executors ----
    def _image_click(self, sct, mon, step: StepData, idx: int) -> bool:
        tpl = step.ensure_tpl()
        if tpl is None:
            self.log.emit("  !! No template")
            return False
        # 일관성: 로그/판정 모두 min_confidence 사용
        th = float(getattr(step, "min_confidence", step.threshold))
        h, w = tpl.shape[:2]

        # Prepare capture region (ROI vs full virtual desktop)
        use_roi = bool(step.search_roi_enabled and step.search_roi_width > 0 and step.search_roi_height > 0)
        if use_roi:
            # clamp ROI to virtual desktop bounds
            L = int(max(mon["left"], step.search_roi_left))
            T = int(max(mon["top"], step.search_roi_top))
            R = int(min(mon["left"] + mon["width"], step.search_roi_left + step.search_roi_width))
            B = int(min(mon["top"] + mon["height"], step.search_roi_top + step.search_roi_height))
            W = max(0, R - L); H = max(0, B - T)
            if W <= 0 or H <= 0:
                self.log.emit("  !! ROI invalid or out of bounds; abort")
                return False
            if W < w or H < h:
                self.log.emit(f"  !! ROI smaller than template ({W}x{H} < {w}x{h})")
                return False
            region = {"left": L, "top": T, "width": W, "height": H}
            self.log.emit(f"  ROI: enabled L{L},T{T} {W}x{H}")
        else:
            self.log.emit("  ROI: disabled (full virtual desktop)")
            region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}

        t0 = time.time()
        consec = 0
        last_ok_ts = 0.0
        max_age_ms = max(2 * int(getattr(step, "poll_ms", 100)), 120)  # 연속 판정 허용 시간창
        need = max(1, int(getattr(step, "detect_consecutive", 1)))
        while (time.time() - t0) * 1000 <= int(step.timeout_ms):
            if self._stop:
                return False
            raw = sct.grab(region)  # BGRA
            view = np.frombuffer(raw.rgb, dtype=np.uint8)  # 이미 RGB 순서(내부 변환), 복사 1회
            frame = view.reshape(raw.height, raw.width, 3).copy()  # 안전하게 소유권 확보
            mr = self._matcher.find_best(frame, step)
            if mr.ok and mr.score >= float(getattr(step, "min_confidence", step.threshold)):
                now_ts = time.time() * 1000.0
                if consec == 0 or (now_ts - last_ok_ts) <= max_age_ms:
                    consec += 1
                else:
                    # 연속성 끊김 → 1부터 다시
                    consec = 1
                last_ok_ts = now_ts

                cx = region["left"] + int(mr.x)
                cy = region["top"] + int(mr.y)
                # --- anchor & offset 기존 코드 그대로 ---
                if getattr(mr, "w", 0) and getattr(mr, "h", 0):
                    halfw, halfh = int(mr.w/2), int(mr.h/2)
                    anchor = getattr(step, "click_anchor", "center") or "center"
                    if anchor == "topleft":        cx -= halfw; cy -= halfh
                    elif anchor == "topright":     cx += halfw; cy -= halfh
                    elif anchor == "bottomleft":   cx -= halfw; cy += halfh
                    elif anchor == "bottomright":  cx += halfw; cy += halfh
                cx += int(getattr(step, "click_offset_x", 0))
                cy += int(getattr(step, "click_offset_y", 0))

                if consec < need:
                    self.log.emit(f"  -> match {mr.score:.3f} at ({cx},{cy})  [{consec}/{need}]")
                    self.msleep(int(step.poll_ms))
                    continue

                self.log.emit(f"  -> match {mr.score:.3f} at ({cx},{cy})  [OK]")
                step._last_match_xy = (int(cx), int(cy))
                if not self.dry_run:
                    self._perform_click(cx, cy, step, btn=(step.click_button or "left"))
                else:
                    self.requestCrosshair.emit(cx, cy, 200)
                # hold_until_next 기존 처리 유지
                if step.hold_until_next:
                    next_img = self._peek_next_image_step(idx + 1)
                    if next_img is not None:
                        ok_hold = self._hold_until_next_image(sct, mon, step, next_img)
                        if not ok_hold:
                            self.log.emit("  !! hold_until_next timeout or stopped")
                            return False
                return True
            self.msleep(int(step.poll_ms))


    def _peek_next_image_step(self, start_index: int) -> Optional[StepData]:
        for j in range(start_index, len(self.steps)):
            s = self.steps[j]
            if s.type == "image_click":
                return s
        return None

    def _hold_until_next_image(self, sct, mon, stepA: StepData, stepB: StepData) -> bool:
        tplB = stepB.ensure_tpl()
        if tplB is None:
            self.log.emit("  !! next image has no template")
            return False

        # regions for A (reclick) and B (detection)
        def _region_for(step: StepData):
            if step.search_roi_enabled and step.search_roi_width > 0 and step.search_roi_height > 0:
                L = int(max(mon["left"], step.search_roi_left))
                T = int(max(mon["top"], step.search_roi_top))
                R = int(min(mon["left"] + mon["width"], step.search_roi_left + step.search_roi_width))
                B = int(min(mon["top"] + mon["height"], step.search_roi_top + step.search_roi_height))
                W = max(0, R - L); H = max(0, B - T)
                if W <= 0 or H <= 0:
                    return None
                return {"left": L, "top": T, "width": W, "height": H}
            return {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}

        regA = _region_for(stepA)
        regB = _region_for(stepB)
        if regA is None or regB is None:
            self.log.emit("  !! ROI invalid in hold_until_next_image");
            return False

        t0 = time.time()
        last_click_ts = time.time()
        consecutive = 0
        while (time.time() - t0) * 1000 <= int(stepA.hold_timeout_ms):
            if self._stop:
                return False
            frameB = np.array(sct.grab(regB), dtype=np.uint8)[:, :, :3].copy()
            mrB = self._matcher.find_best(frameB, stepB)
            maxB = mrB.score if mrB and mrB.ok else 0.0
            if maxB >= float(getattr(stepB, "min_confidence", stepB.threshold)):
                consecutive += 1
                if consecutive >= max(1, int(stepA.hold_release_consecutive or 1)):
                    self.log.emit("  -> next image detected; releasing hold")
                    return True
            else:
                consecutive = 0

            if (time.time() - last_click_ts) * 1000 >= int(stepA.hold_reclick_interval_ms):
                ax, ay = None, None
                if stepA.hold_reacquire_each_time:
                    tplA = stepA.ensure_tpl()
                    frameA = np.array(sct.grab(regA), dtype=np.uint8)[:, :, :3].copy()
                    resA = cv2.matchTemplate(frameA, tplA, cv2.TM_CCOEFF_NORMED)
                    _, maxA, _, locA = cv2.minMaxLoc(resA)
                    if maxA >= float(stepA.threshold):
                        ax = regA["left"] + locA[0] + tplA.shape[1] // 2
                        ay = regA["top"] + locA[1] + tplA.shape[0] // 2
                        stepA._last_match_xy = (int(ax), int(ay))
                else:
                    if stepA._last_match_xy:
                        ax, ay = stepA._last_match_xy
                if ax is not None and ay is not None:
                    if not self.dry_run:
                        self._perform_click(int(ax), int(ay), stepA, btn=(stepA.click_button or "left"))
                    else:
                        self.requestCrosshair.emit(int(ax), int(ay), 200)
                    last_click_ts = time.time()
            self.msleep(int(stepA.poll_ms))
        return False

    def _key(self, s: StepData) -> bool:
        ks = (s.key_string or "").strip()
        if not ks:
            self.log.emit("  !! empty key")
            return False
        if self.dry_run:
            return True
        try:
            for _ in range(max(1, int(s.key_times))):
                if "+" in ks:
                    parts = [p.strip() for p in ks.split("+") if p.strip()]
                    pyautogui.hotkey(*parts)
                elif len(ks) == 1 or ks.lower() in pyautogui.KEYBOARD_KEYS:
                    pyautogui.press(ks)
                else:
                    # 일반 텍스트 입력
                    pyautogui.typewrite(ks)
            return True
        except Exception as e:
            self.log.emit(f"  !! key err: {e}")
            return False

    def _key_down(self, s: StepData) -> bool:
        k = (s.key_string or "").strip()
        if not k:
            self.log.emit("  !! key_down empty")
            return False
        if self.dry_run:
            return True
        try:
            pyautogui.keyDown(k)
            return True
        except Exception as e:
            self.log.emit(f"  !! keyDown err: {e}")
            return False

    def _key_up(self, s: StepData) -> bool:
        k = (s.key_string or "").strip()
        if not k:
            self.log.emit("  !! key_up empty")
            return False
        if self.dry_run:
            return True
        try:
            pyautogui.keyUp(k)
            return True
        except Exception as e:
            self.log.emit(f"  !! keyUp err: {e}")
            return False

    def _key_hold(self, s: StepData) -> bool:
        k = (s.key_string or "").strip()
        if not k:
            self.log.emit("  !! key_hold empty"); return False
        if self.dry_run:
            return True
        try:
            pyautogui.keyDown(k)
            self.msleep(max(0, int(s.hold_ms)))
            pyautogui.keyUp(k)
            return True
        except Exception as e:
            self.log.emit(f"  !! keyHold err: {e}")
            return False

    def _click_point(self, s: StepData) -> bool:
        if s.click_x is None or s.click_y is None:
            self.log.emit("  !! click_point has no coords"); return False
        if self.dry_run:
            self.requestCrosshair.emit(int(s.click_x), int(s.click_y), 200); return True
        try:
            self._perform_click(int(s.click_x), int(s.click_y), s, btn=s.click_button or "left")
            return True
        except Exception as e:
            self.log.emit(f"  !! click_point err: {e}")
            return False

    def _drag(self, s: StepData) -> bool:
        if None in (s.drag_from_x, s.drag_from_y, s.drag_to_x, s.drag_to_y):
            self.log.emit("  !! drag missing coords"); return False
        if self.dry_run:
            self.requestCrosshair.emit(int(s.drag_from_x), int(s.drag_from_y), 200)
            self.requestCrosshair.emit(int(s.drag_to_x), int(s.drag_to_y), 200)
            return True
        try:
            pyautogui.moveTo(int(s.drag_from_x), int(s.drag_from_y))
            pyautogui.mouseDown()
            self.msleep(max(1, int(s.drag_duration_ms)))
            pyautogui.moveTo(int(s.drag_to_x), int(s.drag_to_y))
            pyautogui.mouseUp()
            return True
        except Exception as e:
            self.log.emit(f"  !! drag err: {e}")
            return False

    def _scroll(self, s: StepData) -> bool:
        try:
            for _ in range(max(1, int(s.scroll_times))):
                pyautogui.hscroll(int(s.scroll_dx))
                pyautogui.scroll(int(s.scroll_dy))
                self.msleep(max(0, int(s.scroll_interval_ms)))
            return True
        except Exception as e:
            self.log.emit(f"  !! scroll err: {e}")
            return False

# ---------------- Not-Image Dialog ----------------
class NotImageDialog(QDialog):
    def __init__(self, step: Optional[StepData], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Not-Image Step")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModal)

        form = QFormLayout(self)
        
        # 기본 필드
        self.edType = QLineEdit(step.type if step else "key")
        self.edName = QLineEdit(step.name if step else "")
        self.edKey = QLineEdit(step.key_string or "" if step else "")
        self.spTimes = QSpinBox(); self.spTimes.setRange(1, 99); self.spTimes.setValue(step.key_times if step else 1)
        self.spHold = QSpinBox();  self.spHold.setRange(0, 10000); self.spHold.setValue(step.hold_ms if step else 0)

        # 스크롤/드래그 목적지(또는 벡터)
        self.spDx = QSpinBox(); self.spDx.setRange(-9999, 9999); self.spDx.setValue(step.scroll_dx if step else 0)
        self.spDy = QSpinBox(); self.spDy.setRange(-9999, 9999); self.spDy.setValue(step.scroll_dy if step else 0)
        self.spST = QSpinBox(); self.spST.setRange(1, 99);       self.spST.setValue(step.scroll_times if step else 1)
        self.spSI = QSpinBox(); self.spSI.setRange(0, 10000);    self.spSI.setValue(step.scroll_interval_ms if step else 0)

        # 클릭/드래그 시작 좌표
        self.spClickX = QSpinBox(); self.spClickX.setRange(-9999, 9999); self.spClickX.setValue(step.click_x if step and step.click_x is not None else 0)
        self.spClickY = QSpinBox(); self.spClickY.setRange(-9999, 9999); self.spClickY.setValue(step.click_y if step and step.click_y is not None else 0)
        self.edBtn     = QLineEdit(step.click_button if step else "left")

        # 폼 배치
        form.addRow("Type", self.edType)
        form.addRow("Name", self.edName)
        form.addRow("Key String", self.edKey)
        form.addRow("Key Times", self.spTimes)
        form.addRow("Hold ms", self.spHold)

        # 클릭 좌표 / 드래그 시작 좌표 + 픽커 버튼들
        from PyQt5.QtWidgets import QHBoxLayout, QPushButton
        rowClick = QHBoxLayout()
        rowClick.addWidget(self.spClickX)
        rowClick.addWidget(self.spClickY)
        self.btnPickClick = QPushButton("클릭 좌표 선택")
        self.btnPickDragFrom = QPushButton("드래그 시작 선택")
        rowClick.addWidget(self.btnPickClick)
        rowClick.addWidget(self.btnPickDragFrom)
        form.addRow("Click x / y (또는 Drag 시작)", rowClick)

        # 드래그 끝 좌표(또는 스크롤 dx/dy) + 픽커 버튼
        rowDest = QHBoxLayout()
        rowDest.addWidget(self.spDx)
        rowDest.addWidget(self.spDy)
        self.btnPickDragTo = QPushButton("드래그 끝 선택")
        rowDest.addWidget(self.btnPickDragTo)
        form.addRow("Drag 끝 x / y (또는 Scroll dx / dy)", rowDest)

        # 나머지 스크롤/버튼 설정
        form.addRow("Scroll times / interval", self.spST); form.addRow("", self.spSI)
        form.addRow("Button", self.edBtn)

        # OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        form.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # 좌표 픽커 이벤트 연결 (그대로 복붙)
        self.btnPickClick.clicked.connect(
            lambda _=False, b=self.btnPickClick: self._pick_point_into(self.spClickX, self.spClickY, "클릭 위치", b)
        )
        self.btnPickDragFrom.clicked.connect(
            lambda _=False, b=self.btnPickDragFrom: self._pick_point_into(self.spClickX, self.spClickY, "드래그 시작점", b)
        )
        self.btnPickDragTo.clicked.connect(
            lambda _=False, b=self.btnPickDragTo: self._pick_point_into(self.spDx, self.spDy, "드래그 끝점", b)
        )

    def _robust_restore_self(self):
        """
        좌표 픽커 이후 다이얼로그가 숨김/최소화/비활성으로 남았을 때
        확실히 화면 전면/활성으로 복귀시키는 루틴.
        """
        try:
            # 1) 최소화 풀기 + 보이기
            st = self.windowState()
            if st & Qt.WindowMinimized:
                self.setWindowState(st & ~Qt.WindowMinimized)
            if not self.isVisible():
                # hide 상태였으면 보이게
                try:
                    self.showNormal()
                except Exception:
                    self.show()

            # 2) 불투명도/Enable 복구
            try:
                if self.windowOpacity() < 0.99:
                    self.setWindowOpacity(1.0)
            except Exception:
                pass
            try:
                self.setEnabled(True)
            except Exception:
                pass

            # 3) 최전면 끌어올리기
            # - 일단 show() 후 raise_ → activateWindow 순으로
            self.show()
            self.raise_()
            self.activateWindow()
            # 활성 윈도 지정 (플랫폼별로 필요한 경우가 있음)
            try:
                QApplication.setActiveWindow(self)
            except Exception:
                pass

            # 4) 모달/포커스 끌어오기 (WindowModal 유지 가정)
            try:
                self.setWindowModality(Qt.WindowModal)
            except Exception:
                pass

            # 5) 이벤트 한 틱 밀어 UI 반영
            try:
                QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception:
                pass

            # 6) 일부 플랫폼에서 즉시 raise가 무시되는 경우가 있어 딜레이 재-승격
            try:
                QTimer.singleShot(0, self.raise_)
                QTimer.singleShot(0, self.activateWindow)
            except Exception:
                pass
        except Exception:
            pass


     # ✔ OK/Cancel 모두 이 함수로 정리
    def _finalize_close(self):
        # 1) 커서/픽킹 상태/버튼 비활성 잔여 모두 복구
        try: QApplication.restoreOverrideCursor()
        except Exception: pass
        try:
            if getattr(self, "_picking", False):
                self._picking = False
        except Exception:
            pass
        # 혹시 disable 해둔 경우 복원
        try: self.setEnabled(True)
        except Exception: pass

        # 2) 부모 포커스/활성 회복
        p = self.parent()
        if p is not None:
            try:
                p.setEnabled(True)
                p.raise_(); p.activateWindow(); p.setFocus(Qt.ActiveWindowFocusReason)
            except Exception:
                pass

        # 3) 이벤트 큐 한 번 비워 중첩루프 후유증 제거
        try: QApplication.processEvents(QEventLoop.AllEvents, 50)
        except Exception: pass
        try:
            self._robust_restore_self()  # 혹시 남은 비활성/숨김 상태를 한 번 더 복구
        except Exception:
            pass

    # ✔ OK
    def accept(self):
        self._finalize_close()
        super().accept()

    # ✔ Cancel
    def reject(self):
        self._finalize_close()
        super().reject()
        
    # === DROP-IN REPLACEMENT: 클릭 좌표 선택(오버레이 보장) ===
    def _pick_point_into(self, sp_x, sp_y, label: str, source_btn=None):
        """
        안전한 좌표 픽커:
        - 재진입 방지(self._picking)
        - 픽킹 동안 다이얼로그를 숨기고 비활성화하여 오버레이 간섭 제거
        - 관련 버튼들 일괄 비활성화 → finally에서 복구
        - 선택 취소/실패/예외 전부에서 상태 원복
        """
        # 재진입 가드
        if getattr(self, "_picking", False):
            return
        self._picking = True

        app = QApplication.instance()
        prev_active = QApplication.activeWindow()

        # 비활성화 대상 버튼 모음
        buttons = [x for x in (getattr(self, "btnPickClick", None),
                               getattr(self, "btnPickDragFrom", None),
                               getattr(self, "btnPickDragTo", None)) if x is not None]

        # 현재 보임 상태/활성 상태 기억
        was_visible = self.isVisible()

        try:
            # 모든 버튼/다이얼로그 비활성화
            for bt in buttons:
                try: bt.setEnabled(False)
                except Exception: pass
            try:
                if source_btn is not None:
                    source_btn.setEnabled(False)
            except Exception:
                pass
            try:
                self.setEnabled(False)
            except Exception:
                pass

            # 커서 변경(UX)
            try: QApplication.setOverrideCursor(Qt.CrossCursor)
            except Exception: pass

            # 다이얼로그 임시 숨김(최전면 오버레이 확보)
            if was_visible:
                try: self.hide()
                except Exception: pass

            # 전역 플래그
            if app is not None:
                try: setattr(app, "isPickingPoint", True)
                except Exception: pass

            # 이벤트 큐 비우기(포커스/표시 상태 정리)
            try: QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception: pass

            # 좌표 픽커 실행 (parent=None 권장)
            res = _safe_select_point(None)
            norm = _normalize_point_result(res)
            if norm is None:
                info("Point selection canceled or failed.")
                return

            x, y = norm

            # 시그널 루프 방지하며 값 반영
            try:
                if hasattr(sp_x, "blockSignals"): sp_x.blockSignals(True)
                if hasattr(sp_y, "blockSignals"): sp_y.blockSignals(True)
                sp_x.setValue(int(x))
                sp_y.setValue(int(y))
            finally:
                if hasattr(sp_x, "blockSignals"): sp_x.blockSignals(False)
                if hasattr(sp_y, "blockSignals"): sp_y.blockSignals(False)

            info(f"{label}: ({x},{y})")

        finally:
            # 커서 원복
            try: QApplication.restoreOverrideCursor()
            except Exception: pass

            # 다이얼로그 복귀 (보여주기 → 활성/포커스 복구)
            try:
                if was_visible:
                    self.show()
                    self.raise_()
                    self.activateWindow()
            except Exception:
                pass

            # 메인 이벤트 큐 한 번 더 정리 (중첩 루프 후유증 방지)
            try: QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception: pass

            # 다이얼로그 재활성화
            try: self.setEnabled(True)
            except Exception: pass

            # 값 반영한 위젯에 포커스 주면 직관적
            try:
                sp_x.setFocus()
                if hasattr(sp_x, "selectAll"):
                    sp_x.selectAll()
            except Exception:
                pass
            try:
                self._robust_restore_self()   # ★ 강제 복귀
            except Exception:
                pass

            # 버튼 원복
            for bt in buttons:
                try: bt.setEnabled(True)
                except Exception: pass
            try:
                if source_btn is not None:
                    source_btn.setEnabled(True)
            except Exception:
                pass

            # 전역 플래그 해제
            if app is not None:
                try: setattr(app, "isPickingPoint", False)
                except Exception: pass

            # 가능하면 이전 활성 창 포커스도 복구 (편집 다이얼로그/메인윈도우 포함)
            try:
                if prev_active is not None:
                    prev_active.setEnabled(True)
                    prev_active.raise_()
                    prev_active.activateWindow()
                    prev_active.setFocus(Qt.ActiveWindowFocusReason)
            except Exception:
                pass

            self._picking = False
        
    # === PATCH: ROI 픽커 완성본 ===
    def _pick_roi_into(self, target_widget, label: str):
        """
        ROI 선택기를 띄워 선택된 사각형을 target_widget에 반영합니다.
        - target_widget 이 QLineEdit 류면: "L,T WxH" 형태 텍스트로 세팅
        - target_widget 이 (spL, spT, spW, spH) 튜플/리스트면: 각 스핀박스에 정수값 세팅
        - 그 외: 로그만 남김
        """
        # 재진입 가드
        if getattr(self, "_picking", False):
            return
        self._picking = True  # ADDED: reentrancy guard

        app = QApplication.instance()
        prev_active = QApplication.activeWindow()
        was_visible = self.isVisible()

        # 관련 버튼들 비활성화 (있을 때만)
        buttons = [x for x in (getattr(self, "btnPickClick", None),
                               getattr(self, "btnPickDragFrom", None),
                               getattr(self, "btnPickDragTo", None),
                               getattr(self, "btnPickROI", None)) if x is not None]
        try:
            for bt in buttons:
                try: bt.setEnabled(False)
                except Exception: pass
            try: self.setEnabled(False)
            except Exception: pass

            # 커서 변경 + 다이얼로그 숨김(오버레이 확보)
            try: QApplication.setOverrideCursor(Qt.CrossCursor)
            except Exception: pass
            if was_visible:
                try: self.hide()
                except Exception: pass

            # 이벤트 큐 정리
            try: QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception: pass

            # 전역 플래그 (선택 중)
            if app is not None:
                try: setattr(app, "isPickingROI", True)  # ADDED
                except Exception: pass

            # === 실제 ROI 선택 ===
            try:
                rect, crop, virt = ROISelector.select_from_screen(None)  # parent=None로 충돌 최소화  # ADDED
            except Exception as e:
                err(f"ROISelector error: {e}")  # ADDED
                return

            if rect is None or rect.isNull():
                info("ROI selection canceled.")  # ADDED
                return

            # 좌표/크기 정수화
            L = int(rect.left());  T = int(rect.top())
            W = int(rect.width()); H = int(rect.height())

            # 가상 데스크톱 경계로 clamp (별도 헬퍼 없이 인라인 처리)
            try:
                vL, vT, vW, vH = virt
                maxR = int(vL + vW); maxB = int(vT + vH)
                L = max(int(vL), min(L, maxR - 1))
                T = max(int(vT), min(T, maxB - 1))
                W = max(1, min(W, maxR - L))
                H = max(1, min(H, maxB - T))
            except Exception:
                # virt가 비정상이어도 안전하게 진행
                pass

            # === target_widget에 반영 ===
            def _set_spin(sp, val):
                try:
                    if hasattr(sp, "blockSignals"): sp.blockSignals(True)
                    sp.setValue(int(val))
                finally:
                    try:
                        if hasattr(sp, "blockSignals"): sp.blockSignals(False)
                    except Exception:
                        pass

            try:
                # (1) 4스핀박스 (L,T,W,H) 형태
                if isinstance(target_widget, (tuple, list)) and len(target_widget) >= 4:
                    spL, spT, spW, spH = target_widget[:4]
                    _set_spin(spL, L); _set_spin(spT, T)
                    _set_spin(spW, W); _set_spin(spH, H)
                    # UX: 첫 필드 선택
                    try:
                        if hasattr(spL, "setFocus"): spL.setFocus()
                        if hasattr(spL, "selectAll"): spL.selectAll()
                    except Exception: pass

                # (2) 단일 라인에 텍스트로
                elif hasattr(target_widget, "setText"):
                    try:
                        target_widget.setText(f"{L},{T} {W}x{H}")
                        if hasattr(target_widget, "setCursorPosition"):
                            target_widget.setCursorPosition(len(target_widget.text()))
                    except Exception as e:
                        warn(f"{label}: setText failed: {e}; ROI=({L},{T},{W},{H})")

                # (3) 그 외: 로그만
                else:
                    info(f"{label}: ROI=({L},{T},{W},{H})")

            finally:
                # ROI 반영 후 다이얼로그 복귀
                try:
                    self._robust_restore_self()   # ADDED: 확실한 복귀
                except Exception:
                    # 최소 복구 루틴
                    try:
                        if was_visible:
                            self.show()
                            self.raise_(); self.activateWindow()
                    except Exception:
                        pass

        finally:
            # 버튼/커서/전역 플래그 복구
            for bt in buttons:
                try: bt.setEnabled(True)
                except Exception: pass
            try: self.setEnabled(True)
            except Exception: pass
            try: QApplication.restoreOverrideCursor()
            except Exception: pass
            if app is not None:
                try: setattr(app, "isPickingROI", False)  # ADDED
                except Exception: pass
            # 가능하면 이전 활성 창 포커스 복구
            try:
                if prev_active is not None:
                    prev_active.setEnabled(True)
                    prev_active.raise_()
                    prev_active.activateWindow()
            except Exception:
                pass

            self._picking = False  # ADDED: release reentrancy guard

    # ---- 결과 구성 ----
    def result_step(self) -> Optional[StepData]:
        if self.result() != QDialog.Accepted: 
            return None
        t = self.edType.text().strip()
        name = self.edName.text().strip() or t
        if t == "key":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="key",
                            key_string=self.edKey.text().strip(), key_times=self.spTimes.value())
        if t == "key_down":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="key_down",
                            key_string=self.edKey.text().strip())
        if t == "key_up":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="key_up",
                            key_string=self.edKey.text().strip())
        if t == "key_hold":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="key_hold",
                            key_string=self.edKey.text().strip(), hold_ms=self.spHold.value())
        if t == "click_point":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="click_point",
                            click_button=self.edBtn.text().strip() or "left",
                            click_x=self.spClickX.value(), click_y=self.spClickY.value())
        if t == "drag":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="drag",
                            drag_from_x=self.spClickX.value(), drag_from_y=self.spClickY.value(),
                            drag_to_x=self.spDx.value(), drag_to_y=self.spDy.value(),
                            drag_duration_ms=max(1, self.spSI.value()))
        if t == "scroll":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="scroll",
                            scroll_dx=self.spDx.value(), scroll_dy=self.spDy.value(),
                            scroll_times=self.spST.value(), scroll_interval_ms=self.spSI.value())
        return None


# ---------------- Image Step Dialog (ROI editor) ----------------
class ImageStepDialog(QDialog):
    def __init__(self, step: StepData, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Step")
        self._step = step
        self._tpl = step.ensure_tpl()
        self._roi_enabled = bool(step.search_roi_enabled and step.search_roi_width > 0 and step.search_roi_height > 0)
        self._roi = (int(step.search_roi_left), int(step.search_roi_top),
                       int(step.search_roi_width), int(step.search_roi_height)) if self._roi_enabled else None
        # ✅ 종료/모달 설정 추가
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setModal(False)
        self.setWindowModality(Qt.WindowModal)


        form = QFormLayout(self)  # NOTE: setLayout(form) 호출 불필요 (중복 경고 방지)

        # Template preview
        self.lblPreview = QLabel()
        if self._tpl is not None:
            self.lblPreview.setPixmap(cvimg_to_qpixmap(self._tpl).scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        form.addRow("Template", self.lblPreview)

        # ROI controls
        self.chkROI = QCheckBox("Limit search to region")
        self.chkROI.setChecked(self._roi_enabled)
        form.addRow(self.chkROI)

        self.lblROI = QLabel(self._roi_text())
        form.addRow("ROI", self.lblROI)

        row = QHBoxLayout()
        self.btnPick = QPushButton("Pick Region…")
        self.btnClear = QPushButton("Clear ROI")
        self.btnChangeTpl = QPushButton("Recapture Template…")
        row.addWidget(self.btnPick); row.addWidget(self.btnClear); row.addWidget(self.btnChangeTpl)
        form.addRow(row)

        # --- Matching (Advanced) ---
        boxMatch = QGroupBox("Matching (Advanced)")
        fm = QFormLayout(boxMatch)
        # Preprocessing
        self.chkGray = QCheckBox("Use grayscale"); self.chkGray.setChecked(step.pre_gray)
        self.spBlur = QSpinBox(); self.spBlur.setRange(1, 15); self.spBlur.setSingleStep(2); self.spBlur.setValue(step.pre_blur_ksize)
        self.chkCLAHE = QCheckBox("CLAHE"); self.chkCLAHE.setChecked(step.pre_clahe)
        self.chkEdge = QCheckBox("Edge emphasis"); self.chkEdge.setChecked(step.pre_edge)
        self.chkSharp = QCheckBox("Sharpen"); self.chkSharp.setChecked(step.pre_sharpen)
        fm.addRow(self.chkGray)
        fm.addRow("Gaussian blur ksize (odd)", self.spBlur)
        fm.addRow(self.chkCLAHE); fm.addRow(self.chkEdge); fm.addRow(self.chkSharp)
        # Multiscale
        self.chkMS = QCheckBox("Enable Multi-Scale"); self.chkMS.setChecked(step.ms_enable)
        self.dMSmin = QDoubleSpinBox(); self.dMSmin.setRange(0.30, 2.00); self.dMSmin.setSingleStep(0.01); self.dMSmin.setValue(step.ms_min_scale)
        self.dMSmax = QDoubleSpinBox(); self.dMSmax.setRange(0.50, 3.00); self.dMSmax.setSingleStep(0.01); self.dMSmax.setValue(step.ms_max_scale)
        self.dMSstep = QDoubleSpinBox(); self.dMSstep.setRange(1.01, 1.50); self.dMSstep.setSingleStep(0.01); self.dMSstep.setValue(step.ms_step)
        fm.addRow(self.chkMS); fm.addRow("Min scale", self.dMSmin); fm.addRow("Max scale", self.dMSmax); fm.addRow("Scale step", self.dMSstep)
        # Rotation
        self.chkRot = QCheckBox("Enable Rotation"); self.chkRot.setChecked(step.rot_enable)
        self.dRmin = QDoubleSpinBox(); self.dRmin.setRange(-90.0, 0.0); self.dRmin.setSingleStep(1.0); self.dRmin.setValue(step.rot_min_deg)
        self.dRmax = QDoubleSpinBox(); self.dRmax.setRange(0.0, 90.0); self.dRmax.setSingleStep(1.0); self.dRmax.setValue(step.rot_max_deg)
        self.dRstep = QDoubleSpinBox(); self.dRstep.setRange(1.0, 30.0); self.dRstep.setSingleStep(1.0); self.dRstep.setValue(step.rot_step_deg)
        fm.addRow(self.chkRot); fm.addRow("Min deg", self.dRmin); fm.addRow("Max deg", self.dRmax); fm.addRow("Step deg", self.dRstep)
        # Confidence / Stability
        self.dMinConf = QDoubleSpinBox(); self.dMinConf.setRange(0.50, 0.99); self.dMinConf.setSingleStep(0.01); self.dMinConf.setValue(step.min_confidence)
        self.spConsec = QSpinBox(); self.spConsec.setRange(1, 5); self.spConsec.setValue(step.detect_consecutive)
        fm.addRow("Min confidence", self.dMinConf)
        fm.addRow("Detect consecutive", self.spConsec)
        # Feature fallback
        self.chkFeat = QCheckBox("Feature fallback (ORB)"); self.chkFeat.setChecked(step.feat_fallback_enable)
        self.spFeatN = QSpinBox(); self.spFeatN.setRange(100, 2000); self.spFeatN.setValue(step.feat_nfeatures)
        self.dFeatRatio = QDoubleSpinBox(); self.dFeatRatio.setRange(0.5, 0.95); self.dFeatRatio.setSingleStep(0.01); self.dFeatRatio.setValue(step.feat_match_ratio)
        self.dFeatThr = QDoubleSpinBox(); self.dFeatThr.setRange(0.5, 10.0); self.dFeatThr.setSingleStep(0.1); self.dFeatThr.setValue(step.feat_ransac_reproj_thresh)
        fm.addRow(self.chkFeat); fm.addRow("ORB nfeatures", self.spFeatN); fm.addRow("Lowe ratio", self.dFeatRatio); fm.addRow("RANSAC thr", self.dFeatThr)
        form.addRow(boxMatch)

        # Test
        self.btnTest = QPushButton("Test on current screen")
        form.addRow(self.btnTest)
        self.btnTest.clicked.connect(self._on_test_match)
        self.lblTest = QLabel("")
        form.addRow(self.lblTest)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        form.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        self.btnPick.clicked.connect(self._on_pick_roi)
        self.btnClear.clicked.connect(self._on_clear_roi)
        self.btnChangeTpl.clicked.connect(self._on_change_tpl)

    def _robust_restore_self(self):
        """
        좌표 픽커 이후 다이얼로그가 숨김/최소화/비활성으로 남았을 때
        확실히 화면 전면/활성으로 복귀시키는 루틴.
        """
        try:
            # 1) 최소화 풀기 + 보이기
            st = self.windowState()
            if st & Qt.WindowMinimized:
                self.setWindowState(st & ~Qt.WindowMinimized)
            if not self.isVisible():
                # hide 상태였으면 보이게
                try:
                    self.showNormal()
                except Exception:
                    self.show()

            # 2) 불투명도/Enable 복구
            try:
                if self.windowOpacity() < 0.99:
                    self.setWindowOpacity(1.0)
            except Exception:
                pass
            try:
                self.setEnabled(True)
            except Exception:
                pass

            # 3) 최전면 끌어올리기
            # - 일단 show() 후 raise_ → activateWindow 순으로
            self.show()
            self.raise_()
            self.activateWindow()
            # 활성 윈도 지정 (플랫폼별로 필요한 경우가 있음)
            try:
                QApplication.setActiveWindow(self)
            except Exception:
                pass

            # 4) 모달/포커스 끌어오기 (WindowModal 유지 가정)
            try:
                self.setWindowModality(Qt.WindowModal)
            except Exception:
                pass

            # 5) 이벤트 한 틱 밀어 UI 반영
            try:
                QApplication.processEvents(QEventLoop.AllEvents, 50)
            except Exception:
                pass

            # 6) 일부 플랫폼에서 즉시 raise가 무시되는 경우가 있어 딜레이 재-승격
            try:
                QTimer.singleShot(0, self.raise_)
                QTimer.singleShot(0, self.activateWindow)
            except Exception:
                pass
        except Exception:
            pass


    def _finalize_close(self):
        try: QApplication.restoreOverrideCursor()
        except Exception: pass
        try: self.setEnabled(True)
        except Exception: pass
        p = self.parent()
        if p is not None:
            try:
                p.setEnabled(True)
                p.raise_(); p.activateWindow(); p.setFocus(Qt.ActiveWindowFocusReason)
            except Exception: pass
        try: QApplication.processEvents(QEventLoop.AllEvents, 50)
        except Exception: pass
        try:
            self._robust_restore_self()  # 혹시 남은 비활성/숨김 상태를 한 번 더 복구
        except Exception:
            pass

    def accept(self):
        self._finalize_close()
        super().accept()

    def reject(self):
        self._finalize_close()
        super().reject()

    def _roi_text(self):
        if self._roi_enabled and self._roi:
            L,T,W,H = self._roi
            return f"{L},{T} {W}x{H}"
        return "(full screen)"

    def _on_pick_roi(self):
        rect, _, virt = ROISelector.select_from_screen(self)
        if rect is None or rect.isNull():
            return
        virt_left, virt_top, virt_w, virt_h = virt
        L = virt_left + rect.x()
        T = virt_top + rect.y()
        W = rect.width(); H = rect.height()
        self._roi = (int(L), int(T), int(W), int(H))
        self._roi_enabled = True
        self.lblROI.setText(self._roi_text())
        self.chkROI.setChecked(True)

    def _on_clear_roi(self):
        self._roi = None
        self._roi_enabled = False
        self.lblROI.setText(self._roi_text())
        self.chkROI.setChecked(False)

    def _on_change_tpl(self):
        rect, crop, _ = ROISelector.select_from_screen(self)
        if crop is None:
            return
        self._tpl = crop
        self._step.png_bytes = encode_png_bytes(crop)
        self._step._tpl_bgr = crop
        self.lblPreview.setPixmap(cvimg_to_qpixmap(self._tpl).scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def accept(self):
        self._step.search_roi_enabled = bool(self.chkROI.isChecked() and self._roi is not None)
        if self._step.search_roi_enabled:
            L,T,W,H = self._roi
            self._step.search_roi_left = int(L)
            self._step.search_roi_top = int(T)
            self._step.search_roi_width = int(W)
            self._step.search_roi_height = int(H)
        else:
            # disabled -> store zeros
            self._step.search_roi_left = int(self._step.search_roi_left or 0)
            self._step.search_roi_top = int(self._step.search_roi_top or 0)
            self._step.search_roi_width = 0
            self._step.search_roi_height = 0
        # persist matching params
        self._step.pre_gray = self.chkGray.isChecked()
        self._step.pre_blur_ksize = int(self.spBlur.value())
        self._step.pre_clahe = self.chkCLAHE.isChecked()
        self._step.pre_edge = self.chkEdge.isChecked()
        self._step.pre_sharpen = self.chkSharp.isChecked()
        self._step.ms_enable = self.chkMS.isChecked()
        self._step.ms_min_scale = float(self.dMSmin.value())
        self._step.ms_max_scale = float(self.dMSmax.value())
        self._step.ms_step = float(self.dMSstep.value())
        self._step.rot_enable = self.chkRot.isChecked()
        self._step.rot_min_deg = float(self.dRmin.value())
        self._step.rot_max_deg = float(self.dRmax.value())
        self._step.rot_step_deg = float(self.dRstep.value())
        self._step.min_confidence = float(self.dMinConf.value())
        self._step.detect_consecutive = int(self.spConsec.value())
        self._step.feat_fallback_enable = self.chkFeat.isChecked()
        self._step.feat_nfeatures = int(self.spFeatN.value())
        self._step.feat_match_ratio = float(self.dFeatRatio.value())
        self._step.feat_ransac_reproj_thresh = float(self.dFeatThr.value())

        super().accept()

    def _on_test_match(self):
        """
        Matching (Advanced) 설정값으로, 현재 화면(또는 ROI)에서 즉석 매칭 테스트.
        결과를 self.lblTest에 출력합니다. 실제 클릭은 수행하지 않습니다.
        """
        try:
            # 1) 가상 데스크톱 캡처 (mss 우선, 실패 시 QScreen 폴백)
            try:
                with mss.mss() as sct:
                    mon = sct.monitors[0]  # virtual desktop
                    frame = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3].copy()
                    mon_left, mon_top = int(mon["left"]), int(mon["top"])
            except Exception:
                screen = QApplication.primaryScreen()
                vg = screen.virtualGeometry()
                pm = screen.grabWindow(0)
                qi = pm.toImage().convertToFormat(QImage.Format_RGB888)
                w, h = qi.width(), qi.height()
                ptr = qi.bits(); ptr.setsize(qi.byteCount())
                arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                mon_left, mon_top = vg.x(), vg.y()
                mon = {"left": vg.x(), "top": vg.y(), "width": vg.width(), "height": vg.height()}

            # 2) 현재 스텝 복제 + 템플릿 캐시 공유
            fake = StepData(**asdict(self._step))
            fake._tpl_bgr = self._step._tpl_bgr  # 템플릿 캐시 공유

            # 2-b) 전처리/스케일/회전/임계/연속검증 UI값 반영
            fake.pre_gray = self.chkGray.isChecked()
            fake.pre_blur_ksize = int(self.spBlur.value())
            fake.pre_clahe = self.chkCLAHE.isChecked()
            fake.pre_edge = self.chkEdge.isChecked()
            fake.pre_sharpen = self.chkSharp.isChecked()

            fake.ms_enable = self.chkMS.isChecked()
            fake.ms_min_scale = float(self.dMSmin.value())
            fake.ms_max_scale = float(self.dMSmax.value())
            fake.ms_step = float(self.dMSstep.value())

            fake.rot_enable = self.chkRot.isChecked()
            fake.rot_min_deg = float(self.dRmin.value())
            fake.rot_max_deg = float(self.dRmax.value())
            fake.rot_step_deg = float(self.dRstep.value())

            fake.min_confidence = float(self.dMinConf.value())
            fake.detect_consecutive = int(self.spConsec.value())

            # 3) ROI 계산(있으면 ROI만 테스트)
            region = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
            if self.chkROI.isChecked() and self._roi:
                L, T, W, H = self._roi
                region = {"left": L, "top": T, "width": W, "height": H}
                fake.search_roi_enabled = True
                fake.search_roi_left = L
                fake.search_roi_top = T
                fake.search_roi_width = W
                fake.search_roi_height = H

            # ROI 잘라내기 (좌표계 보정: y는 top-모니터top, x는 left-모니터left)
            y0 = region["top"]  - mon_top
            y1 = y0 + region["height"]
            x0 = region["left"] - mon_left
            x1 = x0 + region["width"]

            # 프레임 경계로 안전 클램프
            fh, fw = frame.shape[:2]
            y0c, y1c = max(0, y0), min(fh, y1)
            x0c, x1c = max(0, x0), min(fw, x1)
            if y0c >= y1c or x0c >= x1c:
                self.lblTest.setText("Test error: ROI out of bounds")
                return

            frame_roi = frame[y0c:y1c, x0c:x1c].copy()

            # 4) 매칭 실행
            m = Matcher()
            mr = m.find_best(frame_roi, fake)
            if mr and mr.ok:
                x = region["left"] + int(mr.x)
                y = region["top"]  + int(mr.y)
                self.lblTest.setText(f"OK: conf={mr.score:.3f}, stage={mr.stage}, at=({x},{y})")
            else:
                self.lblTest.setText("No match")
        except Exception as e:
            self.lblTest.setText(f"Test error: {e}")

# ---------------- Recording Settings Dialog ----------------
class RecordingSettingsDialog(QDialog):
    def __init__(self, parent, vals):
        super().__init__(parent)
        self.setWindowTitle("레코딩 설정")  # 기존 "Recording Settings"

        (typed_gap_ms, click_merge_ms, click_radius_px,
         scroll_flush_ms, scroll_scale_dx, scroll_scale_dy) = vals

        form = QFormLayout(self)

        # 입력 묶음 간격
        self.spTypedGap = QSpinBox()
        self.spTypedGap.setRange(50, 5000)
        self.spTypedGap.setValue(int(typed_gap_ms))
        self.spTypedGap.setSuffix(" ms")
        self.spTypedGap.setToolTip("이 시간 안에 입력된 글자는 한 개의 '입력' 스텝으로 묶입니다.")

        # 클릭 합치기 시간
        self.spClickMerge = QSpinBox()
        self.spClickMerge.setRange(50, 2000)
        self.spClickMerge.setValue(int(click_merge_ms))
        self.spClickMerge.setSuffix(" ms")
        self.spClickMerge.setToolTip("더블/트리플 클릭으로 간주할 최대 시간 간격입니다.")

        # 클릭 합치기 반경
        self.spClickRadius = QSpinBox()
        self.spClickRadius.setRange(1, 50)
        self.spClickRadius.setValue(int(click_radius_px))
        self.spClickRadius.setSuffix(" px")
        self.spClickRadius.setToolTip("이 픽셀 반경 안에서 찍힌 클릭은 같은 위치로 간주합니다.")

        # 스크롤 묶음 간격
        self.spScrollFlush = QSpinBox()
        self.spScrollFlush.setRange(50, 5000)
        self.spScrollFlush.setValue(int(scroll_flush_ms))
        self.spScrollFlush.setSuffix(" ms")
        self.spScrollFlush.setToolTip("이 시간 동안의 스크롤 이벤트를 한 번에 기록합니다.")

        # 수평/수직 스크롤 크기
        self.dScrollDx = QDoubleSpinBox()
        self.dScrollDx.setRange(0.1, 1000.0)
        self.dScrollDx.setValue(float(scroll_scale_dx))
        self.dScrollDx.setDecimals(2)
        self.dScrollDx.setSuffix(" px/틱")
        self.dScrollDx.setToolTip("수평 스크롤 한 번에 보낼 양(값이 클수록 더 멀리 이동).")

        self.dScrollDy = QDoubleSpinBox()
        self.dScrollDy.setRange(0.1, 1000.0)
        self.dScrollDy.setValue(float(scroll_scale_dy))
        self.dScrollDy.setDecimals(2)
        self.dScrollDy.setSuffix(" px/틱")
        self.dScrollDy.setToolTip("수직 스크롤 한 번에 보낼 양(값이 클수록 더 멀리 이동).")

        # ----- 라벨을 한글로 -----
        form.addRow("입력 묶음 간격", self.spTypedGap)
        form.addRow("클릭 합치기 시간", self.spClickMerge)
        form.addRow("클릭 합치기 반경", self.spClickRadius)
        form.addRow("스크롤 묶음 간격", self.spScrollFlush)
        form.addRow("수평 스크롤 크기", self.dScrollDx)
        form.addRow("수직 스크롤 크기", self.dScrollDy)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        form.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def values(self):
        return (self.spTypedGap.value(), self.spClickMerge.value(), self.spClickRadius.value(),
                self.spScrollFlush.value(), self.dScrollDx.value(), self.dScrollDy.value())

# ---------------- Hotkey Dialog ----------------
class HotkeySettingsDialog(QDialog):
    def __init__(self, run_combo: Optional[str], stop_combo: Optional[str], rec_combo: Optional[str],
                 add_img_combo: Optional[str], add_notimg_combo: Optional[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        form = QFormLayout(self)

        self.edRun = QLineEdit(run_combo or "")
        self.edStop = QLineEdit(stop_combo or "")
        self.edRec = QLineEdit(rec_combo or "")
        self.edAddImg = QLineEdit(add_img_combo or "")
        self.edAddNotImg = QLineEdit(add_notimg_combo or "")

        form.addRow("Run", self.edRun)
        form.addRow("Stop", self.edStop)
        form.addRow("Record", self.edRec)
        form.addRow("Add Image Step", self.edAddImg)
        form.addRow("Add Action Step", self.edAddNotImg)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        form.addRow(btns)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        self._result = None

    def _on_ok(self):
        r  = hk_normalize(self.edRun.text())
        s  = hk_normalize(self.edStop.text())
        c  = hk_normalize(self.edRec.text())
        ai = hk_normalize(self.edAddImg.text())
        an = hk_normalize(self.edAddNotImg.text())

        used = [x for x in (r, s, c, ai, an) if x]
        if len(set(used)) != len(used):
            QMessageBox.warning(self, "Conflict", "서로 다른 동작에 같은 단축키를 지정할 수 없습니다.")
            return

        self._result = (r, s, c, ai, an)
        self.accept()

    def result_hotkeys(self):
        return self._result

# ---------------- Recorder ----------------
class InputRecorder(QObject):
    finished = pyqtSignal(list)
    pausedChanged = pyqtSignal(bool)

    def __init__(self, ignore_rect: QRect, parent=None,
                 typed_gap_ms=500,         # 텍스트 묶음 간격
                 click_merge_ms=350,       # 더블/멀티클릭 시간창
                 click_radius_px=3,        # 멀티클릭 좌표 허용 반경(px)
                 scroll_flush_ms=180,      # 스크롤 플러시 대기
                 scroll_scale_dx=30.0,     # 수평 스크롤 계수
                 scroll_scale_dy=120.0,    # 수직 스크롤 계수
                 lock_hwnd=None,           # (선택) 레코딩 잠금 창
                 ignore_combos=None):      # ★ 무시할 단축키 목록(정규화된 문자열)
        super().__init__(parent)
        self.ignore_rect = ignore_rect
        self._typed_gap_ms = typed_gap_ms
        self._click_merge_ms = click_merge_ms
        self._click_merge_px = click_radius_px
        self._scroll_flush_ms = scroll_flush_ms
        self._scroll_scale_dx = scroll_scale_dx
        self._scroll_scale_dy = scroll_scale_dy
        self._active = False
        self._paused = False
        self._mods = set()
        self._typed_buf = ""
        self._typed_last = 0.0
        self._last_click = None  # (btn, x, y, ts, count)
        self._press_pos = None   # for drag
        self._scroll_acc = (0, 0)
        self._scroll_last = 0.0
        self._ignore_until = time.time()
        self._steps: List[StepData] = []

        # ★ 무시 단축키: hk_normalize로 정규화 후 보관
        self._ignore_combos = set()
        if ignore_combos:
            for c in ignore_combos:
                c = hk_normalize(c)
                if c: self._ignore_combos.add(c)

        self._kb = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release, suppress=False)
        self._ms = mouse.Listener(on_click=self._on_click, on_move=self._on_move, on_scroll=self._on_scroll)

    def start(self):
        self._active = True
        self._kb.start(); self._ms.start()
        self._ignore_until = time.time() + 0.25

    def stop(self):
        self._active = False
        try: self._kb.stop(); self._ms.stop()
        except: pass
        self._flush_text(True); self._flush_click(True); self._flush_scroll(True)
        self.finished.emit(self._steps)

    def _now(self): return time.time()

    def _in_ignore(self, x, y) -> bool:
        r = self.ignore_rect
        return r and (r.left() <= x <= r.right()) and (r.top() <= y <= r.bottom())

    def _btn_name(self, b):
        try:
            if b == mouse.Button.left: return "left"
            if b == mouse.Button.right: return "right"
            if b == mouse.Button.middle: return "middle"
        except Exception:
            pass
        return "left"

    # ----- 키 토큰 & 조합 매칭 (MainWindow와 동일 규칙) -----
    def _token_from_key(self, k) -> Optional[str]:
        try:
            mapping = {
                keyboard.Key.enter: 'enter', keyboard.Key.esc: 'esc', keyboard.Key.space: 'space',
                keyboard.Key.tab: 'tab', keyboard.Key.backspace: 'backspace', keyboard.Key.delete: 'delete',
                keyboard.Key.home: 'home', keyboard.Key.end: 'end', keyboard.Key.insert: 'insert',
                keyboard.Key.page_up: 'pageup', keyboard.Key.page_down: 'pagedown',
                keyboard.Key.left: 'left', keyboard.Key.right: 'right', keyboard.Key.up: 'up', keyboard.Key.down: 'down',
                keyboard.Key.shift: 'shift', keyboard.Key.shift_l: 'shift', keyboard.Key.shift_r: 'shift',
                keyboard.Key.ctrl: 'ctrl', keyboard.Key.ctrl_l: 'ctrl', keyboard.Key.ctrl_r: 'ctrl',
                keyboard.Key.alt: 'alt', keyboard.Key.alt_l: 'alt', keyboard.Key.alt_r: 'alt',
                keyboard.Key.cmd: 'win', keyboard.Key.cmd_l: 'win', keyboard.Key.cmd_r: 'win'
            }
            if isinstance(k, keyboard.KeyCode) and k.char is not None:
                ch = k.char.lower()
                if ch:
                    return ch
            for i in range(1, 25):
                try:
                    if k == getattr(keyboard.Key, f"f{i}"): return f"f{i}"
                except AttributeError:
                    pass
            if k in mapping: return mapping[k]
        except Exception:
            pass
        return None

    def _match_combo(self, mods_set, base_token, want_combo: Optional[str]) -> bool:
        if not want_combo: return False
        req_mods, req_base = hk_to_tuple(hk_normalize(want_combo))
        if not req_base: return False
        if base_token != req_base: return False
        # 요구 모디파이어가 현재 누른 모디파이어의 부분집합이면 매칭
        return req_mods.issubset(mods_set)

    def _should_ignore_keypress(self, k) -> bool:
        """
        현재 모디파이어(self._mods)와 이번에 눌린 키 k로 구성되는 조합이
        self._ignore_combos 중 하나와 매칭되면 True.
        """
        base = self._token_from_key(k)
        if not base: return False
        for ig in self._ignore_combos:
            if self._match_combo(self._mods, base, ig):
                return True
        return False

    # ---------------------------------------------------------

    def _on_key_press(self, k):
        if self._now() < self._ignore_until:
            return

        # ★ 먼저: 무시해야 하는 단축키 조합이면 기록하지 않고 즉시 반환
        try:
            if self._should_ignore_keypress(k):
                # 살짝 쿨다운을 줘서 release 등으로 인한 부수 입력도 흡수
                self._ignore_until = self._now() + 0.08
                return
        except Exception:
            pass

        # F8 토글(일시정지/재개)
        if k == keyboard.Key.f8:
            self._flush_text(True); self._flush_click(True); self._flush_scroll(True)
            self._paused = not self._paused
            self.pausedChanged.emit(self._paused)
            return
        if self._paused:
            return

        # 1) 문자 키: 모디파이어(shift 제외)가 없으면 텍스트 버퍼링, 있으면 핫키
        try:
            if isinstance(k, keyboard.KeyCode) and k.char is not None:
                ch = k.char
                mods_effective = set(self._mods) - {'shift'}
                if mods_effective:
                    # 핫키로 기록
                    self._flush_text(True)
                    combo = "+".join(sorted(list(mods_effective))) + ("+" if mods_effective else "") + ch
                    self._steps.append(StepData(id=str(uuid.uuid4())[:8], name=f"Hotkey {combo}", type="key", key_string=combo))
                else:
                    # 텍스트 버퍼링
                    self._typed_last = self._now()
                    self._typed_buf += ch
        except Exception:
            pass

        # 2) 모디파이어 트래킹
        tok = self._key_token(k)
        if tok in ('shift', 'ctrl', 'alt', 'winleft', 'winright'):
            if tok == 'shift': self._mods.add('shift')
            elif tok == 'ctrl': self._mods.add('ctrl')
            elif tok == 'alt': self._mods.add('alt')
            else: self._mods.add('win')
            return

    def _on_key_release(self, k):
        tok = self._key_token(k)
        if tok in ('shift', 'ctrl', 'alt', 'winleft', 'winright'):
            if tok == 'shift': self._mods.discard('shift')
            elif tok == 'ctrl': self._mods.discard('ctrl')
            elif tok == 'alt': self._mods.discard('alt')
            else: self._mods.discard('win')
            return
        # 텍스트 플러시 (간격 기반)
        if self._typed_buf and (self._now() - self._typed_last) * 1000 >= self._typed_gap_ms:
            self._flush_text(False)
        # 스크롤 플러시
        if self._scroll_acc != (0, 0) and (self._now() - self._scroll_last) * 1000 >= self._scroll_flush_ms:
            self._flush_scroll(True)

    def _flush_text(self, force=False):
        if not self._typed_buf: return
        text = self._typed_buf
        self._typed_buf = ""
        self._steps.append(StepData(id=str(uuid.uuid4())[:8], name=f"Type '{text}'", type="key", key_string=text))

    def _on_move(self, x, y): pass

    def _on_click(self, x, y, button, pressed):
        if self._now() < self._ignore_until or self._paused: return
        if self._in_ignore(x, y): return
        btn = self._btn_name(button)
        if pressed:
            self._press_pos = (btn, x, y, self._now()); return
        if not self._press_pos: return
        pbtn, sx, sy, st = self._press_pos
        self._press_pos = None
        ex, ey, et = x, y, self._now()
        if pbtn != btn: return
        dist = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
        if dist <= self._click_merge_px:
                if (self._last_click and self._last_click[0] == btn and
                    abs(ex - self._last_click[1]) <= self._click_merge_px and
                    abs(ey - self._last_click[2]) <= self._click_merge_px and
                    (et - self._last_click[3]) * 1000 <= self._click_merge_ms):
                    b, lx, ly, lt, cnt = self._last_click
                    self._last_click = (b, lx, ly, et, cnt + 1)
                else:
                    self._flush_click(False)
                    self._last_click = (btn, ex, ey, et, 1)
        else:
            self._flush_click(True)
            self._steps.append(StepData(
                id=str(uuid.uuid4())[:8], name=f"Drag ({sx},{sy})->({ex},{ey})",
                type="drag", drag_from_x=sx, drag_from_y=sy, drag_to_x=ex, drag_to_y=ey,
                drag_duration_ms=max(150, int((et - st) * 1000))
            ))

    def _flush_click(self, force: bool):
        if not self._last_click: return
        btn, x, y, ts, cnt = self._last_click
        if not force and (self._now() - ts) * 1000 < self._click_merge_ms: return
        self._last_click = None
        name = "Click" if cnt == 1 else f"{cnt}x Click"
        self._steps.append(StepData(
            id=str(uuid.uuid4())[:8], name=f"{name} ({x},{y})", type="click_point",
            click_button=btn, click_x=x, click_y=y
        ))

    def _on_scroll(self, x, y, dx, dy):
        if self._now() < self._ignore_until or self._paused: return
        if self._in_ignore(x, y): return
        self._scroll_acc = (self._scroll_acc[0] + dx, self._scroll_acc[1] + dy)
        self._scroll_last = self._now()

    def _flush_scroll(self, force: bool):
        if self._scroll_acc == (0, 0): return
        if not force and (self._now() - self._scroll_last) * 1000 < self._scroll_flush_ms: return
        sx, sy = self._scroll_acc
        self._steps.append(StepData(
            id=str(uuid.uuid4())[:8], name=f"Scroll {sx},{sy}", type="scroll",
            scroll_dx=int(sx * self._scroll_scale_dx),
            scroll_dy=int(sy * self._scroll_scale_dy),
            scroll_times=1, scroll_interval_ms=0
        ))
        self._scroll_acc = (0, 0)

    def _key_token(self, k) -> Optional[str]:
        try:
            if isinstance(k, keyboard.KeyCode) and k.char is not None:
                return None
            mapping = {
                keyboard.Key.enter: 'enter', keyboard.Key.esc: 'esc', keyboard.Key.space: 'space',
                keyboard.Key.tab: 'tab', keyboard.Key.backspace: 'backspace', keyboard.Key.delete: 'delete',
                keyboard.Key.home: 'home', keyboard.Key.end: 'end', keyboard.Key.insert: 'insert',
                keyboard.Key.page_up: 'pageup', keyboard.Key.page_down: 'pagedown',
                keyboard.Key.left: 'left', keyboard.Key.right: 'right', keyboard.Key.up: 'up', keyboard.Key.down: 'down',
                keyboard.Key.shift: 'shift', keyboard.Key.shift_l: 'shift', keyboard.Key.shift_r: 'shift',
                keyboard.Key.ctrl: 'ctrl', keyboard.Key.ctrl_l: 'ctrl', keyboard.Key.ctrl_r: 'ctrl',
                keyboard.Key.alt: 'alt', keyboard.Key.alt_l: 'alt', keyboard.Key.alt_r: 'alt',
                keyboard.Key.cmd: 'winleft', keyboard.Key.cmd_l: 'winleft', keyboard.Key.cmd_r: 'winright'
            }
            if k in mapping: return mapping[k]
            for i in range(1, 25):
                try:
                    if k == getattr(keyboard.Key, f"f{i}"): return f"f{i}"
                except AttributeError:
                    pass
        except Exception:
            pass
        return None

# ---------------- Step List ----------------
class StepList(QListWidget):
    requestEdit = pyqtSignal(int)
    requestDelete = pyqtSignal(int)
    requestDuplicate = pyqtSignal(int)
    requestDeleteMany = pyqtSignal(list)   # NEW: 다중 삭제
    orderChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 다중 선택 허용 + 드래그 재정렬 유지
        self.setSelectionMode(self.ExtendedSelection)  # SingleSelection → ExtendedSelection
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(self.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)

    def dropEvent(self, e):
        super().dropEvent(e)
        self.orderChanged.emit()

    def keyPressEvent(self, e):
        # Delete 키로 선택 항목 삭제(단건/다건)
        if e.key() == Qt.Key_Delete:
            rows = sorted({self.row(it) for it in self.selectedItems()})
            if not rows:
                e.ignore(); return
            if len(rows) == 1:
                self.requestDelete.emit(rows[0])
            else:
                self.requestDeleteMany.emit(rows)
            return
        super().keyPressEvent(e)

    def show_menu(self, pos):
        it = self.itemAt(pos)
        sel_items = self.selectedItems()
        sel_rows = sorted({self.row(x) for x in sel_items})

        menu = QMenu(self)
        if sel_rows:
            if len(sel_rows) > 1:
                aDelMany = menu.addAction(f"Delete Selected ({len(sel_rows)})")
                aDelMany.triggered.connect(lambda: self.requestDeleteMany.emit(sel_rows))
                menu.addSeparator()
            # 단일 항목용 메뉴 (편의상 선택 다건이어도, 커서 아래 항목 기준으로 제공)
            if it:
                idx = self.row(it)
                aEdit = menu.addAction("Edit")
                aDel = menu.addAction("Delete")
                aDup = menu.addAction("Duplicate")
                aEdit.triggered.connect(lambda: self.requestEdit.emit(idx))
                aDel.triggered.connect(lambda: self.requestDelete.emit(idx))
                aDup.triggered.connect(lambda: self.requestDuplicate.emit(idx))
            else:
                menu.addAction("(no item under cursor)")
        else:
            menu.addAction("(no selection)")
        menu.exec_(self.viewport().mapToGlobal(pos))


# ---------------- Main Window ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Macro MVP (ROI)")
        self.resize(1200, 760)
        self.steps: List[StepData] = []
        self.runner: Optional[MacroRunner] = None
        self.recorder: Optional[InputRecorder] = None
        self._was_minimized = False
        self._live_overlays: List[CrosshairOverlay] = []
        self._load_record_settings()

        # Hotkey state
        self._hk_run = "end"
        self._hk_stop = "home"
        self._hk_record = "f9"
        # NEW: Add-step shortcuts (window-level)
        self._hk_add_img = "ctrl+shift+i"
        self._hk_add_notimg = "ctrl+shift+n"

        self._mods_global = set()
        self._hk_cooldown = {"run":0.0, "stop":0.0, "record":0.0}
        self._hk_cool_ms = 300
        self._qshortcuts: List[QShortcut] = []
        self._load_hotkeys()

        # Widgets
        self.list = StepList(self)
        self.list.orderChanged.connect(self.sync_order)
        self.list.itemSelectionChanged.connect(self.update_preview)
        self.list.requestEdit.connect(self.edit_step_at)
        self.list.requestDuplicate.connect(self.duplicate_step_at)
        self.list.requestDelete.connect(self.delete_step_at)
        self.list.requestDeleteMany.connect(self.delete_steps_at)
        
        self.lblPreview = QLabel("Preview")
        self.lblPreview.setMinimumHeight(240)
        self.lblPreview.setStyleSheet("background:#222;color:#aaa;border:1px solid #444;")

        self.btnAddImg = QPushButton("이미지 추가")
        self.btnAddImg.clicked.connect(self.add_image_step)
        self.btnAddNotImg = QPushButton("동작 추가")
        self.btnAddNotImg.clicked.connect(self.add_not_image_step)
        self.btnEdit = QPushButton("EDIT STEP"); self.btnEdit.clicked.connect(self.edit_selected_step)
        self.btnRecord = QPushButton("RECORD"); self.btnRecord.setCheckable(True); self.btnRecord.toggled.connect(self.toggle_record)

        self.btnRun = QPushButton("Run"); self.btnRun.clicked.connect(self.run_macro)
        self.btnStop = QPushButton("Stop"); self.btnStop.clicked.connect(self.stop_macro); self.btnStop.setEnabled(False)

        self.chkDry = QCheckBox("Dry Run (no click)")
        self.chkAutoMin = QCheckBox("Minimize while recording"); self.chkAutoMin.setChecked(False)

        self.btnSave = QPushButton("Save .macro"); self.btnSave.clicked.connect(self.save_macro)
        self.btnLoad = QPushButton("Load .macro"); self.btnLoad.clicked.connect(self.load_macro)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setStyleSheet("background:#111;color:#ddd;")

        # Layout
        left = QVBoxLayout()
        left.addWidget(self.list, 1)
        grid = QGridLayout()
        grid.addWidget(self.btnAddImg, 0, 0)
        grid.addWidget(self.btnAddNotImg, 0, 1)
        grid.addWidget(self.btnEdit, 1, 0)
        grid.addWidget(self.btnRecord, 1, 1)
        left.addLayout(grid)

        right = QVBoxLayout()
        right.addWidget(self.lblPreview)
        row2 = QHBoxLayout()
        row2.addWidget(self.btnRun); row2.addWidget(self.btnStop); row2.addWidget(self.chkDry); row2.addWidget(self.chkAutoMin)
        right.addLayout(row2)
        row3 = QHBoxLayout(); row3.addWidget(self.btnSave); row3.addWidget(self.btnLoad); right.addLayout(row3)
        # Scenario Repeat panel
        self.grpRepeat = QGroupBox("Scenario Repeat")
        self.frmRepeat = QFormLayout(self.grpRepeat)
        self.sbRepeatCount = QSpinBox(); self.sbRepeatCount.setRange(0, 10_000_000); self.sbRepeatCount.setValue(1)
        self.sbCooldown    = QSpinBox(); self.sbCooldown.setRange(0, 60_000);         self.sbCooldown.setValue(500)
        self.cbStopOnFail  = QCheckBox(); self.cbStopOnFail.setChecked(True)
        self.sbMaxDuration = QSpinBox(); self.sbMaxDuration.setRange(0, 24*60*60*1000); self.sbMaxDuration.setValue(0)
        self.frmRepeat.addRow("Repeat Count (0=∞)", self.sbRepeatCount)
        self.frmRepeat.addRow("Cooldown (ms)",       self.sbCooldown)
        self.frmRepeat.addRow("Stop on fail",        self.cbStopOnFail)
        self.frmRepeat.addRow("Max Duration (ms)",   self.sbMaxDuration)
        right.addWidget(self.grpRepeat)

        right.addWidget(self.log, 1)

        root = QHBoxLayout(); root.addLayout(left, 2); root.addLayout(right, 3)
        cw = QWidget(); cw.setLayout(root); self.setCentralWidget(cw)

        # Menu
        menu = self.menuBar().addMenu("Settings")
        actHK = QAction("Hotkeys...", self); menu.addAction(actHK); actHK.triggered.connect(self._open_hotkey_dialog)
        actRec = QAction("Recording Settings...", self); menu.addAction(actRec); actRec.triggered.connect(self._open_record_settings)

        # GLOBAL HOTKEY listener
        try:
            self._hotkey_listener = keyboard.Listener(on_press=self._on_global_key, on_release=self._on_global_key_up)
            self._hotkey_listener.start()
            self.info(f"Global hotkeys: Run[{hk_pretty(self._hk_run) or '-'}], Stop[{hk_pretty(self._hk_stop) or '-'}], Record[{hk_pretty(self._hk_record) or '-'}]")
        except Exception as e:
            self.info(f"Global hotkey listener init failed: {e}")
        self._update_hotkey_labels()
        self._install_qshortcuts()

    def delete_steps_at(self, rows: List[int]):
        """선택된 행 인덱스들(rows)을 일괄 삭제."""
        if not rows:
            return
        if self.runner and self.runner.isRunning():
            self.info("Cannot delete while runner is active.")
            return

        # 유효 인덱스만, 중복 제거 후 내림차순(뒤에서부터 pop)
        rows = sorted({r for r in rows if 0 <= r < len(self.steps)}, reverse=True)
        if not rows:
            return

        for r in rows:
            try:
                self.steps.pop(r)
                self.list.takeItem(r)
            except Exception:
                pass

        # 다음 선택 위치 정리
        next_row = min(rows[-1], self.list.count() - 1) if self.list.count() > 0 else -1
        if next_row >= 0:
            self.list.setCurrentRow(next_row)
        else:
            self.lblPreview.setPixmap(QPixmap())
            self.lblPreview.setToolTip("")

        self.update_preview()
        self.info(f"Deleted {len(rows)} step(s).")

    def closeEvent(self, e):
        # 러너 안전 종료
        try:
            if self.runner and self.runner.isRunning():
                self.runner.stop()
                self.runner.wait(2000)
        except Exception:
            pass
        # 레코더 정리
        try:
            if self.recorder:
                self.recorder.stop()
                self.recorder = None
        except Exception:
            pass
        # 글로벌 핫키 리스너 정지
        try:
            if hasattr(self, "_hotkey_listener") and self._hotkey_listener:
                self._hotkey_listener.stop()
                self._hotkey_listener = None
        except Exception:
            pass
        # 오버레이 닫기
        try:
            for ov in list(self._live_overlays):
                ov.close()
        except Exception:
            pass
        super().closeEvent(e)

    def info(self, msg: str):
        try:
            self.log.appendPlainText(str(msg))
        except Exception:
            print(str(msg))

    # --- Settings dialogs ---
    def _open_record_settings(self):
        dlg = RecordingSettingsDialog(self, (
            self.rec_typed_gap_ms, self.rec_click_merge_ms, self.rec_click_radius_px,
            self.rec_scroll_flush_ms, self.rec_scroll_scale_dx, self.rec_scroll_scale_dy
        ))
        if dlg.exec_() == QDialog.Accepted:
            (self.rec_typed_gap_ms, self.rec_click_merge_ms, self.rec_click_radius_px,
             self.rec_scroll_flush_ms, self.rec_scroll_scale_dx, self.rec_scroll_scale_dy) = dlg.values()
            self._save_record_settings()
            self.info("Recording settings updated.")

    def _load_record_settings(self):
        st = QSettings("ImageMacro","MVP")
        self.rec_typed_gap_ms    = int(st.value("rec/typed_gap_ms", 500))
        self.rec_click_merge_ms  = int(st.value("rec/click_merge_ms", 350))
        self.rec_click_radius_px = int(st.value("rec/click_radius_px", 3))
        self.rec_scroll_flush_ms = int(st.value("rec/scroll_flush_ms", 180))
        self.rec_scroll_scale_dx = float(st.value("rec/scroll_scale_dx", 30.0))
        self.rec_scroll_scale_dy = float(st.value("rec/scroll_scale_dy", 120.0))

    def _save_record_settings(self):
        st = QSettings("ImageMacro","MVP")
        st.setValue("rec/typed_gap_ms", self.rec_typed_gap_ms)
        st.setValue("rec/click_merge_ms", self.rec_click_merge_ms)
        st.setValue("rec/click_radius_px", self.rec_click_radius_px)
        st.setValue("rec/scroll_flush_ms", self.rec_scroll_flush_ms)
        st.setValue("rec/scroll_scale_dx", self.rec_scroll_scale_dx)
        st.setValue("rec/scroll_scale_dy", self.rec_scroll_scale_dy)

    # --- Hotkey UI helpers ---
    def _update_hotkey_labels(self):
        def with_hint(base, hk):
            return f"{base} ({hk_pretty(hk)})" if hk else base

        # Run/Stop/Record
        self.btnRun.setText(with_hint("Run", self._hk_run))
        self.btnStop.setText(with_hint("Stop", self._hk_stop))
        if self.btnRecord.isChecked():
            self.btnRecord.setText(with_hint("STOP (RECORDING)", self._hk_record))
        else:
            self.btnRecord.setText(with_hint("RECORD", self._hk_record))

        # Add buttons
        self.btnAddImg.setText(with_hint("이미지 추가", self._hk_add_img))
        self.btnAddNotImg.setText(with_hint("동작 추가", self._hk_add_notimg))

        # 🔽 여기 추가: 툴팁 동기화
        self.btnAddImg.setToolTip(
            f"Shortcut: {hk_pretty(self._hk_add_img)}" if self._hk_add_img else "Shortcut: -"
        )
        self.btnAddNotImg.setToolTip(
            f"Shortcut: {hk_pretty(self._hk_add_notimg)}" if self._hk_add_notimg else "Shortcut: -"
        )


    def _clear_qshortcuts(self):
        for sc in self._qshortcuts:
            sc.setParent(None)
        self._qshortcuts.clear()

    def _install_qshortcuts(self):
        self._clear_qshortcuts()

        def make_sc(combo, slot):
            if not combo: 
                return
            mods, base = hk_to_tuple(combo)
            if not base: 
                return
            seq = "+".join(list(mods) + [base]).replace("win", "Meta")
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(slot)
            self._qshortcuts.append(sc)

        # 기존
        make_sc(self._hk_run, self._act_run_from_hotkey)
        make_sc(self._hk_stop, self._act_stop_from_hotkey)
        make_sc(self._hk_record, self._act_record_from_hotkey)

        # NEW: Add-step shortcuts
        make_sc(self._hk_add_img, self.add_image_step)
        make_sc(self._hk_add_notimg, self.add_not_image_step)


    def _load_hotkeys(self):
        st = QSettings("ImageMacro", "MVP")
        self._hk_run    = hk_normalize(st.value("hotkeys/run",    self._hk_run))
        self._hk_stop   = hk_normalize(st.value("hotkeys/stop",   self._hk_stop))
        self._hk_record = hk_normalize(st.value("hotkeys/record", self._hk_record))
        # NEW
        self._hk_add_img    = hk_normalize(st.value("hotkeys/add_img",    self._hk_add_img))
        self._hk_add_notimg = hk_normalize(st.value("hotkeys/add_notimg", self._hk_add_notimg))

    def _save_hotkeys(self):
        st = QSettings("ImageMacro", "MVP")
        st.setValue("hotkeys/run",    self._hk_run)
        st.setValue("hotkeys/stop",   self._hk_stop)
        st.setValue("hotkeys/record", self._hk_record)
        # NEW
        st.setValue("hotkeys/add_img",    self._hk_add_img)
        st.setValue("hotkeys/add_notimg", self._hk_add_notimg)

    def _open_hotkey_dialog(self):
        dlg = HotkeySettingsDialog(
            self._hk_run,
            self._hk_stop,
            self._hk_record,
            self._hk_add_img,
            self._hk_add_notimg,
            self
        )
        if dlg.exec_() == QDialog.Accepted:
            res = dlg.result_hotkeys()
            if res:
                (self._hk_run,
                 self._hk_stop,
                 self._hk_record,
                 self._hk_add_img,
                 self._hk_add_notimg) = res
                self._save_hotkeys()
                self._update_hotkey_labels()
                self._install_qshortcuts()
                self.info(
                    f"Hotkeys updated: "
                    f"run={self._hk_run or '-'}, "
                    f"stop={self._hk_stop or '-'}, "
                    f"record={self._hk_record or '-'}, "
                    f"add_img={self._hk_add_img or '-'}, "
                    f"add_action={self._hk_add_notimg or '-'}"
                )

    # --- Hotkey global matching ---
    def _match_combo(self, mods_set, base_token, want_combo: str) -> bool:
        req_mods, req_base = hk_to_tuple(want_combo)
        if not req_base: return False
        if base_token != req_base: return False
        return req_mods.issubset(mods_set)

    def _token_from_key(self, k) -> Optional[str]:
        try:
            mapping = {
                keyboard.Key.enter: 'enter', keyboard.Key.esc: 'esc', keyboard.Key.space: 'space',
                keyboard.Key.tab: 'tab', keyboard.Key.backspace: 'backspace', keyboard.Key.delete: 'delete',
                keyboard.Key.home: 'home', keyboard.Key.end: 'end', keyboard.Key.insert: 'insert',
                keyboard.Key.page_up: 'pageup', keyboard.Key.page_down: 'pagedown',
                keyboard.Key.left: 'left', keyboard.Key.right: 'right', keyboard.Key.up: 'up', keyboard.Key.down: 'down',
                keyboard.Key.shift: 'shift', keyboard.Key.shift_l: 'shift', keyboard.Key.shift_r: 'shift',
                keyboard.Key.ctrl: 'ctrl', keyboard.Key.ctrl_l: 'ctrl', keyboard.Key.ctrl_r: 'ctrl',
                keyboard.Key.alt: 'alt', keyboard.Key.alt_l: 'alt', keyboard.Key.alt_r: 'alt',
                keyboard.Key.cmd: 'win', keyboard.Key.cmd_l: 'win', keyboard.Key.cmd_r: 'win'
            }
            if isinstance(k, keyboard.KeyCode) and k.char is not None:
                ch = k.char.lower()
                if ch.isalnum() or ch in "-=[]\\;',./`":
                    return ch
            for i in range(1, 25):
                try:
                    if k == getattr(keyboard.Key, f"f{i}"): return f"f{i}"
                except Exception:
                    pass
            if k in mapping: return mapping[k]
            return None
        except Exception:
            return None

    def _act_run_from_hotkey(self):
        if self.recorder and getattr(self.recorder, "_active", False):
            self.info("[Hotkey] ignored: recording in progress."); return
        if self.runner and self.runner.isRunning():
            self.info("[Hotkey] ignored: already running."); return
        if not self.steps:
            self.info("[Hotkey] ignored: no steps."); return
        self.info("[Hotkey] Run")
        self.run_macro()

    def _act_stop_from_hotkey(self):
        self.info("[Hotkey] Stop")
        self.stop_macro()

    def _act_record_from_hotkey(self):
        if self.runner and self.runner.isRunning():
            self.info("[Hotkey] ignored: running in progress."); return
        self.info("[Hotkey] Toggle Record")
        self.btnRecord.toggle()

    def _on_global_key(self, key):
        try:
            tok = self._token_from_key(key)
            if tok in ('shift','ctrl','alt','win'):
                self._mods_global.add(tok); return
            def ready(tag):
                now = time.time() * 1000
                if now - self._hk_cooldown[tag] < self._hk_cool_ms: return False
                self._hk_cooldown[tag] = now; return True
            base = tok
            if self._hk_run and self._match_combo(self._mods_global, base, self._hk_run) and ready("run"):
                self._act_run_from_hotkey(); return
            if self._hk_stop and self._match_combo(self._mods_global, base, self._hk_stop) and ready("stop"):
                self._act_stop_from_hotkey(); return
            if self._hk_record and self._match_combo(self._mods_global, base, self._hk_record) and ready("record"):
                self._act_record_from_hotkey(); return
        except Exception:
            pass

    def _on_global_key_up(self, key):
        try:
            tok = self._token_from_key(key)
            if tok in ('shift','ctrl','alt','win'):
                self._mods_global.discard(tok)
        except Exception:
            pass

    # --- preview / list ops ---
    def update_preview(self):
        it = self.list.currentItem()
        if not it:
            self.lblPreview.setPixmap(QPixmap()); self.lblPreview.setToolTip(""); return
        s: StepData = it.data(Qt.UserRole)
        if s.type == "image_click" and s.ensure_tpl() is not None:
            self.lblPreview.setPixmap(cvimg_to_qpixmap(s.ensure_tpl()).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if s.search_roi_enabled and s.search_roi_width > 0 and s.search_roi_height > 0:
                self.lblPreview.setToolTip(f"ROI: {s.search_roi_left},{s.search_roi_top} {s.search_roi_width}x{s.search_roi_height}")
            else:
                self.lblPreview.setToolTip("ROI: full screen")
        else:
            self.lblPreview.setPixmap(QPixmap()); self.lblPreview.setToolTip("")

    def add_list_item(self, s: StepData):
        it = QListWidgetItem(s.name)
        it.setData(Qt.UserRole, s)
        if s.type == "image_click" and s.ensure_tpl() is not None:
            it.setIcon(QIcon(cvimg_to_qpixmap(s.ensure_tpl()).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        else:
            it.setIcon(make_letter_icon({"key": "K", "key_down": "Kd", "key_up": "Ku", "key_hold":"Kh",
                                         "click_point": "C", "drag": "D", "scroll": "S"}.get(s.type, "S")[0]))
        self.list.addItem(it)

    def add_image_step(self):
        try:
            self.info("Select template: 화면을 덮는 반투명 오버레이가 뜹니다. 마우스로 드래그해 영역을 지정하세요. (ESC 취소)")
            rect, crop, _ = ROISelector.select_from_screen(self)
            if crop is None:
                self.info("Canceled.")
                return
            s = StepData(id=str(uuid.uuid4())[:8], name=f"Image Step #{len(self.steps)+1}",
                         type="image_click", png_bytes=encode_png_bytes(crop))
            s._tpl_bgr = crop
            self.steps.append(s)
            self.add_list_item(s)
            self.info(f"Added {s.name}: {rect.width()}x{rect.height()}")
        except Exception as e:
            QMessageBox.critical(self, "Add Image Step Error", str(e))

    def add_not_image_step(self):
        dlg = NotImageDialog(None, self)
        if dlg.exec_() == QDialog.Accepted:
            s = dlg.result_step()
            if s:
                self.steps.append(s); self.add_list_item(s)
        # 안전핀: 포커스 복구
        self.raise_(); self.activateWindow()

    def edit_selected_step(self):
        it = self.list.currentItem()
        if not it: return
        idx = self.list.row(it); self.edit_step_at(idx)

    def edit_step_at(self, idx: int):
        if not (0 <= idx < len(self.steps)): return
        s = self.steps[idx]
        if s.type == "image_click":
            dlg = ImageStepDialog(s, self)
            if dlg.exec_() == QDialog.Accepted:
                it = self.list.item(idx); it.setData(Qt.UserRole, s)
                if s.ensure_tpl() is not None:
                    it.setIcon(QIcon(cvimg_to_qpixmap(s.ensure_tpl()).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                self.update_preview()
        else:
            dlg = NotImageDialog(s, self)
            if dlg.exec_() == QDialog.Accepted:
                ns = dlg.result_step()
                if ns:
                    self.steps[idx] = ns
                    it = self.list.item(idx); it.setData(Qt.UserRole, ns); it.setText(ns.name)
                    self.update_preview()
        # 안전핀: 포커스 복구
        self.raise_(); self.activateWindow()

    def duplicate_step_at(self, idx: int):
        if not (0 <= idx < len(self.steps)): return
        s = self.steps[idx]
        # Create a new dictionary for the new step, excluding non-constructor fields
        new_data = {k: v for k, v in asdict(s).items() if k not in ('_tpl_bgr', '_last_match_xy', '_tpl_cache', 'png_bytes')}
        new_data['id'] = str(uuid.uuid4())[:8]
        new_data['name'] = s.name + " (copy)"
        new_data['png_bytes'] = s.png_bytes # Copy the original png bytes
        
        ns = StepData(**new_data)
        ns._tpl_bgr = s._tpl_bgr.copy() if s._tpl_bgr is not None else None
        
        self.steps.insert(idx + 1, ns)
        
        # Rebuild list widget to reflect new order
        self.list.clear()
        for step in self.steps:
            self.add_list_item(step)
        self.list.setCurrentRow(idx + 1)

    def delete_step_at(self, idx: int):
        if not (0 <= idx < len(self.steps)): return
        self.steps.pop(idx)
        self.list.takeItem(idx)
        self.update_preview()

    def sync_order(self):
        new_steps = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            new_steps.append(it.data(Qt.UserRole))
        self.steps = new_steps

    def toggle_record(self, on: bool):
        self._update_hotkey_labels()
        if on:
            self.info("[Record] start")
            self.btnRun.setEnabled(False); self.btnStop.setEnabled(False)
            if self.chkAutoMin.isChecked():
                self._was_minimized = True; self.showMinimized()
            else:
                self._was_minimized = False
            self._start_record()
        else:
            self.info("[Record] stop")
            self.btnRun.setEnabled(True); self.btnStop.setEnabled(True)
            if self._was_minimized:
                self._was_minimized = False; self.showNormal()
            self._stop_record()

    def _start_record(self):
        g = self.geometry()
        ignore_rect = QRect(g.x(), g.y(), g.width(), g.height())
        self.recorder = InputRecorder(
            ignore_rect, self,
            typed_gap_ms=self.rec_typed_gap_ms,
            click_merge_ms=self.rec_click_merge_ms,
            click_radius_px=self.rec_click_radius_px,
            scroll_flush_ms=self.rec_scroll_flush_ms,
            scroll_scale_dx=self.rec_scroll_scale_dx,
            scroll_scale_dy=self.rec_scroll_scale_dy,
            ignore_combos=[self._hk_record]    # ★ 녹화 단축키는 기록에서 무시
            # 필요하면 run/stop도 함께 무시: ignore_combos=[self._hk_record, self._hk_run, self._hk_stop]
        )
        self.recorder.finished.connect(self._on_record_done)
        self.recorder.pausedChanged.connect(lambda p: self.info(f"[Record] {'Paused' if p else 'Resumed'}"))
        self.recorder.start()

    def _stop_record(self):
        if not self.recorder: return
        self.recorder.stop(); self.recorder = None

    def _on_record_done(self, steps: List[StepData]):
        self.info(f"[Record] got {len(steps)} steps")
        for s in steps:
            self.steps.append(s)
            self.add_list_item(s)

    def run_macro(self):
        if self.runner and self.runner.isRunning():
            return
        self.btnRun.setEnabled(False); self.btnStop.setEnabled(True)
        repeat_cfg = RepeatConfig(
            repeat_count=self.sbRepeatCount.value(),
            repeat_cooldown_ms=self.sbCooldown.value(),
            stop_on_fail=self.cbStopOnFail.isChecked(),
            max_duration_ms=self.sbMaxDuration.value(),
        )
        self.runner = MacroRunner(self.steps, repeat=repeat_cfg, dry_run=self.chkDry.isChecked(), parent=self)
        self.runner.log.connect(self.info)
        self.runner.requestCrosshair.connect(lambda x,y,dur: self._spawn_crosshair(x,y,dur))
        self.runner.finished.connect(self._on_run_finished)
        self.runner.start()

    def stop_macro(self):
        if self.runner: self.runner.stop()

    def _on_run_finished(self, ok: bool):
        self.btnRun.setEnabled(True); self.btnStop.setEnabled(False)
        self.info(f"[Run] {'OK' if ok else 'FAILED'}")

    def _spawn_crosshair(self, x, y, dur):
        try:
            with mss.mss() as sct:
                mon = sct.monitors[0]
                left, top, w, h = int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"])
        except Exception:
            scr = QApplication.primaryScreen()
            vg = scr.virtualGeometry()
            left, top, w, h = vg.x(), vg.y(), vg.width(), vg.height()

        ov = CrosshairOverlay(left, top, w, h, x, y, dur)
        self._live_overlays.append(ov)
        QTimer.singleShot(dur + 50, lambda: self._live_overlays.remove(ov) if ov in self._live_overlays else None)

    def save_macro(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Macro", "", "Macro (*.macro)")
        if not path: return
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                scen = {"version": 8, "repeat": {"repeat_count": self.sbRepeatCount.value(), "repeat_cooldown_ms": self.sbCooldown.value(), "stop_on_fail": self.cbStopOnFail.isChecked(), "max_duration_ms": self.sbMaxDuration.value()}, "steps": [s.to_serializable() for s in self.steps]}
                z.writestr("scenario.json", json.dumps(scen, ensure_ascii=False, indent=2))
                for s in self.steps:
                    if s.type == "image_click" and s.png_bytes:
                        z.writestr(f"images/{s.id}.png", s.png_bytes)
            self.info(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_macro(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Macro", "", "Macro (*.macro)")
        if not path: return
        try:
            with zipfile.ZipFile(path, "r") as z:
                scen = json.loads(z.read("scenario.json").decode("utf-8"))
                # apply repeat config if available (v7+)
                rep = scen.get("repeat", {})
                try:
                    rc = RepeatConfig.from_json(rep)
                    self.sbRepeatCount.setValue(rc.repeat_count)
                    self.sbCooldown.setValue(rc.repeat_cooldown_ms)
                    self.cbStopOnFail.setChecked(rc.stop_on_fail)
                    self.sbMaxDuration.setValue(rc.max_duration_ms)
                except Exception:
                    pass
                steps = []
                for d in scen.get("steps", []):
                    png = None
                    if d.get("image_path"):
                        try: png = z.read(d["image_path"])
                        except: png = None
                    
                    # Get all fields from dataclass to build the object
                    valid_fields = {f.name for f in dataclass_fields(StepData)}
                    ctor_args = {k: v for k, v in d.items() if k in valid_fields}
                    ctor_args['png_bytes'] = png

                    s = StepData(**ctor_args)

                    mt = d.get('match')
                    if isinstance(mt, dict):
                        for k,v in mt.items():
                            if hasattr(s, k):
                                try: setattr(s, k, v)
                                except: pass
                    s._tpl_bgr = decode_png_bytes(png) if png else None
                    steps.append(s)
            self.steps = steps; self.list.clear()
            for s in self.steps: self.add_list_item(s)
            self.update_preview()
            self.info(f"Loaded: {path} ({len(self.steps)} steps)")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))

# ---------------- entry ----------------
def _excepthook(etype, value, tb):
    # 콘솔에도 출력
    traceback.print_exception(etype, value, tb)
    # GUI 알림 (QApplication 생성 이후이므로 사용 가능)
    try:
        QMessageBox.critical(None, "Unhandled Error", f"{value}")
    except Exception:
        pass

def main():
    app = QApplication(sys.argv)
    sys.excepthook = _excepthook   # ← 여기! QApplication 바로 다음 줄
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
