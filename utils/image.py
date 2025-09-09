import time
from typing import Optional
import numpy as np
import cv2
from PyQt5.QtGui import QPixmap, QImage


def cvimg_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def encode_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    return bytes(buf) if ok else b""


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

class MatchResult:
    def __init__(self, ok: bool, x: int = None, y: int = None, score: float = 0.0, w: int = 0, h: int = 0,
                 stage: str = "single", scale: float = 1.0, angle: float = 0.0):
        self.ok = ok
        self.x = x
        self.y = y
        self.score = score
        self.w = w
        self.h = h
        self.stage = stage
        self.scale = scale
        self.angle = angle


class Matcher:
    def __init__(self):
        pass

    def _pp_key(self, step):
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
        cache = getattr(step, "_tpl_cache", None)
        if cache is None:
            step._tpl_cache = cache = {}
        ident = int(tpl_like.__array_interface__['data'][0])
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

    def find_best(self, frame_bgr: np.ndarray, step) -> MatchResult:
        tpl_bgr = step.ensure_tpl()
        if tpl_bgr is None:
            return MatchResult(False)

        th = float(getattr(step, "min_confidence", getattr(step, "threshold", 0.85)))
        delta = 0.03
        budget_ms = int(getattr(step, "budget_ms", 0))
        t0 = time.time()

        def over_budget():
            if budget_ms <= 0:
                return False
            return (time.time() - t0) * 1000.0 > budget_ms

        tpl_pp = self._get_pp_tpl(tpl_bgr, step)
        res = self._single_scale(_apply_preprocess(frame_bgr, step), tpl_pp, step)
        if res and res[0][0] >= th:
            sc, x, y = res[0]
            return MatchResult(True, x + tpl_pp.shape[1]//2, y + tpl_pp.shape[0]//2, sc,
                               tpl_pp.shape[1], tpl_pp.shape[0], stage="single")
        if res and res[0][0] >= (th + delta):
            sc, x, y = res[0]
            return MatchResult(True, x + tpl_pp.shape[1]//2, y + tpl_pp.shape[0]//2, sc,
                               tpl_pp.shape[1], tpl_pp.shape[0], stage="single-fast")

        if getattr(step, "ms_enable", False) and not over_budget():
            best_ms, _ = self._multiscale(_apply_preprocess(frame_bgr, step), tpl_pp, step,
                                          th, delta, over_budget)
            if best_ms and best_ms[0][0] >= th:
                sc, x, y = best_ms[0]
                return MatchResult(True, x + tpl_pp.shape[1]//2, y + tpl_pp.shape[0]//2, sc,
                                   tpl_pp.shape[1], tpl_pp.shape[0], stage="multiscale")
        return MatchResult(False)

    def _single_scale(self, img_pp, tpl_pp, step):
        try:
            res = cv2.matchTemplate(img_pp, tpl_pp, cv2.TM_CCOEFF_NORMED)
            return _extract_topk(res, int(getattr(step, "top_k", 1)), tpl_pp.shape[1], tpl_pp.shape[0])
        except Exception:
            return []

    def _multiscale(self, img_pp, tpl_pp, step, th, delta, over_budget_cb):
        scales = np.linspace(float(getattr(step, "ms_scale_min", 0.7)),
                             float(getattr(step, "ms_scale_max", 1.3)),
                             int(getattr(step, "ms_scale_steps", 5)))
        best = []
        best_seen = -1.0
        for sc in scales:
            if over_budget_cb():
                break
            tpl_s = self._get_scale_tpl(tpl_pp, step, sc)
            if tpl_s is None:
                continue
            try:
                res = cv2.matchTemplate(img_pp, tpl_s, cv2.TM_CCOEFF_NORMED)
                cands = _extract_topk(res, int(getattr(step, "top_k", 1)), tpl_s.shape[1], tpl_s.shape[0])
                for (score, x, y) in cands:
                    best.append((score, x, y, sc))
                    if score > best_seen:
                        best_seen = score
            except Exception:
                continue
            if best_seen >= (th + delta):
                break
        best.sort(key=lambda t: t[0], reverse=True)
        return best, {"scales": scales}


def decode_png_bytes(png: Optional[bytes]) -> Optional[np.ndarray]:
    if not png:
        return None
    data = np.frombuffer(png, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img
