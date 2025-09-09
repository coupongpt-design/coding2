from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple
import numpy as np
from utils import decode_png_bytes


@dataclass
class StepData:
    id: str
    name: str
    type: str
    png_bytes: Optional[bytes] = None
    threshold: float = 0.85
    timeout_ms: int = 5000
    poll_ms: int = 100
    jitter: int = 2
    pre_move_sleep_ms: int = 30
    press_duration_ms: int = 70
    post_click_sleep_ms: int = 80
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
    hold_until_next: bool = False
    hold_timeout_ms: int = 5000
    hold_reclick_interval_ms: int = 500
    hold_reacquire_each_time: bool = False
    hold_release_consecutive: int = 1
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
    search_roi_enabled: bool = False
    search_roi_left: int = 0
    search_roi_top: int = 0
    search_roi_width: int = 0
    search_roi_height: int = 0
    on_fail_action: str = "default"
    on_fail_target_id: Optional[str] = None
    _tpl_bgr: Optional[np.ndarray] = None
    _last_match_xy: Optional[Tuple[int,int]] = None
    _tpl_cache: dict = field(default_factory=dict)

    def ensure_tpl(self) -> Optional[np.ndarray]:
        if self._tpl_bgr is not None:
            return self._tpl_bgr
        if not self.png_bytes:
            return None
        self._tpl_bgr = decode_png_bytes(self.png_bytes)
        return self._tpl_bgr

    def to_serializable(self) -> dict:
        d = asdict(self)
        d.pop('_tpl_cache', None)
        d.pop('_tpl_bgr', None)
        d.pop('_last_match_xy', None)
        if self.type == "image_click" and self.png_bytes:
            d["image_path"] = f"images/{self.id}.png"
        d.pop("png_bytes", None)
        if self.type == 'image_click' and self.search_roi_enabled and self.search_roi_width > 0 and self.search_roi_height > 0:
            d['search_roi'] = {
                'left': int(self.search_roi_left),
                'top': int(self.search_roi_top),
                'width': int(self.search_roi_width),
                'height': int(self.search_roi_height)
            }
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


@dataclass
class RepeatConfig:
    repeat_count: int = 1
    repeat_cooldown_ms: int = 500
    stop_on_fail: bool = True
    max_duration_ms: int = 0

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
