import time
import traceback
from typing import List, Optional
import cv2
import numpy as np
import mss
import pyautogui
from PyQt5.QtCore import QThread, pyqtSignal

from utils import info, warn, err, Matcher
from core.models import StepData, RepeatConfig


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

        return False


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
            btn = s.click_button or "left"
            pyautogui.moveTo(int(s.drag_from_x), int(s.drag_from_y))
            pyautogui.mouseDown(button=btn)
            pyautogui.dragTo(
                int(s.drag_to_x),
                int(s.drag_to_y),
                max(1, int(s.drag_duration_ms)) / 1000.0,
                button=btn,
            )
            pyautogui.mouseUp(button=btn)
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
