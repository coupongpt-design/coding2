from typing import List, Optional
import time
import uuid
import pyautogui
from PyQt5.QtCore import QObject, QRect, pyqtSignal
from pynput import keyboard, mouse

from utils import (
    info,
    warn,
    err,
    encode_png_bytes,
    _normalize_point_result,
    safe_select_point,
    hk_normalize,
    hk_to_tuple,
)
from core.models import StepData


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
