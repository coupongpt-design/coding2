import json
import uuid
import time
import traceback
import zipfile
from dataclasses import fields as dataclass_fields, asdict
from typing import List, Optional

import numpy as np
import cv2
import mss
from pynput import keyboard
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QSettings, QTimer, QEventLoop, QObject, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QPixmap, QImage, QPainter, QColor, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFileDialog, QPlainTextEdit, QListWidget, QListWidgetItem, QMainWindow, QAction,
    QCheckBox, QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit,
    QMessageBox, QShortcut, QDialogButtonBox, QGroupBox, QComboBox, QMenu, QInputDialog
)

from utils import (
    cvimg_to_qpixmap,
    encode_png_bytes,
    info,
    warn,
    err,
    make_letter_icon,
    hk_pretty,
    hk_normalize,
    hk_to_tuple,
)
from core.models import StepData, RepeatConfig
from core.runner import MacroRunner
from core.recorder import InputRecorder
from gui.dialogs import NotImageDialog, ImageStepDialog, RecordingSettingsDialog, HotkeySettingsDialog
from gui.overlays import CrosshairOverlay, ROISelector, PointSelector


KEY_MAPPING = {
    keyboard.Key.enter: "enter",
    keyboard.Key.esc: "esc",
    keyboard.Key.space: "space",
    keyboard.Key.tab: "tab",
    keyboard.Key.backspace: "backspace",
    keyboard.Key.delete: "delete",
    keyboard.Key.home: "home",
    keyboard.Key.end: "end",
    keyboard.Key.insert: "insert",
    keyboard.Key.page_up: "pageup",
    keyboard.Key.page_down: "pagedown",
    keyboard.Key.left: "left",
    keyboard.Key.right: "right",
    keyboard.Key.up: "up",
    keyboard.Key.down: "down",
    keyboard.Key.shift: "shift",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
    keyboard.Key.alt: "alt",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.cmd: "win",
    keyboard.Key.cmd_l: "win",
    keyboard.Key.cmd_r: "win",
}

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

    def _safe_cleanup(self, func, desc: str):
        try:
            func()
        except (RuntimeError, AttributeError) as e:
            self.info(f"{desc}: {e}")

    def closeEvent(self, e):
        # 러너 안전 종료
        if self.runner and self.runner.isRunning():
            self._safe_cleanup(lambda: (self.runner.stop(), self.runner.wait(2000)), "Runner cleanup failed")

        # 레코더 정리
        if self.recorder:
            self._safe_cleanup(self.recorder.stop, "Recorder stop failed")
            self.recorder = None

        # 글로벌 핫키 리스너 정지
        if hasattr(self, "_hotkey_listener") and self._hotkey_listener:
            self._safe_cleanup(self._hotkey_listener.stop, "Hotkey listener stop failed")
            self._hotkey_listener = None

        # 오버레이 닫기
        for ov in list(self._live_overlays):
            self._safe_cleanup(ov.close, "Overlay close failed")

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
        if isinstance(k, keyboard.KeyCode) and k.char is not None:
            ch = k.char.lower()
            if ch.isalnum() or ch in "-=[]\\;',./`":
                return ch
        for i in range(1, 25):
            fkey = getattr(keyboard.Key, f"f{i}", None)
            if k == fkey:
                return f"f{i}"
        return KEY_MAPPING.get(k)

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
        tok = self._token_from_key(key)
        if tok in ('shift', 'ctrl', 'alt', 'win'):
            self._mods_global.add(tok)
            return
        def ready(tag):
            now = time.time() * 1000
            if now - self._hk_cooldown[tag] < self._hk_cool_ms:
                return False
            self._hk_cooldown[tag] = now
            return True
        base = tok
        if self._hk_run and self._match_combo(self._mods_global, base, self._hk_run) and ready("run"):
            self._act_run_from_hotkey(); return
        if self._hk_stop and self._match_combo(self._mods_global, base, self._hk_stop) and ready("stop"):
            self._act_stop_from_hotkey(); return
        if self._hk_record and self._match_combo(self._mods_global, base, self._hk_record) and ready("record"):
            self._act_record_from_hotkey(); return

    def _on_global_key_up(self, key):
        tok = self._token_from_key(key)
        if tok in ('shift', 'ctrl', 'alt', 'win'):
            self._mods_global.discard(tok)

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
                repeat_cfg = RepeatConfig(
                    repeat_count=self.sbRepeatCount.value(),
                    repeat_cooldown_ms=self.sbCooldown.value(),
                    stop_on_fail=self.cbStopOnFail.isChecked(),
                    max_duration_ms=self.sbMaxDuration.value(),
                )
                scen = {
                    "repeat": asdict(repeat_cfg),
                    "steps": [s.to_serializable() for s in self.steps],
                    "version": 8,
                }
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
                        try:
                            png = z.read(d["image_path"])
                        except Exception as e:
                            self.info(f"Failed to read image '{d['image_path']}' from '{path}': {e}. Skipping step.")
                            continue
                    
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
