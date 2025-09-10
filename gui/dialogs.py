from typing import Optional
import json
import uuid
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QRect, QPoint, QEventLoop, QTimer
from PyQt5.QtGui import QCursor, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QLineEdit, QDialogButtonBox,
    QVBoxLayout, QLabel, QPushButton, QGroupBox, QHBoxLayout,
    QComboBox, QMessageBox, QCheckBox, QApplication
)
from PyQt5.QtGui import QKeySequence, QCursor

from utils import (
    cvimg_to_qpixmap,
    encode_png_bytes,
    _normalize_point_result,
    info,
    hk_normalize,
    safe_select_point,
)
from core.models import StepData
from gui.overlays import ROISelector, PointSelector


class KeyCaptureEdit(QLineEdit):
    def keyPressEvent(self, e):
        key = e.key()
        if key in (Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta):
            return
        if key in (Qt.Key_Backspace, Qt.Key_Delete) and not e.modifiers():
            self.clear()
            e.accept()
            return
        seq = QKeySequence(int(e.modifiers()) | key).toString()
        seq = seq.replace("Meta+", "Win+").replace("Meta", "Win")
        txt = hk_normalize(seq)
        if txt:
            self.setText(txt)
        e.accept()


class NotImageDialog(QDialog):
    _last_point = None  # 마지막으로 선택된 좌표 기억

    def __init__(self, step: Optional[StepData], parent=None):
        super().__init__(parent)
        self._accepted = False
        self.setWindowTitle("Not-Image Step")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModal)

        form = QFormLayout(self)

        cur = QCursor.pos()
        last = NotImageDialog._last_point or (cur.x(), cur.y())
        
        # 기본 필드
        self.cbType = QComboBox()
        types = ["key", "key_down", "key_up", "key_hold", "click_point", "drag", "scroll"]
        self.cbType.addItems(types)
        if step and step.type in types:
            self.cbType.setCurrentText(step.type)
        self.edName = QLineEdit(step.name if step else "")
        self.edKey = KeyCaptureEdit(step.key_string or "" if step else "")
        self.spTimes = QSpinBox(); self.spTimes.setRange(1, 99); self.spTimes.setValue(step.key_times if step else 1)
        self.spHold = QSpinBox();  self.spHold.setRange(0, 10000); self.spHold.setValue(step.hold_ms if step else 0)

        # 스크롤/드래그 목적지(또는 벡터)
        self.spDx = QSpinBox(); self.spDx.setRange(-9999, 9999)
        self.spDy = QSpinBox(); self.spDy.setRange(-9999, 9999)
        if step:
            if step.type == "scroll":
                self.spDx.setValue(step.scroll_dx if step.scroll_dx is not None else 0)
                self.spDy.setValue(step.scroll_dy if step.scroll_dy is not None else 0)
            elif step.drag_to_x is not None and step.drag_to_y is not None:
                self.spDx.setValue(step.drag_to_x)
                self.spDy.setValue(step.drag_to_y)
            else:
                self.spDx.setValue(last[0])
                self.spDy.setValue(last[1])
        else:
            self.spDx.setValue(last[0])
            self.spDy.setValue(last[1])
        self.spST = QSpinBox(); self.spST.setRange(1, 99);       self.spST.setValue(step.scroll_times if step else 1)
        self.spSI = QSpinBox(); self.spSI.setRange(0, 10000);    self.spSI.setValue(step.scroll_interval_ms if step else 0)
        self.spDragDur = QSpinBox(); self.spDragDur.setRange(1, 10000);
        self.spDragDur.setValue(step.drag_duration_ms if step and step.type == "drag" else 200)

        # 클릭/드래그 시작 좌표
        self.spClickX = QSpinBox(); self.spClickX.setRange(-9999, 9999)
        self.spClickY = QSpinBox(); self.spClickY.setRange(-9999, 9999)
        if step:
            if step.click_x is not None and step.click_y is not None:
                self.spClickX.setValue(step.click_x)
                self.spClickY.setValue(step.click_y)
            elif step.drag_from_x is not None and step.drag_from_y is not None:
                self.spClickX.setValue(step.drag_from_x)
                self.spClickY.setValue(step.drag_from_y)
            else:
                self.spClickX.setValue(cur.x())
                self.spClickY.setValue(cur.y())
        else:
            self.spClickX.setValue(cur.x())
            self.spClickY.setValue(cur.y())
        self.edBtn     = QLineEdit(step.click_button if step else "left")

        # 폼 배치
        form.addRow("Type", self.cbType)
        form.addRow("Name", self.edName)
        form.addRow("Key String", self.edKey)
        self.lblKey = form.labelForField(self.edKey)
        form.addRow("Key Times", self.spTimes)
        self.lblTimes = form.labelForField(self.spTimes)
        form.addRow("Hold ms", self.spHold)
        self.lblHold = form.labelForField(self.spHold)

        # 클릭 좌표 / 드래그 시작 좌표 + 픽커 버튼들
        from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QWidget
        rowClickWidget = QWidget()
        rowClick = QHBoxLayout(rowClickWidget); rowClick.setContentsMargins(0, 0, 0, 0)
        rowClick.addWidget(self.spClickX)
        rowClick.addWidget(self.spClickY)
        self.btnPickClick = QPushButton("클릭 좌표 선택")
        self.btnPickDragFrom = QPushButton("드래그 시작 선택")
        rowClick.addWidget(self.btnPickClick)
        rowClick.addWidget(self.btnPickDragFrom)
        form.addRow("Click x / y (또는 Drag 시작)", rowClickWidget)
        self.rowClick = rowClickWidget
        self.lblClick = form.labelForField(rowClickWidget)

        # 드래그 끝 좌표(또는 스크롤 dx/dy) + 픽커 버튼
        rowDestWidget = QWidget()
        rowDest = QHBoxLayout(rowDestWidget); rowDest.setContentsMargins(0, 0, 0, 0)
        rowDest.addWidget(self.spDx)
        rowDest.addWidget(self.spDy)
        self.btnPickDragTo = QPushButton("드래그 끝 선택")
        rowDest.addWidget(self.btnPickDragTo)
        form.addRow("Drag 끝 x / y (또는 Scroll dx / dy)", rowDestWidget)
        self.rowDest = rowDestWidget
        self.lblDest = form.labelForField(rowDestWidget)

        form.addRow("Drag duration ms", self.spDragDur)
        self.lblDragDur = form.labelForField(self.spDragDur)

        # 나머지 스크롤/버튼 설정
        rowScrollWidget = QWidget()
        rowScroll = QHBoxLayout(rowScrollWidget); rowScroll.setContentsMargins(0, 0, 0, 0)
        rowScroll.addWidget(self.spST)
        rowScroll.addWidget(self.spSI)
        form.addRow("Scroll times / interval", rowScrollWidget)
        self.rowScroll = rowScrollWidget
        self.lblScroll = form.labelForField(rowScrollWidget)
        form.addRow("Button", self.edBtn)
        self.lblBtn = form.labelForField(self.edBtn)

        # Type 변경 시 필드 표시 제어
        self.cbType.currentTextChanged.connect(self._on_type_changed)
        self._on_type_changed(self.cbType.currentText())

        # OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        form.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # 좌표 픽커 버튼들 연결: 선택 즉시 필드 반영
        self.btnPickClick.clicked.connect(
            lambda: self._pick_point_into(self.spClickX, self.spClickY, "클릭 좌표", self.btnPickClick)
        )
        self.btnPickDragFrom.clicked.connect(
            lambda: self._pick_point_into(self.spClickX, self.spClickY, "드래그 시작", self.btnPickDragFrom, update_dest=False)
        )
        self.btnPickDragTo.clicked.connect(
            lambda: self._pick_point_into(self.spDx, self.spDy, "드래그 끝", self.btnPickDragTo)
        )

    def _on_type_changed(self, t: str):
        show_key = t.startswith("key")
        self.edKey.setVisible(show_key)
        self.lblKey.setVisible(show_key)

        self.spTimes.setVisible(t == "key")
        self.lblTimes.setVisible(t == "key")
        self.spHold.setVisible(t == "key_hold")
        self.lblHold.setVisible(t == "key_hold")

        show_click = t in ("click_point", "drag")
        self.rowClick.setVisible(show_click)
        self.lblClick.setVisible(show_click)
        self.btnPickClick.setVisible(show_click)
        self.btnPickDragFrom.setVisible(t == "drag")

        show_dest = t in ("drag", "scroll")
        self.rowDest.setVisible(show_dest)
        self.lblDest.setVisible(show_dest)
        self.btnPickDragTo.setVisible(t == "drag")

        show_drag_dur = t == "drag"
        self.spDragDur.setVisible(show_drag_dur)
        self.lblDragDur.setVisible(show_drag_dur)

        show_scroll = t == "scroll"
        self.rowScroll.setVisible(show_scroll)
        self.lblScroll.setVisible(show_scroll)

        show_btn = t in ("click_point", "drag")
        self.edBtn.setVisible(show_btn)
        self.lblBtn.setVisible(show_btn)

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
            if self.isVisible():
                self._robust_restore_self()  # 혹시 남은 비활성/숨김 상태를 한 번 더 복구
        except Exception:
            pass

    # ✔ OK
    def accept(self):
        self._accepted = True
        self._finalize_close()
        super().accept()

    # ✔ Cancel
    def reject(self):
        self._accepted = False
        self._finalize_close()
        super().reject()
        
    # === DROP-IN REPLACEMENT: 클릭 좌표 선택(오버레이 보장) ===
    def _pick_point_into(self, sp_x, sp_y, label: str, source_btn=None, update_dest: bool = True):
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
            res = safe_select_point(None)
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
                if update_dest and not (sp_x is self.spDx and sp_y is self.spDy):
                    if hasattr(self.spDx, "blockSignals"): self.spDx.blockSignals(True)
                    if hasattr(self.spDy, "blockSignals"): self.spDy.blockSignals(True)
                    self.spDx.setValue(int(x))
                    self.spDy.setValue(int(y))
            finally:
                if hasattr(sp_x, "blockSignals"): sp_x.blockSignals(False)
                if hasattr(sp_y, "blockSignals"): sp_y.blockSignals(False)
                if update_dest and not (sp_x is self.spDx and sp_y is self.spDy):
                    if hasattr(self.spDx, "blockSignals"): self.spDx.blockSignals(False)
                    if hasattr(self.spDy, "blockSignals"): self.spDy.blockSignals(False)

            NotImageDialog._last_point = (int(x), int(y))

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
        if not self._accepted:
            return None
        t = self.cbType.currentText().strip()
        name = self.edName.text().strip() or t
        key_string = self.edKey.text().strip()
        if t in ("key", "key_down", "key_up", "key_hold"):
            if not key_string:
                QMessageBox.warning(self, "키 입력 오류", "키 문자열이 비어 있습니다.")
                return None
            if t == "key":
                return StepData(id=str(uuid.uuid4())[:8], name=name, type="key",
                                key_string=key_string, key_times=self.spTimes.value())
            if t == "key_down":
                return StepData(id=str(uuid.uuid4())[:8], name=name, type="key_down",
                                key_string=key_string)
            if t == "key_up":
                return StepData(id=str(uuid.uuid4())[:8], name=name, type="key_up",
                                key_string=key_string)
            if t == "key_hold":
                return StepData(id=str(uuid.uuid4())[:8], name=name, type="key_hold",
                                key_string=key_string, hold_ms=self.spHold.value())
        if t == "click_point":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="click_point",
                            click_button=self.edBtn.text().strip() or "left",
                            click_x=self.spClickX.value(), click_y=self.spClickY.value())
        if t == "drag":
            return StepData(id=str(uuid.uuid4())[:8], name=name, type="drag",
                            drag_from_x=self.spClickX.value(), drag_from_y=self.spClickY.value(),
                            drag_to_x=self.spDx.value(), drag_to_y=self.spDy.value(),
                            drag_duration_ms=max(1, self.spDragDur.value()),
                            click_button=self.edBtn.text().strip() or "left")
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
            if self.isVisible():
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
