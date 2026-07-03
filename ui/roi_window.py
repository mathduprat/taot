"""
ROI selection dialogs.

- RectRoiDialog    : rectangular zone for the intrusion detector
- TrackerRoiDialog : polygon or circle zone for the global tracker, shape chosen in-dialog
"""

import math
import cv2
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QPushButton, QSizePolicy, QRadioButton, QStackedWidget, QWidget
)
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage

HANDLE_HALF  = 5    # half-size of corner resize handles (px)
HANDLE_HIT   = 10   # click tolerance around handles
POINT_RADIUS = 6    # polygon point marker radius
POINT_HIT    = 12   # click tolerance around polygon points

GREEN  = QColor(0, 230, 80)
BLUE   = QColor(50, 150, 255)
BLACK  = QColor(0, 0, 0)
BG     = QColor(30, 30, 30)

# 4-corner handle layout:  0 ── 1
#                          │    │
#                          2 ── 3
HANDLE_MOVES_LEFT   = {0, 2}
HANDLE_MOVES_RIGHT  = {1, 3}
HANDLE_MOVES_TOP    = {0, 1}
HANDLE_MOVES_BOTTOM = {2, 3}

HANDLE_CURSORS = [
    Qt.CursorShape.SizeFDiagCursor,  # 0 top-left
    Qt.CursorShape.SizeBDiagCursor,  # 1 top-right
    Qt.CursorShape.SizeBDiagCursor,  # 2 bottom-left
    Qt.CursorShape.SizeFDiagCursor,  # 3 bottom-right
]


class _Viewport:
    """
    Shared zoom/pan state and coordinate helpers for both canvas types.

    Subclasses must:
      - call _init_viewport(fw, fh) in __init__
      - set self._frame_pixmap (QPixmap, BGR converted to RGB)
      - implement _repaint() and _update_cursor(pos)
    """

    def _init_viewport(self, frame_width: int, frame_height: int):
        self._frame_width  = frame_width
        self._frame_height = frame_height
        self._scale        = 1.0   # current display scale
        self._offset_x     = 0     # display-space X of the frame top-left corner
        self._offset_y     = 0     # display-space Y of the frame top-left corner
        self._zoom_level   = 1.0
        self._pan_start_pos    = None   # mouse position when right-drag started
        self._pan_start_offset = None   # (offset_x, offset_y) at pan start

    def _fit_frame_to_widget(self):
        """Center and scale the frame to fill the widget at the current zoom level."""
        widget_w, widget_h = self.width(), self.height()
        if not widget_w or not widget_h:
            return
        base_scale    = min(widget_w / self._frame_width, widget_h / self._frame_height)
        self._scale   = base_scale * self._zoom_level
        display_w     = int(self._frame_width  * self._scale)
        display_h     = int(self._frame_height * self._scale)
        self._offset_x = (widget_w - display_w) // 2
        self._offset_y = (widget_h - display_h) // 2

    def _clamp_pan_to_frame(self):
        """Prevent panning so far that the frame disappears off screen (40 px margin)."""
        widget_w, widget_h = self.width(), self.height()
        margin    = 40
        display_w = self._frame_width  * self._scale
        display_h = self._frame_height * self._scale
        min_offset_x = -(display_w - margin)
        max_offset_x = widget_w - margin
        min_offset_y = -(display_h - margin)
        max_offset_y = widget_h - margin
        offset_x_clamped = max(min_offset_x, min(self._offset_x, max_offset_x))
        offset_y_clamped = max(min_offset_y, min(self._offset_y, max_offset_y))
        self._offset_x = int(offset_x_clamped)
        self._offset_y = int(offset_y_clamped)

    def _frame_to_display(self, frame_x, frame_y) -> tuple[int, int]:
        return int(frame_x * self._scale + self._offset_x), int(frame_y * self._scale + self._offset_y)

    def _display_to_frame(self, display_x, display_y) -> tuple[int, int]:
        return int((display_x - self._offset_x) / self._scale), int((display_y - self._offset_y) / self._scale)

    def _draw_frame_pixmap(self, painter: QPainter):
        """Paint the video frame onto an existing painter at the current zoom/pan offset."""
        display_w = max(1, int(self._frame_width  * self._scale))
        display_h = max(1, int(self._frame_height * self._scale))
        scaled = self._frame_pixmap.scaled(display_w, display_h,
                                           Qt.AspectRatioMode.IgnoreAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(self._offset_x, self._offset_y, scaled)

    def wheelEvent(self, event):
        # Keep the frame pixel under the cursor fixed while zooming.
        old_scale   = self._scale
        if event.angleDelta().y() > 0:
            zoom_factor = 1.2
        else:
            zoom_factor = 1 / 1.2
        widget_w, widget_h = self.width(), self.height()
        base_scale     = min(widget_w / self._frame_width, widget_h / self._frame_height)
        self._zoom_level = max(1.0, min(self._zoom_level * zoom_factor, 16.0))
        new_scale      = base_scale * self._zoom_level
        mouse_x        = int(event.position().x())
        mouse_y        = int(event.position().y())
        frame_x_under_cursor = (mouse_x - self._offset_x) / old_scale
        frame_y_under_cursor = (mouse_y - self._offset_y) / old_scale
        self._scale    = new_scale
        self._offset_x = int(mouse_x - frame_x_under_cursor * new_scale)
        self._offset_y = int(mouse_y - frame_y_under_cursor * new_scale)
        self._clamp_pan_to_frame()
        self._repaint()

    def _start_pan(self, pos):
        self._pan_start_pos    = pos
        self._pan_start_offset = (self._offset_x, self._offset_y)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _continue_pan(self, pos):
        self._offset_x = self._pan_start_offset[0] + pos.x() - self._pan_start_pos.x()
        self._offset_y = self._pan_start_offset[1] + pos.y() - self._pan_start_pos.y()
        self._clamp_pan_to_frame()
        self._repaint()

    def _stop_pan(self, pos):
        self._pan_start_pos = None
        self._update_cursor(pos)

    def _is_panning(self) -> bool:
        return self._pan_start_pos is not None


class _RectCanvas(_Viewport, QLabel):
    """
    Interactive canvas for drawing, moving and resizing a rectangular ROI.
    - Left click + drag on empty area : draw a new rectangle
    - Left click + drag inside rect   : move the rectangle
    - Left click + drag on a corner   : resize
    - Right click + drag              : pan
    - Scroll wheel                    : zoom
    """

    roi_changed = pyqtSignal(int, int, int, int, float)   # x, y, w, h in frame coords, avg_gray

    def __init__(self, frame, parent=None):
        QLabel.__init__(self, parent)
        frame_h, frame_w = frame.shape[:2]
        self._init_viewport(frame_w, frame_h)

        self._gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_pixmap = QPixmap.fromImage(
            QImage(rgb.data, frame_w, frame_h, frame_w * 3, QImage.Format.Format_RGB888)
        )

        self._roi: QRect | None        = None
        self._drag_mode                = None    # None | 'draw' | 'move' | int (corner index)
        self._drag_origin: QPoint | None = None  # mouse position at drag start
        self._roi_snapshot: QRect | None = None  # copy of roi at drag start

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 400)

    def _roi_in_display_coords(self) -> QRect:
        r  = self._roi.normalized()
        x1, y1 = self._frame_to_display(r.left(), r.top())
        x2, y2 = self._frame_to_display(r.right(), r.bottom())
        return QRect(QPoint(x1, y1), QPoint(x2, y2))

    def _corner_centers(self) -> list[tuple[int, int]]:
        d = self._roi_in_display_coords()
        return [
            (d.left(),  d.top()),     # 0  top-left
            (d.right(), d.top()),     # 1  top-right
            (d.left(),  d.bottom()),  # 2  bottom-left
            (d.right(), d.bottom()),  # 3  bottom-right
        ]

    def _corner_at_pos(self, pos) -> int | None:
        """Return the index of the corner handle under pos, or None."""
        for index, (cx, cy) in enumerate(self._corner_centers()):
            if abs(pos.x() - cx) <= HANDLE_HIT and abs(pos.y() - cy) <= HANDLE_HIT:
                return index
        return None

    def _clamp_roi_to_frame(self):
        """Keep the roi within frame boundaries."""
        if not self._roi:
            return
        r = self._roi.normalized()
        self._roi = QRect(
            QPoint(max(0, min(r.x(), self._frame_width - 1)),
                   max(0, min(r.y(), self._frame_height - 1))),
            QPoint(max(0, min(r.right(), self._frame_width - 1)),
                   max(0, min(r.bottom(), self._frame_height - 1)))
        )

    def _compute_avg_gray(self, x, y, w, h) -> float:
        region = self._gray_frame[y:y + h, x:x + w]
        if region.size > 0:
            return float(cv2.mean(region)[0])
        return 0.0

    def resizeEvent(self, _):
        self._fit_frame_to_widget()
        self._repaint()

    def _repaint(self):
        widget_w, widget_h = self.width(), self.height()
        if not widget_w or not widget_h:
            return
        pixmap = QPixmap(widget_w, widget_h)
        pixmap.fill(BG)
        painter = QPainter(pixmap)
        self._draw_frame_pixmap(painter)

        if self._roi and self._roi.normalized().width() > 0:
            roi_disp = self._roi_in_display_coords()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(GREEN, 2))
            painter.drawRect(roi_disp)
            # Draw 4 corner handles
            painter.setBrush(QBrush(GREEN))
            painter.setPen(QPen(BLACK, 1))
            for cx, cy in self._corner_centers():
                painter.drawRect(cx - HANDLE_HALF, cy - HANDLE_HALF, HANDLE_HALF*2, HANDLE_HALF*2)

        painter.end()
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.MouseButton.RightButton:
            self._start_pan(pos)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._roi:
            corner = self._corner_at_pos(pos)
            if corner is not None:
                self._drag_mode, self._drag_origin, self._roi_snapshot = corner, pos, QRect(self._roi)
                return
            if self._roi_in_display_coords().contains(pos):
                self._drag_mode, self._drag_origin, self._roi_snapshot = 'move', pos, QRect(self._roi)
                return

        # Click outside existing roi → start drawing a new one
        frame_x, frame_y = self._display_to_frame(pos.x(), pos.y())
        self._roi = QRect(frame_x, frame_y, 0, 0)
        self._drag_mode, self._drag_origin, self._roi_snapshot = 'draw', pos, QRect(self._roi)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._is_panning():
            self._continue_pan(pos)
            return
        if self._drag_mode is None:
            self._update_cursor(pos)
            return

        # Convert mouse delta from display pixels to frame pixels
        delta_frame_x = int((pos.x() - self._drag_origin.x()) / self._scale)
        delta_frame_y = int((pos.y() - self._drag_origin.y()) / self._scale)
        snap = self._roi_snapshot

        if self._drag_mode == 'draw':
            fx, fy = self._display_to_frame(pos.x(), pos.y())
            ox, oy = self._display_to_frame(self._drag_origin.x(), self._drag_origin.y())
            self._roi = QRect(QPoint(ox, oy), QPoint(fx, fy)).normalized()

        elif self._drag_mode == 'move':
            self._roi = QRect(snap.x() + delta_frame_x, snap.y() + delta_frame_y,
                              snap.width(), snap.height())

        else:
            # Resize: only move the edges linked to the active corner
            left   = snap.left()
            top    = snap.top()
            right  = snap.right()
            bottom = snap.bottom()
            corner = self._drag_mode
            if corner in HANDLE_MOVES_LEFT:
                left += delta_frame_x
            if corner in HANDLE_MOVES_RIGHT:
                right += delta_frame_x
            if corner in HANDLE_MOVES_TOP:
                top += delta_frame_y
            if corner in HANDLE_MOVES_BOTTOM:
                bottom += delta_frame_y
            self._roi = QRect(QPoint(left, top), QPoint(right, bottom)).normalized()

        self._clamp_roi_to_frame()
        self._repaint()
        if self._roi:
            r = self._roi.normalized()
            self.roi_changed.emit(r.x(), r.y(), r.width(), r.height(),
                                  self._compute_avg_gray(r.x(), r.y(), r.width(), r.height()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._stop_pan(event.pos())
            return
        self._drag_mode = None
        self._repaint()

    def _update_cursor(self, pos):
        if self._roi:
            corner = self._corner_at_pos(pos)
            if corner is not None:
                self.setCursor(HANDLE_CURSORS[corner])
                return
            if self._roi_in_display_coords().contains(pos):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_roi(self, x, y, w, h):
        self._roi = QRect(x, y, w, h)
        self._clamp_roi_to_frame()
        self._repaint()

    def get_roi(self):
        if self._roi and self._roi.normalized().width() > 0:
            r = self._roi.normalized()
            return r.x(), r.y(), r.width(), r.height()
        return None


class _PolyCanvas(_Viewport, QLabel):
    """
    Canvas for clicking 4 points that define the tracker polygon ROI.
    - Left click : add a point (up to 4); use Reset if a mistake is made
    - Right click + drag : pan
    - Scroll wheel       : zoom
    """

    points_changed = pyqtSignal(list)   # list of 4 (x, y) in frame coords, emitted once complete

    def __init__(self, frame, initial_points=None, parent=None):
        QLabel.__init__(self, parent)
        frame_h, frame_w = frame.shape[:2]
        self._init_viewport(frame_w, frame_h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_pixmap = QPixmap.fromImage(
            QImage(rgb.data, frame_w, frame_h, frame_w * 3, QImage.Format.Format_RGB888)
        )

        self._placed_points: list[tuple[int, int]] = list(initial_points) if initial_points else []

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 400)

    def resizeEvent(self, _):
        self._fit_frame_to_widget()
        self._repaint()

    def _repaint(self):
        widget_w, widget_h = self.width(), self.height()
        if not widget_w or not widget_h:
            return
        pixmap = QPixmap(widget_w, widget_h)
        pixmap.fill(BG)
        painter = QPainter(pixmap)
        self._draw_frame_pixmap(painter)

        if self._placed_points:
            display_pts = [self._frame_to_display(fx, fy) for fx, fy in self._placed_points]

            # Draw edges between consecutive points
            painter.setPen(QPen(BLUE, 2))
            for i in range(len(display_pts) - 1):
                painter.drawLine(*display_pts[i], *display_pts[i + 1])
            # Close the polygon once all 4 points are placed
            if len(display_pts) == 4:
                painter.drawLine(*display_pts[3], *display_pts[0])

            # Draw a dot for each point
            painter.setBrush(QBrush(BLUE))
            painter.setPen(QPen(BLACK, 1))
            for dx, dy in display_pts:
                painter.drawEllipse(dx - POINT_RADIUS, dy - POINT_RADIUS,
                                    POINT_RADIUS * 2, POINT_RADIUS * 2)

        painter.end()
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._start_pan(event.pos())
            return
        if event.button() == Qt.MouseButton.LeftButton and len(self._placed_points) < 4:
            frame_x, frame_y = self._display_to_frame(event.pos().x(), event.pos().y())
            # Clamp to frame bounds
            frame_x = max(0, min(frame_x, self._frame_width - 1))
            frame_y = max(0, min(frame_y, self._frame_height - 1))
            self._placed_points.append((frame_x, frame_y))
            if len(self._placed_points) == 4:
                self.points_changed.emit(list(self._placed_points))
            self._repaint()

    def mouseMoveEvent(self, event):
        if self._is_panning():
            self._continue_pan(event.pos())
            return
        self._update_cursor(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._stop_pan(event.pos())

    def _update_cursor(self, _):
        if len(self._placed_points) < 4:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def clear_points(self):
        self._placed_points.clear()
        self._repaint()

    def get_points(self):
        return list(self._placed_points) if len(self._placed_points) == 4 else None


class _CircleCanvas(_Viewport, QLabel):
    """
    Interactive canvas for drawing, moving and resizing a circular ROI.
    - Left click + drag on empty area  : draw a new circle (click = center, drag = radius)
    - Left click + drag inside circle  : move the circle
    - Left click + drag on the handle  : resize the radius
    - Right click + drag               : pan
    - Scroll wheel                     : zoom
    """

    roi_changed = pyqtSignal(int, int, int)   # cx, cy, r in frame coords

    def __init__(self, frame, parent=None):
        QLabel.__init__(self, parent)
        frame_h, frame_w = frame.shape[:2]
        self._init_viewport(frame_w, frame_h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_pixmap = QPixmap.fromImage(
            QImage(rgb.data, frame_w, frame_h, frame_w * 3, QImage.Format.Format_RGB888)
        )

        self._cx: int | None = None
        self._cy: int | None = None
        self._r: int         = 0
        self._drag_mode                  = None    # None | 'draw' | 'move' | 'resize'
        self._drag_origin: QPoint | None = None    # mouse position at drag start
        self._roi_snapshot               = None    # (cx, cy, r) at drag start

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 400)

    def _has_circle(self) -> bool:
        return self._cx is not None and self._r > 0

    def _handle_center(self) -> tuple[int, int]:
        """Resize handle, placed on the circle's rim to the right of center."""
        return self._frame_to_display(self._cx + self._r, self._cy)

    def _clamp_circle_to_frame(self):
        """Keep the circle's center within frame boundaries."""
        if self._cx is None:
            return
        self._cx = max(0, min(self._cx, self._frame_width - 1))
        self._cy = max(0, min(self._cy, self._frame_height - 1))
        self._r  = max(0, self._r)

    def resizeEvent(self, _):
        self._fit_frame_to_widget()
        self._repaint()

    def _repaint(self):
        widget_w, widget_h = self.width(), self.height()
        if not widget_w or not widget_h:
            return
        pixmap = QPixmap(widget_w, widget_h)
        pixmap.fill(BG)
        painter = QPainter(pixmap)
        self._draw_frame_pixmap(painter)

        if self._has_circle():
            dcx, dcy = self._frame_to_display(self._cx, self._cy)
            dr = int(self._r * self._scale)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(GREEN, 2))
            painter.drawEllipse(QPoint(dcx, dcy), dr, dr)

            # Center dot + resize handle on the rim.
            painter.setBrush(QBrush(GREEN))
            painter.setPen(QPen(BLACK, 1))
            painter.drawEllipse(QPoint(dcx, dcy), 3, 3)
            hx, hy = self._handle_center()
            painter.drawRect(hx - HANDLE_HALF, hy - HANDLE_HALF, HANDLE_HALF * 2, HANDLE_HALF * 2)

        painter.end()
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.MouseButton.RightButton:
            self._start_pan(pos)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._has_circle():
            hx, hy = self._handle_center()
            if abs(pos.x() - hx) <= HANDLE_HIT and abs(pos.y() - hy) <= HANDLE_HIT:
                self._drag_mode = 'resize'
                self._drag_origin = pos
                self._roi_snapshot = (self._cx, self._cy, self._r)
                return
            dcx, dcy = self._frame_to_display(self._cx, self._cy)
            if math.hypot(pos.x() - dcx, pos.y() - dcy) <= self._r * self._scale:
                self._drag_mode = 'move'
                self._drag_origin = pos
                self._roi_snapshot = (self._cx, self._cy, self._r)
                return

        # Click outside existing circle → start drawing a new one, centered on the click.
        frame_x, frame_y = self._display_to_frame(pos.x(), pos.y())
        self._cx, self._cy, self._r = frame_x, frame_y, 0
        self._drag_mode = 'draw'
        self._drag_origin = pos
        self._roi_snapshot = (self._cx, self._cy, self._r)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._is_panning():
            self._continue_pan(pos)
            return
        if self._drag_mode is None:
            self._update_cursor(pos)
            return

        snap_cx, snap_cy, snap_r = self._roi_snapshot

        if self._drag_mode == 'draw':
            fx, fy = self._display_to_frame(pos.x(), pos.y())
            self._r = int(math.hypot(fx - self._cx, fy - self._cy))

        elif self._drag_mode == 'move':
            delta_frame_x = int((pos.x() - self._drag_origin.x()) / self._scale)
            delta_frame_y = int((pos.y() - self._drag_origin.y()) / self._scale)
            self._cx = snap_cx + delta_frame_x
            self._cy = snap_cy + delta_frame_y

        else:  # resize
            fx, fy = self._display_to_frame(pos.x(), pos.y())
            self._r = int(math.hypot(fx - snap_cx, fy - snap_cy))

        self._clamp_circle_to_frame()
        self._repaint()
        self.roi_changed.emit(self._cx, self._cy, self._r)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._stop_pan(event.pos())
            return
        self._drag_mode = None
        self._repaint()

    def _update_cursor(self, pos):
        if self._has_circle():
            hx, hy = self._handle_center()
            if abs(pos.x() - hx) <= HANDLE_HIT and abs(pos.y() - hy) <= HANDLE_HIT:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
            dcx, dcy = self._frame_to_display(self._cx, self._cy)
            if math.hypot(pos.x() - dcx, pos.y() - dcy) <= self._r * self._scale:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_roi(self, cx, cy, r):
        self._cx, self._cy, self._r = cx, cy, r
        self._clamp_circle_to_frame()
        self._repaint()

    def get_roi(self):
        if self._has_circle():
            return self._cx, self._cy, self._r
        return None


class RectRoiDialog(QDialog):
    """
    Dialog for rectangular ROI selection (intrusion detector zone).

    Usage:
        dlg = RectRoiDialog(frame, initial_roi=(x, y, w, h), initial_mode="intrusion", parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            x, y, w, h = dlg.get_roi()
            mode = dlg.get_mode()   # "intrusion" or "light"
    """

    def __init__(self, frame, initial_roi=None, initial_mode="intrusion", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Intrusion ROI  —  scroll: zoom  |  right-drag: pan")
        self.resize(960, 680)
        layout = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Detection mode:"))
        self._radio_intrusion = QRadioButton("Intrusion")
        self._radio_intrusion.setToolTip(
            "Rejects drops so large the ROI is fully covered (e.g. an animal passing\n"
            "over the biberon rather than drinking)."
        )
        self._radio_light = QRadioButton("Light detection")
        self._radio_light.setToolTip(
            "No upper bound on the gray drop: any large enough drop counts as presence,\n"
            "including full ROI coverage. Catches more, at the cost of more false positives."
        )
        mode_row.addWidget(self._radio_intrusion)
        mode_row.addWidget(self._radio_light)
        mode_row.addStretch()
        layout.addLayout(mode_row)
        if initial_mode == "light":
            self._radio_light.setChecked(True)
        else:
            self._radio_intrusion.setChecked(True)

        self._canvas = _RectCanvas(frame, self)
        self._canvas.roi_changed.connect(self._update_spinboxes)
        layout.addWidget(self._canvas, 1)

        # Spinboxes for pixel-precise coordinate input
        spin_row = QHBoxLayout()
        self._spinboxes: dict[str, QSpinBox] = {}
        for key, label, max_val in [('x', 'X:', frame.shape[1]), ('y', 'Y:', frame.shape[0]),
                                     ('w', 'W:', frame.shape[1]), ('h', 'H:', frame.shape[0])]:
            spin_row.addWidget(QLabel(label))
            spinbox = QSpinBox()
            spinbox.setRange(0, max_val)
            spinbox.setFixedWidth(80)
            spinbox.valueChanged.connect(self._update_canvas)
            spin_row.addWidget(spinbox)
            self._spinboxes[key] = spinbox
            spin_row.addSpacing(8)
        spin_row.addSpacing(16)
        self._avg_gray_label = QLabel("Avg Gray: —")
        spin_row.addWidget(self._avg_gray_label)
        spin_row.addStretch()
        layout.addLayout(spin_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Guard flag to avoid infinite loop: spinbox → canvas → spinbox → ...
        self._updating = False
        if initial_roi:
            self._set_spinboxes(*initial_roi)
            self._canvas.set_roi(*initial_roi)

    def _update_spinboxes(self, x, y, w, h, avg_gray):
        """Called when the user draws/moves/resizes on the canvas."""
        self._set_spinboxes(x, y, w, h)
        self._avg_gray_label.setText(f"Avg Gray: {avg_gray:.1f}")

    def _set_spinboxes(self, x, y, w, h):
        self._updating = True
        for key, value in zip(('x', 'y', 'w', 'h'), (x, y, w, h)):
            self._spinboxes[key].setValue(value)
        self._updating = False

    def _update_canvas(self):
        """Called when the user edits a spinbox."""
        if not self._updating:
            x_val = self._spinboxes['x'].value()
            y_val = self._spinboxes['y'].value()
            w_val = self._spinboxes['w'].value()
            h_val = self._spinboxes['h'].value()
            self._canvas.set_roi(x_val, y_val, w_val, h_val)

    def get_roi(self):
        return self._canvas.get_roi()

    def get_mode(self):
        return "light" if self._radio_light.isChecked() else "intrusion"


class _PolyRoiPage(QWidget):
    """Polygon canvas + its Reset/status controls, embedded as one page of TrackerRoiDialog."""

    def __init__(self, frame, initial_points=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = _PolyCanvas(frame, initial_points=initial_points, parent=self)
        self._canvas.points_changed.connect(self._on_four_points_placed)
        layout.addWidget(self._canvas, 1)

        info_row = QHBoxLayout()
        self._status_label = QLabel("Click to place 4 points. Use Reset if you make a mistake.")
        info_row.addWidget(self._status_label)
        info_row.addStretch()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset_points)
        info_row.addWidget(reset_btn)
        layout.addLayout(info_row)

    def _on_four_points_placed(self, _):
        self._status_label.setText("4 points placed — click OK to confirm.")

    def _on_reset_points(self):
        self._canvas.clear_points()
        self._status_label.setText("Click to place 4 points. Use Reset if you make a mistake.")

    def get_points(self):
        return self._canvas.get_points()


class _CircleRoiPage(QWidget):
    """Circle canvas + its center/radius spinboxes, embedded as one page of TrackerRoiDialog."""

    def __init__(self, frame, initial_circle=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = _CircleCanvas(frame, self)
        self._canvas.roi_changed.connect(self._update_spinboxes)
        layout.addWidget(self._canvas, 1)

        frame_h, frame_w = frame.shape[:2]
        max_radius = int(math.hypot(frame_w, frame_h))

        spin_row = QHBoxLayout()
        self._spinboxes: dict[str, QSpinBox] = {}
        for key, label, max_val in [('cx', 'Center X:', frame_w), ('cy', 'Center Y:', frame_h),
                                     ('r', 'Radius:', max_radius)]:
            spin_row.addWidget(QLabel(label))
            spinbox = QSpinBox()
            spinbox.setRange(0, max_val)
            spinbox.setFixedWidth(80)
            spinbox.valueChanged.connect(self._update_canvas)
            spin_row.addWidget(spinbox)
            self._spinboxes[key] = spinbox
            spin_row.addSpacing(8)
        spin_row.addStretch()
        layout.addLayout(spin_row)

        # Guard flag to avoid infinite loop: spinbox → canvas → spinbox → ...
        self._updating = False
        if initial_circle:
            self._set_spinboxes(*initial_circle)
            self._canvas.set_roi(*initial_circle)

    def _update_spinboxes(self, cx, cy, r):
        """Called when the user draws/moves/resizes on the canvas."""
        self._set_spinboxes(cx, cy, r)

    def _set_spinboxes(self, cx, cy, r):
        self._updating = True
        for key, value in zip(('cx', 'cy', 'r'), (cx, cy, r)):
            self._spinboxes[key].setValue(value)
        self._updating = False

    def _update_canvas(self):
        """Called when the user edits a spinbox."""
        if not self._updating:
            cx = self._spinboxes['cx'].value()
            cy = self._spinboxes['cy'].value()
            r  = self._spinboxes['r'].value()
            self._canvas.set_roi(cx, cy, r)

    def get_circle(self):
        return self._canvas.get_roi()


class TrackerRoiDialog(QDialog):
    """
    Dialog for the global tracker ROI — shape (polygon or circle) is chosen inside the
    dialog itself via radio buttons, rather than in the main window.

    Usage:
        dlg = TrackerRoiDialog(frame, initial_roi_str="poly:x0,y0,...,x3,y3" or "circle:cx,cy,r",
                                parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            roi_str = dlg.get_roi_str()   # shape-tagged string, e.g. "circle:120,80,40"
    """

    def __init__(self, frame, initial_roi_str="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            "Tracker ROI  —  scroll: zoom  |  right-drag: pan"
        )
        self.resize(960, 720)
        layout = QVBoxLayout(self)

        initial_shape = "poly"
        initial_points = None
        initial_circle = None
        if initial_roi_str:
            shape, coords_part = (initial_roi_str.split(":", 1) + [""])[:2] \
                if ":" in initial_roi_str else ("poly", initial_roi_str)
            coords = [int(c) for c in coords_part.split(",")]
            if shape == "circle" and len(coords) == 3:
                initial_shape = "circle"
                initial_circle = tuple(coords)
            elif len(coords) == 8:
                initial_points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Shape:"))
        self._radio_poly = QRadioButton("Polygon (4 pts)")
        self._radio_circle = QRadioButton("Circle")
        shape_row.addWidget(self._radio_poly)
        shape_row.addWidget(self._radio_circle)
        shape_row.addStretch()
        layout.addLayout(shape_row)

        self._stack = QStackedWidget()
        self._poly_page = _PolyRoiPage(frame, initial_points=initial_points, parent=self)
        self._circle_page = _CircleRoiPage(frame, initial_circle=initial_circle, parent=self)
        self._stack.addWidget(self._poly_page)
        self._stack.addWidget(self._circle_page)
        layout.addWidget(self._stack, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._radio_poly.toggled.connect(self._on_shape_toggled)
        if initial_shape == "circle":
            self._radio_circle.setChecked(True)
        else:
            self._radio_poly.setChecked(True)
        self._on_shape_toggled()

    def _on_shape_toggled(self):
        self._stack.setCurrentWidget(self._circle_page if self._radio_circle.isChecked() else self._poly_page)

    def get_roi_str(self):
        """Return the shape-tagged ROI string for the currently selected shape, or None."""
        if self._radio_circle.isChecked():
            circle = self._circle_page.get_circle()
            if not circle:
                return None
            return "circle:" + ",".join(str(c) for c in circle)

        points = self._poly_page.get_points()
        if not points:
            return None
        return "poly:" + ",".join(str(c) for pt in points for c in pt)
