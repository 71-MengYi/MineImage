"""MineImage 浅色主题：调色板 + QSS。

界面统一走浅色风格（Windows 桌面观感）：白底卡片、细边框、6~8px 圆角，
强调色为蓝色 ``#2563eb``。所有尺寸使用 pt / QSS 像素，随系统 DPI 缩放。
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#2563eb"
ACCENT_HOVER = "#1d4ed8"
ACCENT_SOFT = "#dbeafe"
WINDOW_BG = "#f5f6f8"
SURFACE = "#ffffff"
BORDER = "#d8dee8"
BORDER_SOFT = "#e6ebf2"
TEXT = "#1f2937"
TEXT_MUTED = "#64748b"
TEXT_FAINT = "#94a3b8"
DANGER = "#b91c1c"
SUCCESS = "#15803d"

_PREFERRED_FONTS = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Segoe UI",
    "PingFang SC",
    "Noto Sans CJK SC",
)

_QSS = f"""
/* ---------------------------------------------------------------- 基础 */
QWidget {{
    color: {TEXT};
    font-size: 9.5pt;
}}
QMainWindow, QDialog {{
    background-color: {WINDOW_BG};
}}
QWidget:disabled {{
    color: {TEXT_FAINT};
}}
QToolTip {{
    background-color: #111827;
    color: #f9fafb;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 9pt;
}}
QLabel#sectionTitle {{
    font-size: 12pt;
    font-weight: 600;
    color: #0f172a;
}}
QLabel#hintLabel {{
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QLabel#errorLabel {{
    color: {DANGER};
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 6px 10px;
}}
QLabel#valueLabel {{
    color: #334155;
    background-color: #f1f5f9;
    border: 1px solid {BORDER_SOFT};
    border-radius: 4px;
    padding: 3px 6px;
}}
QLabel#liveBadge {{
    border-radius: 9px;
    padding: 2px 9px;
    font-size: 8.5pt;
    font-weight: 600;
}}
QLabel#liveBadge[state="ok"] {{
    color: {SUCCESS};
    background-color: #dcfce7;
    border: 1px solid #bbf7d0;
}}
QLabel#liveBadge[state="error"] {{
    color: {DANGER};
    background-color: #fee2e2;
    border: 1px solid #fecaca;
}}
QLabel[hasError="true"] {{
    color: {DANGER};
}}
QFrame#panelHeader {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 8px;
}}
QFrame#panelFooter {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 8px;
}}
QFrame#libraryInfoPanel {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#libraryInfoPanel QLabel {{
    border: none;
}}
QFrame#card, QFrame#pageCard {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
}}

/* ---------------------------------------------------------------- 菜单/栏 */
QMenuBar {{
    background-color: #fbfcfe;
    border-bottom: 1px solid {BORDER_SOFT};
    padding: 2px 6px;
}}
QMenuBar::item {{
    padding: 5px 10px;
    border-radius: 5px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background-color: {ACCENT_SOFT};
    color: #1e3a8a;
}}
QMenu {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 26px 6px 22px;
    border-radius: 5px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_SOFT};
    color: #1e3a8a;
}}
QMenu::item:disabled {{
    color: {TEXT_FAINT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER_SOFT};
    margin: 4px 8px;
}}
QToolBar {{
    background-color: #fbfcfe;
    border: none;
    border-bottom: 1px solid {BORDER_SOFT};
    spacing: 4px;
    padding: 3px 6px;
}}
QToolBar::separator {{
    width: 1px;
    background: {BORDER_SOFT};
    margin: 4px 6px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{
    background-color: #eef2f8;
    border-color: {BORDER_SOFT};
}}
QToolButton:pressed {{
    background-color: #e2e8f0;
}}
/* 一级分类胶囊：横向并排、可换行；选中态用强调色。 */
QToolButton#categoryChip {{
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid {BORDER_SOFT};
    border-radius: 11px;
    padding: 4px 9px;
    min-height: 18px;
    font-size: 9pt;
    text-align: left;
}}
QToolButton#categoryChip:hover {{
    background-color: #e8eef8;
    color: #1e3a8a;
    border-color: #cbd5e1;
}}
QToolButton#categoryChip:checked {{
    background-color: {ACCENT_SOFT};
    color: #1e3a8a;
    border-color: #93c5fd;
    font-weight: 600;
}}
QToolButton#categoryChip:checked:hover {{
    background-color: #cfe2fd;
    border-color: #60a5fa;
}}
QToolButton#treeAction {{
    background: transparent;
    color: {TEXT_MUTED};
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 8.5pt;
}}
QToolButton#treeAction:hover {{
    background-color: #eef2f8;
    color: {ACCENT};
    border-color: {BORDER_SOFT};
}}
QToolButton#treeAction:disabled {{
    color: {TEXT_FAINT};
    background: transparent;
}}
QScrollArea#categoryBar {{
    background: transparent;
    border: none;
}}
QStatusBar {{
    background-color: #fbfcfe;
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER_SOFT};
}}
QStatusBar::item {{
    border: none;
}}
QStatusBar QLabel {{
    color: {TEXT_MUTED};
    padding: 0 6px;
}}
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background-color: #f1f5f9;
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER_SOFT};
    font-weight: 600;
}}

/* ---------------------------------------------------------------- 分割器 */
QSplitter::handle {{
    background-color: {BORDER_SOFT};
}}
QSplitter::handle:horizontal {{
    width: 6px;
    margin: 2px 1px;
}}
QSplitter::handle:vertical {{
    height: 6px;
    margin: 1px 2px;
}}
QSplitter::handle:hover {{
    background-color: #c7d6ea;
}}

/* ---------------------------------------------------------------- 标签页 */
QTabWidget::pane {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
}}
QTabWidget::tab-bar {{
    left: 2px;
}}
QTabBar::tab {{
    background-color: #eef2f8;
    color: #475569;
    border: 1px solid {BORDER_SOFT};
    border-radius: 6px;
    padding: 7px 13px;
    margin: 2px;
    min-width: 64px;
}}
QTabBar::tab:hover {{
    background-color: #e3eaf4;
    color: #1e3a8a;
}}
QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: {ACCENT};
    border-color: {BORDER};
    font-weight: 600;
}}
QTabBar::tab:disabled {{
    color: {TEXT_FAINT};
}}
QTabBar::tab:west {{
    min-width: 88px;
    padding: 9px 7px;
    margin: 3px 2px;
}}
QTabBar::tab:west:selected {{
    border-left: 3px solid {ACCENT};
}}

/* ---------------------------------------------------------------- 树/表 */
QTreeWidget, QTreeView {{
    background-color: transparent;
    border: none;
    show-decoration-selected: 1;
    outline: none;
}}
QTreeWidget::item, QTreeView::item {{
    height: 26px;
    padding: 1px 4px;
    border-radius: 5px;
}}
QTreeWidget::item:hover, QTreeView::item:hover {{
    background-color: #f1f5f9;
}}
QTreeWidget::item:selected, QTreeView::item:selected {{
    background-color: {ACCENT_SOFT};
    color: #1e3a8a;
}}
QTreeWidget::branch {{
    background: transparent;
}}
QTableWidget, QTableView {{
    background-color: {SURFACE};
    alternate-background-color: #f8fafc;
    border: 1px solid {BORDER_SOFT};
    border-radius: 6px;
    gridline-color: #edf1f6;
    selection-background-color: {ACCENT_SOFT};
    selection-color: #1e3a8a;
}}
QTableWidget::item, QTableView::item {{
    padding: 3px 6px;
}}
QHeaderView::section {{
    background-color: #f1f5f9;
    color: #475569;
    border: none;
    border-right: 1px solid {BORDER_SOFT};
    border-bottom: 1px solid {BORDER_SOFT};
    padding: 5px 7px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background-color: #f1f5f9;
    border: none;
}}

/* ---------------------------------------------------------------- 分组框 */
QGroupBox {{
    background-color: #fbfcfe;
    border: 1px solid {BORDER_SOFT};
    border-radius: 8px;
    margin-top: 16px;
    padding: 12px 10px 8px 10px;
    font-weight: 600;
    color: #334155;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 5px;
    color: #334155;
}}
QGroupBox::indicator {{
    width: 14px;
    height: 14px;
}}

/* ---------------------------------------------------------------- 按钮 */
QPushButton {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 15px;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: #f1f5f9;
    border-color: #b9c6d8;
}}
QPushButton:pressed {{
    background-color: #e2e8f0;
}}
QPushButton:focus {{
    border-color: {ACCENT};
}}
QPushButton:disabled {{
    background-color: #f4f6f9;
    border-color: {BORDER_SOFT};
    color: {TEXT_FAINT};
}}
QPushButton#primaryButton {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton#primaryButton:pressed {{
    background-color: #1e40af;
}}
QPushButton#primaryButton:disabled {{
    background-color: #bfd3f6;
    border-color: #bfd3f6;
    color: #eef4ff;
}}
QPushButton#subtleButton {{
    background-color: transparent;
    border-color: transparent;
    color: {TEXT_MUTED};
    padding: 5px 9px;
}}
QPushButton#subtleButton:hover {{
    background-color: #eef2f8;
    color: {TEXT};
}}
QPushButton#linkButton {{
    background: transparent;
    border: none;
    color: {ACCENT};
    padding: 3px 6px;
}}
QPushButton#linkButton:hover {{
    color: {ACCENT_HOVER};
    text-decoration: underline;
}}
QPushButton#dangerButton {{
    background-color: {SURFACE};
    border-color: #f3b7b7;
    color: {DANGER};
}}
QPushButton#dangerButton:hover {{
    background-color: #fef2f2;
}}
QPushButton#colorSwatch {{
    border: 1px solid #94a3b8;
    border-radius: 5px;
    padding: 5px 12px;
    color: #0f172a;
    font-weight: 600;
}}

/* ---------------------------------------------------------------- 输入 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: #bfdbfe;
    selection-color: #0f172a;
    min-height: 20px;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover,
QPlainTextEdit:hover, QTextEdit:hover {{
    border-color: #b9c6d8;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    background-color: #f4f6f9;
    color: {TEXT_FAINT};
}}
QLineEdit[readOnly="true"] {{
    background-color: #f8fafc;
    color: #475569;
}}
QComboBox {{
    padding-right: 24px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    border-left: 1px solid {BORDER_SOFT};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background: transparent;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: #1e3a8a;
    outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 18px;
    background-color: transparent;
    border-left: 1px solid {BORDER_SOFT};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #eef2f8;
}}

/* ---------------------------------------------------------------- 勾选/滑块 */
QCheckBox, QRadioButton {{
    spacing: 6px;
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid #b9c6d8;
    border-radius: 4px;
    background-color: {SURFACE};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: #f1f5f9;
    border-color: {BORDER_SOFT};
}}
QSlider::groove:horizontal {{
    height: 5px;
    background: #e2e8f0;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: {SURFACE};
    border: 2px solid {ACCENT};
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: #eff6ff;
}}

/* ---------------------------------------------------------------- 滚动条 */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #c8d3e0;
    min-height: 26px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: #a9b8cb;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #c8d3e0;
    min-width: 26px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #a9b8cb;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
    background: transparent;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QListWidget, QListView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    outline: none;
}}
QListWidget::item, QListView::item {{
    padding: 5px 7px;
    border-radius: 5px;
}}
QListWidget::item:hover, QListView::item:hover {{
    background-color: #f1f5f9;
}}
QListWidget::item:selected, QListView::item:selected {{
    background-color: {ACCENT_SOFT};
    color: #1e3a8a;
}}
QProgressBar {{
    background-color: #eef2f8;
    border: 1px solid {BORDER_SOFT};
    border-radius: 6px;
    text-align: center;
    color: {TEXT_MUTED};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

/* ---------------------------------------------------------------- 对话框 */
QDialogButtonBox QPushButton {{
    min-width: 84px;
}}
QMessageBox {{
    background-color: {WINDOW_BG};
}}
"""


def _preferred_font() -> QFont:
    families = set(QFontDatabase.families())
    for name in _PREFERRED_FONTS:
        if name in families:
            font = QFont(name)
            font.setPointSizeF(9.5)
            return font
    font = QFont()
    font.setPointSizeF(9.5)
    return font


def _build_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WINDOW_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f9fafb"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Mid, QColor(BORDER))
    palette.setColor(QPalette.ColorRole.Dark, QColor("#94a3b8"))
    palette.setColor(QPalette.ColorRole.Light, QColor("#ffffff"))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_FAINT))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#f4f6f9"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor("#f4f6f9"))
    return palette


def apply_theme(app: QApplication) -> None:
    """应用浅色主题（QSS + 调色板 + 字体）。幂等，可重复调用。"""
    if app is None:  # pragma: no cover - 防御式
        return
    try:
        app.setStyle("Fusion")
    except Exception:  # noqa: BLE001 - 某些平台没有 Fusion 时忽略
        pass
    app.setFont(_preferred_font())
    app.setPalette(_build_palette())
    app.setStyleSheet(_QSS)
    # 属性仅用于诊断，不参与逻辑。
    app.setProperty("mineimageTheme", "light")
