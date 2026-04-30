#!/usr/bin/env python3
"""
Gemini Delegator GUI — PyQt6 desktop interface for gemini_delegator.py
"""

# ── Standard library imports ──────────────────────────────────────────────────
import sys           # lets us exit the app cleanly
from pathlib import Path  # cross-platform file path handling

# ── PyQt6 GUI widgets ─────────────────────────────────────────────────────────
# QMainWindow = the outer app window
# QWidget     = a blank container you can put other things inside
# QVBoxLayout / QHBoxLayout = stack children vertically or horizontally
# QTabWidget  = the tab bar (Research / Code)
# QLabel      = a non-editable text or image display
# QComboBox   = a dropdown menu
# QTextEdit   = a multi-line text box (editable or read-only)
# QPushButton = a clickable button
# QLineEdit   = a single-line text input
# QMessageBox = a pop-up dialog for errors
# QGroupBox   = a titled border box that groups related controls
# QSizePolicy = controls how a widget grows/shrinks when the window resizes
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QComboBox, QTextEdit, QPushButton,
    QLineEdit, QMessageBox, QGroupBox, QSizePolicy, QFrame,
)
# QThread     = runs work in the background so the UI doesn't freeze
# pyqtSignal  = a message a background thread sends back to the UI when done
# Qt          = constants like AlignCenter, etc.
# QTimer      = fires a function on a repeating interval (used for rate limit cooldown)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
# QFont       = lets us set font family and size on a widget
from PyQt6.QtGui import QFont

# ── Backend import ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
# GeminiDelegator  = the class that actually calls the Gemini API
# MODEL_REGISTRY   = dict of model aliases → {id, type, desc}
from gemini_delegator import GeminiDelegator, MODEL_REGISTRY

# ── Tier rankings for the Launch tab ─────────────────────────────────────────
# Ranked by general research + code utility on the free tier.
# Format: alias → (letter, hex_color)
TIER_RANKS = {
    # S — best all-round on free tier
    "gemini25flash":   ("S", "#E53935"),   # vivid red
    "flashlatest":     ("S", "#E53935"),
    # A — excellent for demanding tasks
    "gemma431b":       ("A", "#E67E22"),   # orange
    "gemma327b":       ("A", "#E67E22"),
    "gemma4":          ("A", "#E67E22"),
    # B — solid workhorses
    "gemini3flash":    ("B", "#D4AC0D"),   # gold
    "gemma312b":       ("B", "#D4AC0D"),
    "gemini31lite":    ("B", "#D4AC0D"),
    # C — good for simple tasks
    "gemma34b":        ("C", "#27AE60"),   # green
    "flashlitelatest": ("C", "#27AE60"),
    # D — limited / multimodal niche
    "gemma3ne4b":      ("D", "#2980B9"),   # blue
    "gemma3ne2b":      ("D", "#2980B9"),
    # E — very small, speed-only
    "gemma31b":        ("E", "#8E44AD"),   # purple
    # F — highly specialised (robotics), not general purpose
    "robotics16":      ("F", "#7F8C8D"),   # gray
    "robotics15":      ("F", "#7F8C8D"),
}

# Ordered list for the grid (S first, F last)
TIER_ORDER = [alias for alias in TIER_RANKS]


# ── Model list helpers ────────────────────────────────────────────────────────
def _text_models():
    return [k for k, v in MODEL_REGISTRY.items() if v["type"] == "text"]


def _model_desc(alias):
    """Return the description string for a model alias, or empty string."""
    return MODEL_REGISTRY.get(alias, {}).get("desc", "")


def _short_desc(alias):
    """Return just the first sentence of a model's description for inline display."""
    desc = _model_desc(alias)
    if not desc:
        return ""
    # Take up to first period-space so the item stays short enough to read
    if ". " in desc:
        return desc.split(". ")[0]
    return desc[:55] if len(desc) > 55 else desc


def _make_model_combo(aliases, default):
    """
    Build a QComboBox from a list of model aliases.
    Each item shows "alias — first sentence of description" so you can scan the list
    without clicking. The alias is also stored as hidden UserRole data so the backend
    gets the correct alias regardless of the display text.
    Tooltip shows the full description on hover.
    """
    combo = QComboBox()
    default_idx = 0
    for i, alias in enumerate(aliases):
        short = _short_desc(alias)
        display = f"{alias}  —  {short}" if short else alias
        combo.addItem(display)
        # UserRole stores the raw alias — used when passing model name to the delegator
        combo.setItemData(i, alias, Qt.ItemDataRole.UserRole)
        # ToolTipRole stores the full description for hover
        combo.setItemData(i, _model_desc(alias), Qt.ItemDataRole.ToolTipRole)
        if alias == default:
            default_idx = i
    combo.setCurrentIndex(default_idx)
    return combo


def _make_desc_label(alias):
    """
    Build the small grey description label shown below a model dropdown.
    Updates live as the user changes the selection.
    """
    label = QLabel(_model_desc(alias))
    label.setWordWrap(True)
    label.setStyleSheet("color: #888; font-size: 11px; padding: 2px 0 4px 0;")
    return label


# ── Model card widget ────────────────────────────────────────────────────────
# One clickable card per free model on the Launch tab.
# The large tier letter (S/A/B/C/D/E/F) sits on the left like a game HUD badge.

class ModelCard(QFrame):
    clicked = pyqtSignal(str)  # emits the model alias when clicked

    def __init__(self, alias, meta, parent=None):
        super().__init__(parent)
        self.alias = alias

        tier_letter, tier_color = TIER_RANKS.get(alias, ("?", "#555"))
        self._tier_color = tier_color

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(310)
        # MinimumExpanding lets the card grow taller to fit wrapped text
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)
        self.setMinimumHeight(148)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # ── Outer horizontal layout: [badge] [text content] ──────────────────
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(14)

        # ── Tier badge (big coloured square with letter) ──────────────────────
        badge_frame = QFrame()
        badge_frame.setFixedSize(52, 52)
        badge_frame.setStyleSheet(
            f"background:{tier_color};border-radius:8px;"
        )
        badge_layout = QVBoxLayout(badge_frame)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        tier_lbl = QLabel(tier_letter)
        tier_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ltr_font = QFont()
        ltr_font.setBold(True)
        ltr_font.setPointSize(22)
        tier_lbl.setFont(ltr_font)
        tier_lbl.setStyleSheet("color:white;background:transparent;")
        badge_layout.addWidget(tier_lbl)
        outer.addWidget(badge_frame, 0, Qt.AlignmentFlag.AlignTop)

        # ── Text content ──────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet("background:transparent;")
        col = QVBoxLayout(content)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        name_lbl = QLabel(alias)
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(13)
        name_lbl.setFont(name_font)
        name_lbl.setStyleSheet("color:#f0f0f0;background:transparent;")
        col.addWidget(name_lbl)

        desc_lbl = QLabel(meta.get("desc", ""))
        desc_lbl.setWordWrap(True)
        desc_lbl.setMaximumWidth(218)   # known width → Qt calculates wrap height correctly
        desc_lbl.setStyleSheet("font-size:11px;color:#bbb;background:transparent;")
        col.addWidget(desc_lbl)

        best_lbl = QLabel("Best for: " + meta.get("best_for", "—"))
        best_lbl.setWordWrap(True)
        best_lbl.setMaximumWidth(218)
        best_lbl.setStyleSheet("font-size:10px;color:#888;background:transparent;margin-top:1px;")
        col.addWidget(best_lbl)

        outer.addWidget(content, 1)

        self._apply_style(selected=False)

    def _apply_style(self, selected):
        border = self._tier_color if selected else "#444"
        bg     = "#1c2533" if selected else "#252525"
        self.setStyleSheet(f"""
            ModelCard {{
                border: 2px solid {border};
                border-radius: 8px;
                background: {bg};
            }}
        """)

    def set_selected(self, selected):
        self._apply_style(selected)

    def mousePressEvent(self, event):
        self.clicked.emit(self.alias)
        super().mousePressEvent(event)


# ── Rate limit tracker ────────────────────────────────────────────────────────
# The Gemini API doesn't expose a "remaining quota" endpoint — there's no way
# to ask Google how many requests you have left. What we CAN track locally:
#   • Session count  — how many requests have been sent since the app opened
#   • Cooldown timer — the 4-second rate-limit delay between requests
# Both are tracked here and displayed in the status bar.

class RateLimitTracker:
    """
    Tracks request count and cooldown state for the session.
    The actual rate limit (15 RPM for most models) is enforced inside
    GeminiDelegator._call_gemini_with_retry with a 4-second sleep.
    This class just surfaces what's happening in the UI.
    """
    RPM_LIMIT = 15          # requests per minute — Gemini default for most models
    COOLDOWN_SECONDS = 4    # enforced sleep between requests in the backend

    def __init__(self):
        self.session_count = 0          # total requests sent this session
        self._last_request_time = None  # wall-clock time of most recent request start

    def record_request(self):
        self.session_count += 1
        self._last_request_time = time.time()

    def seconds_since_last(self):
        if self._last_request_time is None:
            return None
        return time.time() - self._last_request_time

    def status_text(self):
        """Short string shown in the stats label."""
        secs = self.seconds_since_last()
        if secs is None:
            cooldown_str = "no requests yet"
        elif secs < self.COOLDOWN_SECONDS:
            remaining = self.COOLDOWN_SECONDS - secs
            cooldown_str = f"cooldown ~{remaining:.0f}s"
        else:
            cooldown_str = "ready"
        return f"Requests this session: {self.session_count}  |  {cooldown_str}  |  Limit: {self.RPM_LIMIT} RPM"


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

class GeminiWorker(QThread):
    text_done = pyqtSignal(str, str)  # (task_type, result_text)
    error     = pyqtSignal(str)

    def __init__(self, delegator, task_type, payload):
        super().__init__()
        self.delegator = delegator
        self.task_type = task_type
        self.payload   = payload

    def run(self):
        try:
            p = self.payload

            if self.task_type == "research":
                result = self.delegator.delegate_research(
                    p["query"], p["level"], p["model"],
                )
                self.text_done.emit("research", result)
            else:  # code
                result = self.delegator.delegate_code(
                    p["request"], p["level"], p["model"],
                )
                self.text_done.emit("code", result)

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class GeminiDelegatorGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Delegator")
        self.resize(1060, 760)
        self.worker = None
        self.tracker = RateLimitTracker()

        try:
            self.delegator = GeminiDelegator()
        except Exception as e:
            QMessageBox.critical(None, "Startup Error",
                                 f"Could not initialize Gemini client:\n{e}")
            sys.exit(1)

        self._init_ui()

        # Refresh the stats label every second so the cooldown countdown ticks live
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(1000)  # every 1000ms = 1 second

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._init_launch_tab()   # first tab — model browser + inline query
        self._init_research_tab()
        self._init_code_tab()

        # ── Elapsed time timer ─────────────────────────────────────────────────
        # Tracks how long requests take (seconds)
        self._elapsed_seconds = 0
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        # ── Stats bar ─────────────────────────────────────────────────────────
        # Persistent label at the bottom of the window showing request count +
        # cooldown state. Uses the status bar slot so it never overlaps content.
        self._stats_label = QLabel(self.tracker.status_text())
        self._stats_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 6px;")
        self.statusBar().addPermanentWidget(self._stats_label)

        # Elapsed time display (right side of status bar)
        self._elapsed_label = QLabel("⏱ 0s")
        self._elapsed_label.setStyleSheet(
            "color: #4FC3F7; font-size: 12px; font-weight: bold; "
            "padding: 2px 10px; min-width: 60px;"
        )
        self.statusBar().addPermanentWidget(self._elapsed_label)

        self.statusBar().showMessage("Ready")

    def _refresh_stats(self):
        """Called every second by the timer to keep the stats label current."""
        self._stats_label.setText(self.tracker.status_text())

    def _tick_elapsed(self):
        """Called every second while a request is running — update the elapsed display."""
        self._elapsed_seconds += 1
        self._elapsed_label.setText(f"⏱ {self._elapsed_seconds}s")


    # ─────────────────────────────────────────────────────────────────────────
    # LAUNCH TAB
    # ─────────────────────────────────────────────────────────────────────────

    def _init_launch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel("Model Launch Pad  —  select any free model to start a query")
        hdr_font = QFont()
        hdr_font.setBold(True)
        hdr_font.setPointSize(13)
        hdr.setFont(hdr_font)
        layout.addWidget(hdr)

        sub = QLabel("FREE models are interactive. PAID / ? require a billing upgrade.")
        sub.setStyleSheet("color:#888;font-size:11px;margin-bottom:4px;")
        layout.addWidget(sub)

        # ── Scrollable card grid ──────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(430)
        scroll.setStyleSheet("QScrollArea{border:none;}")

        cards_container = QWidget()
        self._card_grid = QGridLayout(cards_container)
        self._card_grid.setSpacing(10)
        self._card_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._cards: dict = {}         # alias → ModelCard
        self._selected_card: str = ""  # currently highlighted alias

        # Only show free-tier models, in tier order (S → F) using TIER_ORDER
        cols = 2
        for i, alias in enumerate(TIER_ORDER):
            meta = MODEL_REGISTRY.get(alias)
            if not meta:
                continue
            card = ModelCard(alias, meta)
            card.clicked.connect(self._on_card_clicked)
            self._cards[alias] = card
            self._card_grid.addWidget(card, i // cols, i % cols)

        scroll.setWidget(cards_container)
        layout.addWidget(scroll)

        # ── Query panel (hidden until a card is clicked) ──────────────────────
        self._launch_panel = QWidget()
        panel_layout = QVBoxLayout(self._launch_panel)
        panel_layout.setContentsMargins(0, 4, 0, 0)
        panel_layout.setSpacing(6)

        # Selected model label + task + depth selectors
        ctrl_row = QHBoxLayout()
        self._launch_model_label = QLabel("—")
        sel_font = QFont()
        sel_font.setBold(True)
        sel_font.setPointSize(12)
        self._launch_model_label.setFont(sel_font)
        ctrl_row.addWidget(self._launch_model_label)
        ctrl_row.addSpacing(16)

        ctrl_row.addWidget(QLabel("Task:"))
        self._launch_task = QComboBox()
        self._launch_task.addItems(["Research", "Code"])
        ctrl_row.addWidget(self._launch_task)
        ctrl_row.addSpacing(8)

        ctrl_row.addWidget(QLabel("Depth:"))
        self._launch_depth = QComboBox()
        self._launch_depth.addItems(["Basic", "Intermediate", "Advanced", "Expert"])
        self._launch_depth.setCurrentText("Intermediate")
        ctrl_row.addWidget(self._launch_depth)
        ctrl_row.addStretch()
        panel_layout.addLayout(ctrl_row)

        # Query input
        self._launch_query = QTextEdit()
        self._launch_query.setPlaceholderText("Enter your query here…")
        self._launch_query.setMaximumHeight(80)
        panel_layout.addWidget(self._launch_query)

        self._launch_btn = QPushButton("Submit")
        self._launch_btn.clicked.connect(self._run_launch)
        panel_layout.addWidget(self._launch_btn)

        out_group = QGroupBox("Output  (capped at 2000 chars — full file auto-saved to Desktop)")
        out_layout = QVBoxLayout()
        self._launch_output = QTextEdit()
        self._launch_output.setReadOnly(True)
        out_layout.addWidget(self._launch_output)
        out_group.setLayout(out_layout)
        panel_layout.addWidget(out_group)

        self._launch_panel.setVisible(False)
        layout.addWidget(self._launch_panel)

        self.tabs.addTab(tab, "Launch")

    def _on_card_clicked(self, alias):
        # Deselect the previously highlighted card
        if self._selected_card and self._selected_card in self._cards:
            self._cards[self._selected_card].set_selected(False)

        # Highlight the new card
        self._selected_card = alias
        self._cards[alias].set_selected(True)

        # Show the query panel and update the model label
        self._launch_model_label.setText(f"Model: {alias}")
        self._launch_panel.setVisible(True)
        self._launch_query.setFocus()

    def _run_launch(self):
        if not self._selected_card:
            return
        query = self._launch_query.toPlainText().strip()
        if not query:
            return

        task_type = self._launch_task.currentText().lower()  # "research" or "code"
        self._launch_btn.setEnabled(False)
        self.statusBar().showMessage(f"Running {task_type} with {self._selected_card}…")
        self.tracker.record_request()

        # Start the elapsed time timer
        self._elapsed_seconds = 0
        self._elapsed_timer.start(1000)  # tick every second

        payload = {"model": self._selected_card, "level": self._launch_depth.currentText().lower()}
        if task_type == "research":
            payload["query"] = query
        else:
            payload["request"] = query

        # Dedicated worker wired to the launch output box, not the tab outputs
        self._launch_worker = GeminiWorker(self.delegator, task_type, payload)
        self._launch_worker.text_done.connect(self._on_launch_done)
        self._launch_worker.error.connect(self._on_error)
        self._launch_worker.start()

    def _on_launch_done(self, _task_type, text):
        self._elapsed_timer.stop()
        self._elapsed_label.setText(f"⏱ {self._elapsed_seconds}s (done)")
        self._launch_output.setPlainText(text)
        self._launch_btn.setEnabled(True)
        self.statusBar().showMessage("Done — full file saved to Desktop.")


    # ─────────────────────────────────────────────────────────────────────────
    # RESEARCH TAB
    # ─────────────────────────────────────────────────────────────────────────

    def _init_research_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Model + depth options row ─────────────────────────────────────────
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Model:"))

        # _make_model_combo populates the dropdown AND sets per-item tooltips
        self.research_model = _make_model_combo(_text_models(), "gemini25pro")
        opts.addWidget(self.research_model)
        opts.addSpacing(20)
        opts.addWidget(QLabel("Depth:"))
        self.research_depth = QComboBox()
        self.research_depth.addItems(["Basic", "Intermediate", "Advanced", "Expert"])
        self.research_depth.setCurrentText("Intermediate")
        opts.addWidget(self.research_depth)
        opts.addStretch()
        layout.addLayout(opts)

        # ── Description label — updates live as the model dropdown changes ────
        # Shows the selected model's full description in small grey text.
        # We use currentIndexChanged (index) rather than currentTextChanged (display text)
        # because the display text now includes the short description, not just the alias.
        # itemData(idx, UserRole) gives us back the raw alias stored in _make_model_combo.
        self.research_model_desc = _make_desc_label("gemini25pro")
        self.research_model.currentIndexChanged.connect(
            lambda idx: self.research_model_desc.setText(
                _model_desc(self.research_model.itemData(idx, Qt.ItemDataRole.UserRole) or "")
            )
        )
        layout.addWidget(self.research_model_desc)

        # ── Query input ───────────────────────────────────────────────────────
        q_group = QGroupBox("Research Query")
        q_layout = QVBoxLayout()
        self.research_query = QTextEdit()
        self.research_query.setPlaceholderText("Enter your research topic here…")
        self.research_query.setMaximumHeight(110)
        q_layout.addWidget(self.research_query)
        q_group.setLayout(q_layout)
        layout.addWidget(q_group)

        self.research_btn = QPushButton("Delegate Research")
        self.research_btn.clicked.connect(self._run_research)
        layout.addWidget(self.research_btn)

        out_group = QGroupBox("Results  (capped at 2000 chars — full file auto-saved to Desktop)")
        out_layout = QVBoxLayout()
        self.research_output = QTextEdit()
        self.research_output.setReadOnly(True)
        out_layout.addWidget(self.research_output)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        self.tabs.addTab(tab, "Research")


    # ─────────────────────────────────────────────────────────────────────────
    # CODE TAB
    # ─────────────────────────────────────────────────────────────────────────

    def _init_code_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Model:"))
        self.code_model = _make_model_combo(_text_models(), "gemini25pro")
        opts.addWidget(self.code_model)
        opts.addSpacing(20)
        opts.addWidget(QLabel("Depth:"))
        self.code_depth = QComboBox()
        self.code_depth.addItems(["Basic", "Intermediate", "Advanced", "Expert"])
        self.code_depth.setCurrentText("Intermediate")
        opts.addWidget(self.code_depth)
        opts.addStretch()
        layout.addLayout(opts)

        self.code_model_desc = _make_desc_label("gemini25pro")
        self.code_model.currentIndexChanged.connect(
            lambda idx: self.code_model_desc.setText(
                _model_desc(self.code_model.itemData(idx, Qt.ItemDataRole.UserRole) or "")
            )
        )
        layout.addWidget(self.code_model_desc)

        q_group = QGroupBox("Code Request")
        q_layout = QVBoxLayout()
        self.code_query = QTextEdit()
        self.code_query.setPlaceholderText("Describe the code you need written…")
        self.code_query.setMaximumHeight(110)
        q_layout.addWidget(self.code_query)
        q_group.setLayout(q_layout)
        layout.addWidget(q_group)

        self.code_btn = QPushButton("Generate Code")
        self.code_btn.clicked.connect(self._run_code)
        layout.addWidget(self.code_btn)

        out_group = QGroupBox("Results  (capped at 2000 chars — full file auto-saved to Desktop)")
        out_layout = QVBoxLayout()
        self.code_output = QTextEdit()
        self.code_output.setReadOnly(True)
        mono = QFont("Menlo", 12)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.code_output.setFont(mono)
        out_layout.addWidget(self.code_output)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        self.tabs.addTab(tab, "Code")


    # ─────────────────────────────────────────────────────────────────────────
    # RUN HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _run_research(self):
        query = self.research_query.toPlainText().strip()
        if not query:
            return
        self.research_btn.setEnabled(False)
        self.statusBar().showMessage("Running research…")
        self.tracker.record_request()  # increment session counter
        self._start_worker("research", {
            "query": query,
            "level": self.research_depth.currentText().lower(),
            "model": self.research_model.currentData(),  # UserRole = raw alias
        })

    def _run_code(self):
        request = self.code_query.toPlainText().strip()
        if not request:
            return
        self.code_btn.setEnabled(False)
        self.statusBar().showMessage("Generating code…")
        self.tracker.record_request()
        self._start_worker("code", {
            "request": request,
            "level": self.code_depth.currentText().lower(),
            "model": self.code_model.currentData(),  # UserRole = raw alias
        })

    def _start_worker(self, task_type, payload):
        self.worker = GeminiWorker(self.delegator, task_type, payload)
        self.worker.text_done.connect(self._on_text_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()


    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _on_text_done(self, task_type, text):
        if task_type == "research":
            self.research_output.setPlainText(text)
            self.research_btn.setEnabled(True)
        elif task_type == "code":
            self.code_output.setPlainText(text)
            self.code_btn.setEnabled(True)
        self.statusBar().showMessage("Done — full file saved to Desktop.")

    def _on_error(self, msg):
        self._elapsed_timer.stop()
        self._elapsed_label.setText(f"⏱ {self._elapsed_seconds}s (done)")
        self.research_btn.setEnabled(True)
        self.code_btn.setEnabled(True)
        self._launch_btn.setEnabled(True)
        self.statusBar().showMessage("Error.")
        QMessageBox.critical(self, "Error", msg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiDelegatorGUI()
    window.show()
    sys.exit(app.exec())
