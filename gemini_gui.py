#!/usr/bin/env python3
"""
Gemini Delegator GUI — PyQt6 desktop interface for gemini_delegator.py
"""

# ── Standard library imports ──────────────────────────────────────────────────
import sys           # lets us exit the app cleanly
import time          # used to timestamp image requests so we can find new files
from pathlib import Path  # cross-platform file path handling

# ── PyQt6 GUI widgets ─────────────────────────────────────────────────────────
# QMainWindow = the outer app window
# QWidget     = a blank container you can put other things inside
# QVBoxLayout / QHBoxLayout = stack children vertically or horizontally
# QTabWidget  = the tab bar (Research / Code / Image)
# QLabel      = a non-editable text or image display
# QComboBox   = a dropdown menu
# QTextEdit   = a multi-line text box (editable or read-only)
# QPushButton = a clickable button
# QFormLayout = a two-column label + field layout (used in the Image params panel)
# QSpinBox    = a numeric up/down counter
# QLineEdit   = a single-line text input
# QMessageBox = a pop-up dialog for errors
# QGroupBox   = a titled border box that groups related controls
# QScrollArea = a scrollable container (used for the image preview grid)
# QGridLayout = arranges children in rows and columns (used for image thumbnails)
# QSizePolicy = controls how a widget grows/shrinks when the window resizes
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QComboBox, QTextEdit, QPushButton,
    QFormLayout, QSpinBox, QLineEdit, QMessageBox, QGroupBox,
    QScrollArea, QGridLayout, QSizePolicy,
)
# QThread     = runs work in the background so the UI doesn't freeze
# pyqtSignal  = a message a background thread sends back to the UI when done
# Qt          = constants like AlignCenter, KeepAspectRatio, etc.
# QTimer      = fires a function on a repeating interval (used for rate limit cooldown)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
# QFont   = lets us set font family and size on a widget
# QPixmap = loads an image file so it can be displayed in a QLabel
from PyQt6.QtGui import QFont, QPixmap

# ── Backend import ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
# GeminiDelegator  = the class that actually calls the Gemini API
# MODEL_REGISTRY   = dict of all 49 model aliases → {id, type, desc}
# STYLE_PRESETS    = dict of all 40+ image style names → expanded prompt strings
# ASPECT_RATIOS    = dict of ratio aliases → actual ratios
from gemini_delegator import GeminiDelegator, MODEL_REGISTRY, STYLE_PRESETS, ASPECT_RATIOS


# ── Model list helpers ────────────────────────────────────────────────────────
def _text_models():
    return [k for k, v in MODEL_REGISTRY.items() if v["type"] == "text"]

def _image_models():
    return [k for k, v in MODEL_REGISTRY.items() if v["type"] == "image"]

def _model_desc(alias):
    """Return the description string for a model alias, or empty string."""
    return MODEL_REGISTRY.get(alias, {}).get("desc", "")


def _make_model_combo(aliases, default):
    """
    Build a QComboBox from a list of model aliases.
    Each item gets a tooltip showing the model's description on hover.
    Returns the combo widget.
    """
    combo = QComboBox()
    combo.addItems(aliases)
    combo.setCurrentText(default)
    # Set per-item tooltip so description appears when hovering over an item in the list
    for i, alias in enumerate(aliases):
        combo.setItemData(i, _model_desc(alias), Qt.ItemDataRole.ToolTipRole)
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
    text_done  = pyqtSignal(str, str)   # (task_type, result_text)
    image_done = pyqtSignal(str, list)  # (message, [desktop_png_paths])
    error      = pyqtSignal(str)

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

            elif self.task_type == "code":
                result = self.delegator.delegate_code(
                    p["request"], p["level"], p["model"],
                )
                self.text_done.emit("code", result)

            elif self.task_type == "image":
                # Record time before the call so we can find new Desktop PNGs after
                before = time.time()
                result = self.delegator.delegate_image(
                    prompt=p["prompt"],
                    model=p["model"],
                    style=p["style"] or None,
                    aspect_ratio=p["aspect_ratio"],
                    num_images=p["count"],
                    lighting=p["lighting"] or None,
                    mood=p["mood"] or None,
                    quality=p["quality"] or None,
                    negative_prompt=p["negative"] or None,
                    person_generation=p["people"],
                    extra=p["extra"] or None,
                )
                desktop = Path.home() / "Desktop"
                new_files = sorted(
                    [str(f) for f in desktop.glob("image_*.png")
                     if f.stat().st_mtime >= before]
                )
                self.image_done.emit(result, new_files)

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

        self._init_research_tab()
        self._init_code_tab()
        self._init_image_tab()

        # ── Stats bar ─────────────────────────────────────────────────────────
        # Persistent label at the bottom of the window showing request count +
        # cooldown state. Uses the status bar slot so it never overlaps content.
        self._stats_label = QLabel(self.tracker.status_text())
        self._stats_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 6px;")
        self.statusBar().addPermanentWidget(self._stats_label)
        self.statusBar().showMessage("Ready")

    def _refresh_stats(self):
        """Called every second by the timer to keep the stats label current."""
        self._stats_label.setText(self.tracker.status_text())


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
        # Shows the selected model's one-line description in small grey text
        self.research_model_desc = _make_desc_label("gemini25pro")
        self.research_model.currentTextChanged.connect(
            lambda alias: self.research_model_desc.setText(_model_desc(alias))
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
        self.code_model.currentTextChanged.connect(
            lambda alias: self.code_model_desc.setText(_model_desc(alias))
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
    # IMAGE TAB
    # ─────────────────────────────────────────────────────────────────────────

    def _init_image_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(12)

        # ── Left: controls ────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        self.image_prompt = QTextEdit()
        self.image_prompt.setPlaceholderText("Describe the image…")
        self.image_prompt.setMaximumHeight(80)
        self.image_negative = QLineEdit()
        self.image_negative.setPlaceholderText("Negative prompt — what to avoid…")
        prompt_layout.addWidget(self.image_prompt)
        prompt_layout.addWidget(self.image_negative)
        prompt_group.setLayout(prompt_layout)
        left_layout.addWidget(prompt_group)

        params_group = QGroupBox("Parameters")
        form = QFormLayout()
        form.setVerticalSpacing(6)

        # Model dropdown with per-item tooltips + live description label
        self.img_model = _make_model_combo(_image_models(), "imagen4")
        form.addRow("Model:", self.img_model)
        self.img_model_desc = _make_desc_label("imagen4")
        self.img_model.currentTextChanged.connect(
            lambda alias: self.img_model_desc.setText(_model_desc(alias))
        )
        form.addRow("", self.img_model_desc)

        self.img_count = QSpinBox()
        self.img_count.setRange(1, 4)
        self.img_count.setValue(1)
        form.addRow("Count:", self.img_count)

        # Aspect ratio dropdown — populated from ASPECT_RATIOS keys
        self.img_aspect = QComboBox()
        self.img_aspect.addItems(list(ASPECT_RATIOS.keys()))
        self.img_aspect.setCurrentText("1:1")
        form.addRow("Aspect Ratio:", self.img_aspect)

        # Style preset dropdown — "None" means no preset, raw prompt sent as-is
        self.img_style = QComboBox()
        self.img_style.addItems(["None"] + list(STYLE_PRESETS.keys()))
        form.addRow("Style:", self.img_style)

        self.img_lighting = QLineEdit()
        self.img_lighting.setPlaceholderText("golden hour, studio, neon…")
        form.addRow("Lighting:", self.img_lighting)

        self.img_mood = QLineEdit()
        self.img_mood.setPlaceholderText("moody, dramatic, serene…")
        form.addRow("Mood:", self.img_mood)

        self.img_quality = QLineEdit()
        self.img_quality.setPlaceholderText("8k, hyperdetailed, masterpiece…")
        form.addRow("Quality:", self.img_quality)

        self.img_extra = QLineEdit()
        self.img_extra.setPlaceholderText("Any extra modifiers…")
        form.addRow("Extra:", self.img_extra)

        # ALLOW_ADULT = adults only (default), DONT_ALLOW = no people, ALLOW_ALL = all ages
        self.img_people = QComboBox()
        self.img_people.addItems(["ALLOW_ADULT", "DONT_ALLOW", "ALLOW_ALL"])
        form.addRow("People:", self.img_people)

        params_group.setLayout(form)
        left_layout.addWidget(params_group)

        self.image_btn = QPushButton("Generate Images")
        self.image_btn.clicked.connect(self._run_image)
        left_layout.addWidget(self.image_btn)
        left_layout.addStretch()
        layout.addWidget(left)

        # ── Right: image preview grid ─────────────────────────────────────────
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self._image_grid_widget = QWidget()
        self.image_grid = QGridLayout(self._image_grid_widget)
        self.image_grid.setSpacing(8)
        self.image_scroll.setWidget(self._image_grid_widget)

        self._placeholder = QLabel("Generated images will appear here.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_grid.addWidget(self._placeholder, 0, 0)

        preview_layout.addWidget(self.image_scroll)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group, 1)

        self.tabs.addTab(tab, "Image")


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
            "model": self.research_model.currentText(),
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
            "model": self.code_model.currentText(),
        })

    def _run_image(self):
        prompt = self.image_prompt.toPlainText().strip()
        if not prompt:
            return
        self.image_btn.setEnabled(False)
        self.statusBar().showMessage("Generating images — this may take a moment…")
        self.tracker.record_request()
        style = self.img_style.currentText()
        self._start_worker("image", {
            "prompt":       prompt,
            "model":        self.img_model.currentText(),
            "style":        None if style == "None" else style,
            "aspect_ratio": self.img_aspect.currentText(),
            "count":        self.img_count.value(),
            "lighting":     self.img_lighting.text().strip(),
            "mood":         self.img_mood.text().strip(),
            "quality":      self.img_quality.text().strip(),
            "negative":     self.image_negative.text().strip(),
            "extra":        self.img_extra.text().strip(),
            "people":       self.img_people.currentText(),
        })

    def _start_worker(self, task_type, payload):
        self.worker = GeminiWorker(self.delegator, task_type, payload)
        self.worker.text_done.connect(self._on_text_done)
        self.worker.image_done.connect(self._on_image_done)
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

    def _on_image_done(self, message, paths):
        self.image_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Done — {len(paths)} image(s) saved to Desktop." if paths else "Done."
        )

        # Clear previous grid contents
        while self.image_grid.count():
            item = self.image_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not paths:
            label = QLabel("Images saved but could not locate files on Desktop.\n\n" + message)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            self.image_grid.addWidget(label, 0, 0)
            return

        # 2 columns for multiple images, 1 column for a single image
        cols = 2 if len(paths) > 1 else 1
        for i, path in enumerate(paths):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                # Scale to fit 500×500 while keeping original proportions
                scaled = pixmap.scaled(
                    500, 500,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                label.setPixmap(scaled)
                self.image_grid.addWidget(label, i // cols, i % cols)
            else:
                self.image_grid.addWidget(
                    QLabel(f"Could not load:\n{path}"), i // cols, i % cols
                )

    def _on_error(self, msg):
        self.research_btn.setEnabled(True)
        self.code_btn.setEnabled(True)
        self.image_btn.setEnabled(True)
        self.statusBar().showMessage("Error.")
        QMessageBox.critical(self, "Error", msg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiDelegatorGUI()
    window.show()
    sys.exit(app.exec())
