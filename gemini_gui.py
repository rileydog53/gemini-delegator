#!/usr/bin/env python3
"""
Gemini Delegator GUI — PyQt6 desktop interface for gemini_delegator.py
"""

# ── Standard library imports ──────────────────────────────────────────────────
import sys           # lets us exit the app cleanly
import time          # used to timestamp image requests so we can find new files
from pathlib import Path  # cross-platform file path handling

# ── PyQt6 GUI widgets ─────────────────────────────────────────────────────────
# Each name here is a specific type of UI element we use somewhere in the app.
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
# pyqtSignal  = a message a background thread sends back to the UI when it's done
# Qt          = constants like AlignCenter, KeepAspectRatio, etc.
from PyQt6.QtCore import QThread, pyqtSignal, Qt
# QFont   = lets us set font family and size on a widget
# QPixmap = loads an image file so it can be displayed in a QLabel
from PyQt6.QtGui import QFont, QPixmap

# ── Backend import ────────────────────────────────────────────────────────────
# Makes sure Python can find gemini_delegator.py even if you run this script
# from a different working directory.
sys.path.insert(0, str(Path(__file__).parent))
# GeminiDelegator  = the class that actually calls the Gemini API
# MODEL_REGISTRY   = dict of all 49 model aliases → {id, type}
# STYLE_PRESETS    = dict of all 40+ image style names → expanded prompt strings
# ASPECT_RATIOS    = dict of ratio aliases (square, wide, etc.) → actual ratios
from gemini_delegator import GeminiDelegator, MODEL_REGISTRY, STYLE_PRESETS, ASPECT_RATIOS


# ── Model list helpers ────────────────────────────────────────────────────────
# Returns only the model aliases suitable for text tasks (research / code).
# These are shown in the Research and Code tab model dropdowns.
def _text_models():
    return [k for k, v in MODEL_REGISTRY.items() if v["type"] == "text"]

# Returns only the model aliases suitable for image generation.
# These are shown in the Image tab model dropdown.
def _image_models():
    return [k for k, v in MODEL_REGISTRY.items() if v["type"] == "image"]


# ─────────────────────────────────────────────────────────────────────────────
# WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────
# Gemini API calls can take several seconds. If we ran them on the main thread,
# the entire window would freeze until they finished. Instead, we run them here
# in a background thread. When the call is done, we "emit" a signal back to the
# main window, which updates the UI safely.

class GeminiWorker(QThread):

    # Signals are how the worker communicates back to the main window.
    # text_done fires when a research or code task completes.
    #   sends: (task_type string, result text string)
    text_done  = pyqtSignal(str, str)

    # image_done fires when an image task completes.
    #   sends: (status message string, list of file paths on Desktop)
    image_done = pyqtSignal(str, list)

    # error fires if anything throws an exception during the API call.
    #   sends: (error message string)
    error      = pyqtSignal(str)

    def __init__(self, delegator, task_type, payload):
        super().__init__()
        self.delegator = delegator  # shared GeminiDelegator instance
        self.task_type = task_type  # "research", "code", or "image"
        self.payload   = payload    # dict of all inputs from the UI

    # run() is called automatically when the thread starts.
    # Everything inside here runs in the background.
    def run(self):
        try:
            p = self.payload  # shorthand so we don't type self.payload everywhere

            # ── Research task ────────────────────────────────────────────────
            if self.task_type == "research":
                result = self.delegator.delegate_research(
                    p["query"],   # the text the user typed
                    p["level"],   # basic / intermediate / advanced / expert
                    p["model"],   # model alias chosen in the dropdown
                )
                # Send result back to the main window
                self.text_done.emit("research", result)

            # ── Code task ────────────────────────────────────────────────────
            elif self.task_type == "code":
                result = self.delegator.delegate_code(
                    p["request"],  # the code description the user typed
                    p["level"],
                    p["model"],
                )
                self.text_done.emit("code", result)

            # ── Image task ───────────────────────────────────────────────────
            elif self.task_type == "image":
                # Record the time before we call the API. After the call we'll
                # scan the Desktop for any PNG files newer than this timestamp —
                # that's how we find which files to display.
                before = time.time()

                result = self.delegator.delegate_image(
                    prompt=p["prompt"],                   # main description
                    model=p["model"],                     # imagen4, imagen4ultra, etc.
                    style=p["style"] or None,             # style preset or None
                    aspect_ratio=p["aspect_ratio"],       # e.g. "1:1", "16:9"
                    num_images=p["count"],                # 1–4
                    lighting=p["lighting"] or None,       # e.g. "golden hour"
                    mood=p["mood"] or None,               # e.g. "moody"
                    quality=p["quality"] or None,         # e.g. "8k, hyperdetailed"
                    negative_prompt=p["negative"] or None,# what to avoid
                    person_generation=p["people"],        # ALLOW_ADULT etc.
                    extra=p["extra"] or None,             # any freeform additions
                )

                # Find PNG files on Desktop created after we started the request.
                # delegate_image() already saves them there; we just need the paths.
                desktop = Path.home() / "Desktop"
                new_files = sorted(
                    [str(f) for f in desktop.glob("image_*.png")
                     if f.stat().st_mtime >= before]
                )
                # Send message + file paths back to the main window
                self.image_done.emit(result, new_files)

        except Exception as e:
            # Any API error, network issue, etc. comes through here
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class GeminiDelegatorGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        # ── Window setup ─────────────────────────────────────────────────────
        self.setWindowTitle("Gemini Delegator")  # title bar text — change to rename the app
        self.resize(1060, 740)                   # initial window size in pixels (width, height)
        self.worker = None                       # placeholder; gets set when a request starts

        # ── Initialize the backend ───────────────────────────────────────────
        # GeminiDelegator reads the API key and config once at startup.
        # We create it here so all three tabs share one connection.
        try:
            self.delegator = GeminiDelegator()
        except Exception as e:
            # If the API key is missing or invalid, show an error and quit.
            QMessageBox.critical(None, "Startup Error",
                                 f"Could not initialize Gemini client:\n{e}")
            sys.exit(1)

        # Build all the visual elements
        self._init_ui()

    def _init_ui(self):
        # QMainWindow requires a "central widget" as the root container.
        central = QWidget()
        self.setCentralWidget(central)

        # Stack everything vertically inside the central widget.
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)  # padding around the edges (px)
        root.setSpacing(8)                        # gap between stacked items (px)

        # Create the tab bar and add it to the layout.
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # Build each tab's contents.
        self._init_research_tab()
        self._init_code_tab()
        self._init_image_tab()

        # Show "Ready" in the bottom status bar when the app first loads.
        self.statusBar().showMessage("Ready")


    # ─────────────────────────────────────────────────────────────────────────
    # RESEARCH TAB
    # Layout: [Model dropdown]  [Depth dropdown]
    #         [Query text box]
    #         [Delegate Research button]
    #         [Results text box  (read-only)]
    # ─────────────────────────────────────────────────────────────────────────

    def _init_research_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)  # children stack top-to-bottom

        # ── Top row: Model + Depth dropdowns ─────────────────────────────────
        opts = QHBoxLayout()  # these two sit side-by-side
        opts.addWidget(QLabel("Model:"))

        self.research_model = QComboBox()
        self.research_model.addItems(_text_models())        # all text-capable models
        self.research_model.setCurrentText("gemini25pro")   # default selection
        opts.addWidget(self.research_model)

        opts.addSpacing(20)  # visual gap between the two dropdowns
        opts.addWidget(QLabel("Depth:"))

        self.research_depth = QComboBox()
        self.research_depth.addItems(["Basic", "Intermediate", "Advanced", "Expert"])
        self.research_depth.setCurrentText("Intermediate")  # default
        opts.addWidget(self.research_depth)

        opts.addStretch()  # pushes everything left so it doesn't spread across the full width
        layout.addLayout(opts)

        # ── Query input ───────────────────────────────────────────────────────
        q_group = QGroupBox("Research Query")  # titled border box
        q_layout = QVBoxLayout()
        self.research_query = QTextEdit()
        self.research_query.setPlaceholderText("Enter your research topic here…")
        self.research_query.setMaximumHeight(110)  # keep it from taking over the whole screen
        q_layout.addWidget(self.research_query)
        q_group.setLayout(q_layout)
        layout.addWidget(q_group)

        # ── Submit button ─────────────────────────────────────────────────────
        # .clicked.connect(...) means "call this function when the button is pressed"
        self.research_btn = QPushButton("Delegate Research")
        self.research_btn.clicked.connect(self._run_research)
        layout.addWidget(self.research_btn)

        # ── Results output ────────────────────────────────────────────────────
        out_group = QGroupBox("Results  (capped at 2000 chars — full file auto-saved to Desktop)")
        out_layout = QVBoxLayout()
        self.research_output = QTextEdit()
        self.research_output.setReadOnly(True)  # user can read but not edit
        out_layout.addWidget(self.research_output)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        # Register this tab with the tab bar. Second argument is the tab label.
        self.tabs.addTab(tab, "Research")


    # ─────────────────────────────────────────────────────────────────────────
    # CODE TAB
    # Same structure as Research — Model + Depth dropdowns, a text input,
    # a button, and a read-only output box. Output uses a monospace font
    # (Menlo on Mac) so code is easier to read.
    # ─────────────────────────────────────────────────────────────────────────

    def _init_code_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Model:"))
        self.code_model = QComboBox()
        self.code_model.addItems(_text_models())
        self.code_model.setCurrentText("gemini25pro")
        opts.addWidget(self.code_model)
        opts.addSpacing(20)
        opts.addWidget(QLabel("Depth:"))
        self.code_depth = QComboBox()
        self.code_depth.addItems(["Basic", "Intermediate", "Advanced", "Expert"])
        self.code_depth.setCurrentText("Intermediate")
        opts.addWidget(self.code_depth)
        opts.addStretch()
        layout.addLayout(opts)

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
        # Use a monospace font so code columns and indentation line up correctly.
        # To change the font size, edit the second argument (12).
        mono = QFont("Menlo", 12)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.code_output.setFont(mono)
        out_layout.addWidget(self.code_output)
        out_group.setLayout(out_layout)
        layout.addWidget(out_group)

        self.tabs.addTab(tab, "Code")


    # ─────────────────────────────────────────────────────────────────────────
    # IMAGE TAB
    # Layout: two side-by-side panels.
    #   Left  (max 310px wide) — all the generation controls
    #   Right (fills remaining space) — scrollable image preview grid
    # ─────────────────────────────────────────────────────────────────────────

    def _init_image_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)  # children sit side-by-side
        layout.setSpacing(12)

        # ── LEFT PANEL: all the controls ──────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(310)  # cap width so preview gets most of the space
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # ── Prompt section ────────────────────────────────────────────────────
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()

        self.image_prompt = QTextEdit()
        self.image_prompt.setPlaceholderText("Describe the image…")
        self.image_prompt.setMaximumHeight(80)

        # Single-line text field for things to leave out of the image
        self.image_negative = QLineEdit()
        self.image_negative.setPlaceholderText("Negative prompt — what to avoid…")

        prompt_layout.addWidget(self.image_prompt)
        prompt_layout.addWidget(self.image_negative)
        prompt_group.setLayout(prompt_layout)
        left_layout.addWidget(prompt_group)

        # ── Parameters section ────────────────────────────────────────────────
        # QFormLayout renders as a two-column grid: label on the left, widget on the right.
        # To add a new parameter row: form.addRow("Label:", self.your_widget)
        params_group = QGroupBox("Parameters")
        form = QFormLayout()
        form.setVerticalSpacing(6)  # gap between rows (px) — increase to spread out

        # Model dropdown — only shows imagen models (filtered by _image_models())
        self.img_model = QComboBox()
        self.img_model.addItems(_image_models())
        self.img_model.setCurrentText("imagen4")  # default model
        form.addRow("Model:", self.img_model)

        # Number of images to generate (1–4)
        self.img_count = QSpinBox()
        self.img_count.setRange(1, 4)
        self.img_count.setValue(1)  # default
        form.addRow("Count:", self.img_count)

        # Aspect ratio dropdown — populated from ASPECT_RATIOS in gemini_delegator.py
        # To add a new ratio, add it there and it will appear here automatically.
        self.img_aspect = QComboBox()
        self.img_aspect.addItems(list(ASPECT_RATIOS.keys()))
        self.img_aspect.setCurrentText("1:1")
        form.addRow("Aspect Ratio:", self.img_aspect)

        # Style preset dropdown — populated from STYLE_PRESETS in gemini_delegator.py
        # "None" means no style preset; the raw prompt is sent as-is.
        self.img_style = QComboBox()
        self.img_style.addItems(["None"] + list(STYLE_PRESETS.keys()))
        form.addRow("Style:", self.img_style)

        # Freeform lighting modifier — passed directly to the image prompt
        self.img_lighting = QLineEdit()
        self.img_lighting.setPlaceholderText("golden hour, studio, neon…")
        form.addRow("Lighting:", self.img_lighting)

        # Freeform mood modifier
        self.img_mood = QLineEdit()
        self.img_mood.setPlaceholderText("moody, dramatic, serene…")
        form.addRow("Mood:", self.img_mood)

        # Freeform quality modifier
        self.img_quality = QLineEdit()
        self.img_quality.setPlaceholderText("8k, hyperdetailed, masterpiece…")
        form.addRow("Quality:", self.img_quality)

        # Any other additions you want appended to the final prompt
        self.img_extra = QLineEdit()
        self.img_extra.setPlaceholderText("Any extra modifiers…")
        form.addRow("Extra:", self.img_extra)

        # Controls whether people/faces can appear in the output.
        # ALLOW_ADULT = adults only (default), DONT_ALLOW = no people, ALLOW_ALL = all ages
        self.img_people = QComboBox()
        self.img_people.addItems(["ALLOW_ADULT", "DONT_ALLOW", "ALLOW_ALL"])
        form.addRow("People:", self.img_people)

        params_group.setLayout(form)
        left_layout.addWidget(params_group)

        # ── Generate button ───────────────────────────────────────────────────
        self.image_btn = QPushButton("Generate Images")
        self.image_btn.clicked.connect(self._run_image)
        left_layout.addWidget(self.image_btn)

        left_layout.addStretch()  # pushes everything up; keeps button near the params
        layout.addWidget(left)

        # ── RIGHT PANEL: image preview grid ───────────────────────────────────
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        # QScrollArea lets the preview panel scroll if images are taller than the window.
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)  # grid resizes with the scroll area

        # The actual grid lives inside a plain QWidget inside the scroll area.
        self._image_grid_widget = QWidget()
        self.image_grid = QGridLayout(self._image_grid_widget)
        self.image_grid.setSpacing(8)  # gap between thumbnail cells (px)
        self.image_scroll.setWidget(self._image_grid_widget)

        # Placeholder shown before any images are generated
        self._placeholder = QLabel("Generated images will appear here.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_grid.addWidget(self._placeholder, 0, 0)

        preview_layout.addWidget(self.image_scroll)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group, 1)  # stretch factor 1 = take all remaining width

        self.tabs.addTab(tab, "Image")


    # ─────────────────────────────────────────────────────────────────────────
    # RUN HANDLERS
    # Each function is called when the user clicks a submit button.
    # They read the current values from the UI, disable the button so the user
    # can't double-submit, update the status bar, then hand off to _start_worker.
    # ─────────────────────────────────────────────────────────────────────────

    def _run_research(self):
        # Read the query box; strip() removes accidental leading/trailing whitespace.
        query = self.research_query.toPlainText().strip()
        if not query:
            return  # do nothing if the box is empty

        self.research_btn.setEnabled(False)  # grey out the button during the request
        self.statusBar().showMessage("Running research…")

        # Build the payload dict — everything the worker needs to call the API.
        # .lower() converts "Intermediate" → "intermediate" as the API expects.
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

        # Read the style dropdown separately so we can convert "None" → Python None.
        style = self.img_style.currentText()

        self._start_worker("image", {
            "prompt":       prompt,
            "model":        self.img_model.currentText(),
            "style":        None if style == "None" else style,  # "None" string → actual None
            "aspect_ratio": self.img_aspect.currentText(),
            "count":        self.img_count.value(),               # int from the spinner
            "lighting":     self.img_lighting.text().strip(),
            "mood":         self.img_mood.text().strip(),
            "quality":      self.img_quality.text().strip(),
            "negative":     self.image_negative.text().strip(),
            "extra":        self.img_extra.text().strip(),
            "people":       self.img_people.currentText(),
        })

    def _start_worker(self, task_type, payload):
        # Create the background thread, wire up its signals, and start it.
        # text_done  → _on_text_done    (research and code results)
        # image_done → _on_image_done   (image results + file paths)
        # error      → _on_error        (any exception from the API)
        self.worker = GeminiWorker(self.delegator, task_type, payload)
        self.worker.text_done.connect(self._on_text_done)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()  # kicks off GeminiWorker.run() in the background


    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE HANDLERS
    # These are called by the worker signals once the API call completes.
    # They run on the main thread, so it's safe to update the UI here.
    # ─────────────────────────────────────────────────────────────────────────

    def _on_text_done(self, task_type, text):
        # Route the result to whichever tab made the request, then re-enable its button.
        if task_type == "research":
            self.research_output.setPlainText(text)
            self.research_btn.setEnabled(True)
        elif task_type == "code":
            self.code_output.setPlainText(text)
            self.code_btn.setEnabled(True)
        self.statusBar().showMessage("Done — full file saved to Desktop.")

    def _on_image_done(self, message, paths):
        # Re-enable the button and update the status bar.
        self.image_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Done — {len(paths)} image(s) saved to Desktop." if paths else "Done."
        )

        # Remove all existing widgets from the grid before showing new images.
        # takeAt(0) always removes the first item; the loop runs until the grid is empty.
        while self.image_grid.count():
            item = self.image_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()  # free memory

        # Edge case: API succeeded but we couldn't find the files on disk.
        if not paths:
            label = QLabel("Images saved but could not locate files on Desktop.\n\n" + message)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            self.image_grid.addWidget(label, 0, 0)
            return

        # Use 2 columns when there's more than 1 image, otherwise 1 column.
        # To change the grid layout, edit this value.
        cols = 2 if len(paths) > 1 else 1

        for i, path in enumerate(paths):
            pixmap = QPixmap(path)  # load the PNG file from disk
            if not pixmap.isNull():
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Expanding,
                )
                # Scale the image to fit within 500×500 px while keeping its proportions.
                # To change thumbnail size, edit both 500 values here.
                scaled = pixmap.scaled(
                    500, 500,
                    Qt.AspectRatioMode.KeepAspectRatio,       # never distorts the image
                    Qt.TransformationMode.SmoothTransformation, # high-quality downscale
                )
                label.setPixmap(scaled)
                # Place the label in the grid: row = i // cols, column = i % cols
                # e.g. 4 images in 2 columns → positions (0,0) (0,1) (1,0) (1,1)
                self.image_grid.addWidget(label, i // cols, i % cols)
            else:
                # If the file can't be loaded as an image, show a text error in its cell.
                self.image_grid.addWidget(
                    QLabel(f"Could not load:\n{path}"), i // cols, i % cols
                )

    def _on_error(self, msg):
        # Re-enable all three buttons — we don't know which task errored.
        self.research_btn.setEnabled(True)
        self.code_btn.setEnabled(True)
        self.image_btn.setEnabled(True)
        self.statusBar().showMessage("Error.")
        # Show a pop-up dialog with the full error message.
        QMessageBox.critical(self, "Error", msg)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# Python runs this block when you execute the file directly.
# QApplication is the global app object — there must be exactly one.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeminiDelegatorGUI()
    window.show()
    sys.exit(app.exec())  # starts the event loop; app runs until the window is closed
