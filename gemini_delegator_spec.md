# Gemini Delegator — Architecture Reference

## What it does

Routes three types of tasks to the Gemini API:
- **Research** — structured findings + sources returned as JSON, saved to outputs/ and Desktop
- **Code** — runnable code + explanation returned as JSON, saved to outputs/ and Desktop
- **Image** — PNG files generated via Imagen 4, saved to outputs/ and Desktop, displayed in the GUI

All responses are truncated to 2000 characters for terminal/GUI display. The full file is always saved regardless.

---

## Files

| File | Role |
|------|------|
| `gemini_delegator.py` | Core backend. All API calls happen here. |
| `gemini_gui.py` | PyQt6 desktop app. Imports the backend directly. |
| `gemini_delegator_config.yaml` | Reasoning budgets, paths. Read at startup. |
| `gemini-delegate.skill` | Claude Code skill — lets Claude invoke the CLI. |
| `config/gemini_api_key.json` | API key. Gitignored. |
| `outputs/` | All saved responses (JSON + PNG). |
| `logs/` | Execution log. |

---

## Backend structure (`gemini_delegator.py`)

```
MODEL_REGISTRY          dict of 49 model aliases → {id, type}
STYLE_PRESETS           dict of 40+ image style names → expanded prompt strings
ASPECT_RATIOS           dict of ratio aliases → actual ratios (e.g. "wide" → "16:9")

GeminiDelegator
├── __init__()              loads config + API key, creates google.genai.Client
├── delegate_research()     builds prompt → _call_gemini_with_retry → _parse_and_format_response
├── delegate_code()         builds prompt → _call_gemini_with_retry → _parse_and_format_response
├── delegate_image()        builds prompt via _build_image_prompt → generate_images → saves PNGs
├── _build_research_prompt()    formats query + depth hint
├── _build_code_prompt()        formats request + complexity hint
├── _build_image_prompt()       stacks style/lighting/mood/quality/extra onto base prompt
├── _call_gemini_with_retry()   calls generate_content with rate-limit retry (max 3x)
└── _parse_and_format_response() parses JSON, saves to outputs/ + Desktop, caps at 2000 chars
```

---

## Reasoning depth

Thinking-token budgets are set per task type and level in `gemini_delegator_config.yaml`.
Only applies to thinking-capable models (`gemini-2.5-pro`, `gemini-2.5-flash`).

| Level | Research tokens | Code tokens |
|-------|----------------|-------------|
| basic | 1,000 | 2,000 |
| intermediate | 10,000 | 8,000 |
| advanced | 30,000 | 25,000 |
| expert | 50,000 | 50,000 |

---

## Model types

Models in the registry are tagged by type, which determines the API path used:

| Type | API call | Used for |
|------|----------|---------|
| `text` | `generate_content` | research, code |
| `image` | `generate_images` | image generation (Imagen) |
| `video` | `generate_video` | video generation (Veo) — registered, not yet implemented |
| `tts` | `generate_content` + audio config | text-to-speech — registered, not yet implemented |
| `audio` | `generate_content` + audio config | native audio I/O — registered, not yet implemented |
| `embedding` | `embed_content` | vector embeddings |
| `aqa` | `generate_answer` | grounded Q&A against source documents |

---

## GUI structure (`gemini_gui.py`)

```
GeminiWorker(QThread)
└── run()       calls real GeminiDelegator methods in a background thread
                emits text_done(task_type, text) or image_done(message, paths) when done

GeminiDelegatorGUI(QMainWindow)
├── _init_research_tab()    model dropdown (text models), depth, query input, output box
├── _init_code_tab()        same as research; output uses monospace font
├── _init_image_tab()       all image params on the left; QPixmap preview grid on the right
├── _run_research/code/image()   read UI → disable button → start worker
├── _on_text_done()         write result to correct output box, re-enable button
├── _on_image_done()        clear grid, load PNGs via QPixmap, display in grid
└── _on_error()             re-enable all buttons, show QMessageBox
```

---

## Image generation details

`delegate_image()` builds the final prompt by appending style/lighting/mood/quality/extra
modifiers to the base description, then calls `client.models.generate_images()` with
`GenerateImagesConfig`. PNGs are saved to both `outputs/` and `~/Desktop/`.

The GUI detects new image files by recording `time.time()` before the API call, then
scanning Desktop for `image_*.png` files with a modification time ≥ that timestamp.

---

## Rate limiting

`_call_gemini_with_retry()` sleeps 4 seconds before every request (15 RPM limit = ~4 sec/request).
On 429 errors it waits 60 seconds and retries up to 3 times total.
