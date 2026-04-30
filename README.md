# Gemini Delegator

AI-to-AI delegation for research, coding, and image generation. Routes requests to the Gemini API with reasoning control and a full desktop GUI.

## Setup

1. **Get a Gemini API key** from https://aistudio.google.com/app/apikey

2. **Create the API key file** at `config/gemini_api_key.json`:
   ```json
   {"api_key": "YOUR_GEMINI_API_KEY_HERE"}
   ```
   (See `config/gemini_api_key.json.example` for the format.)

3. **Install dependencies**:
   ```bash
   pip install google-genai pyyaml PyQt6
   ```

## Usage

### GUI (desktop app)
```bash
python3 gemini_gui.py
```
Opens a three-tab window: Research, Code, and Image. All 49 models and 40+ image styles are available from dropdowns.

### CLI

**Research:**
```bash
python3 gemini_delegator.py --type research --query "Your query" --level advanced
```

**Code:**
```bash
python3 gemini_delegator.py --type code --request "Build X" --level intermediate
```

**Image:**
```bash
python3 gemini_delegator.py --type image --prompt "A cat astronaut on Mars" \
  --style cinematic --aspect wide --lighting "golden hour" --num 2
```

**Levels:** `basic` | `intermediate` | `advanced` | `expert`

**Models:** 49 available — see `MODEL_REGISTRY` in `gemini_delegator.py` for the full list.
Default for research/code: `gemini25pro`. Default for images: `imagen4`.

### Output behavior
All responses are saved automatically:
- `outputs/` — inside this project folder
- `~/Desktop/` — a copy lands on your Desktop every time

Text responses are capped at 2000 characters in the terminal/GUI; the full file is always saved.

## Files

- `gemini_delegator.py` — core backend: API calls, model registry, image generation
- `gemini_gui.py` — PyQt6 desktop GUI
- `gemini_delegator_config.yaml` — reasoning budgets and paths
- `gemini_delegator_spec.md` — architecture reference
- `gemini-delegate.skill` — Claude Code skill bundle (lets Claude call this via `/gemini-delegate`)
- `config/` — API key (gitignored)
- `outputs/` — saved responses and images
- `logs/` — execution log
