# Gemini Delegator

AI-to-AI delegation for research and coding tasks. Routes requests from Claude to Gemini API with reasoning control.

## Setup

1. **Get a Gemini API key** from https://aistudio.google.com/app/apikey

2. **Create the API key file** at `config/gemini_api_key.json`:
   ```json
   {"api_key": "YOUR_GEMINI_API_KEY_HERE"}
   ```
   (See `config/gemini_api_key.json.example` for the format.)

3. **Install dependencies**:
   ```bash
   pip install google-genai pyyaml
   ```

## Usage

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

**Models:** `gemini25pro` (default) | `gemini25flash` | `pro` | `flash` — see `gemini_delegator.py` for full registry.

## Files

- `gemini_delegator.py` — main script
- `gemini_delegator_config.yaml` — model, reasoning budgets, paths
- `gemini_delegator_spec.md` — full design doc
- `gemini-delegate.skill` — Claude Code skill bundle
- `config/` — API key (gitignored)
- `outputs/` — generated files (>5KB responses)
- `logs/` — execution log
