# Gemini Delegator Program — Research & Coding Agent

## Overview
A Python-based program that delegates **research and coding tasks** to Gemini API. Joseph dictates or types a request, Claude routes it to the Gemini delegator, and results come back as text (small) or files (large).

**Token savings:** ~3–5× reduction in Claude token usage by offloading heavy research and coding work to Gemini.

---

## Simplified Workflow

```
Joseph: "Research the Mezzogiorno (advanced)" or "Code a batch image processor (intermediate)"
  ↓
Claude: Parses request, builds Gemini prompt
  ↓
gemini_delegator.py: Calls Gemini API with reasoning budget
  ↓
Gemini: Performs research/coding with controlled reasoning depth
  ↓
gemini_delegator.py: Formats output (text or file)
  ↓
Claude: Delivers to Joseph
```

---

## Use Cases

### 1. **Research Tasks**
- Input: Research query + scope (basic/intermediate/advanced)
- Gemini: Web search + synthesis with reasoning depth control
- Output: Structured findings, sources, summary
- Returned as: Text (short) or markdown file (long)
- Example: "Research Italy's Mezzogiorno economic divide (advanced)"

### 2. **Programming Tasks**
- Input: Code request + complexity level (basic/intermediate/advanced)
- Gemini: Code generation with appropriate reasoning depth
- Output: Runnable code + inline comments + explanation
- Returned as: .py file (always), with summary text
- Example: "Write a Python script to batch-process images (intermediate)"

---

## Controlling Gemini's Reasoning Level

Gemini exposes reasoning depth via the **`thinking` parameter** in API calls:

```python
# Basic reasoning (fast, direct)
{
  "thinking": {
    "type": "disabled"
  }
}

# Intermediate reasoning (balanced)
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  }
}

# Advanced reasoning (deep analysis)
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 30000
  }
}

# Expert reasoning (maximum depth)
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 50000
  }
}
```

**When to use each:**
- **Basic:** Quick research, simple coding tasks
- **Intermediate:** Complex research, intermediate coding
- **Advanced:** Multi-angle analysis, advanced coding patterns
- **Expert:** Cutting-edge research, architectural decisions

---

## Program Structure

```python
gemini_delegator.py
├── authenticate_gemini()           # OAuth for Gemini API
├── parse_request(user_input)       # Identify type + complexity
├── build_gemini_prompt()           # Format request for Gemini
├── set_reasoning_budget(level)     # Determine thinking tokens
├── call_gemini_api()               # Send to Gemini with reasoning budget
├── parse_response()                # Extract results
├── format_output()                 # Text or file
│   ├── format_research()           # Markdown with citations
│   └── format_code()               # .py file + docs
├── save_to_file()                  # Store locally if needed
└── log_execution()                 # Track what was delegated
```

---

## Configuration

```yaml
# gemini_delegator_config.yaml
gemini:
  model: "gemini-2.0-pro"  # or latest available
  api_version: "v1beta"

reasoning_budgets:
  research:
    basic: 1000
    intermediate: 10000
    advanced: 30000
  
  code:
    basic: 2000
    intermediate: 8000
    advanced: 25000

output_formats:
  research: markdown
  code: python
  
paths:
  outputs: ~/Desktop/Claude/outputs/
  logs: ~/Desktop/Claude/logs/
```

---

## Usage Examples

**Research:**
```bash
python3 gemini_delegator.py \
  --type research \
  --query "Mezzogiorno economic factors" \
  --level advanced
```

Returns: Markdown report + summary in chat, or file path if >5KB

**Code:**
```bash
python3 gemini_delegator.py \
  --type code \
  --request "Batch process PNG files, resize to 800x600" \
  --level intermediate
```

Returns: Python script saved as file, summary with file path

---

## One-Shot Setup Checklist

- [ ] Gemini API key generated → stored in `~/Desktop/Claude/config/gemini_api_key.json`
- [ ] Google Cloud project created (optional, only if web search needed)
- [ ] Directory structure created: `config/`, `outputs/`, `logs/`
- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `google-ai-python-sdk`, `pyyaml`, `python-dateutil`

---

## To Build This

Just ask Claude:
> "Build gemini_delegator.py"

Claude will create the complete program (~200–300 lines) that handles everything. You run it via Claude Code.

---

## Output Examples

### Research (Advanced)
```
📊 Research Results: Mezzogiorno Economic Divide

FINDINGS:
- Historical GDP gap: North ~2.5x higher than South (1950s–1990s)
- Structural issues: Infrastructure, education, governance
- Recent trends: Convergence slowing post-2008

SOURCES:
- OECD Regional Development Database
- Banca d'Italia economic reports
- Journal of Economic History (2015–2024)

[Full report saved to: ~/Desktop/Claude/outputs/research_mezzogiorno_20260407.md]
```

### Code (Intermediate)
```python
#!/usr/bin/env python3
"""
Batch image processor: Resize PNG files to 800x600
Usage: python3 batch_resize.py input_dir output_dir
"""

import os
from PIL import Image
from pathlib import Path

# ... full code ...

# Summary: Script created with error handling, logging, progress bar
# Location: ~/Desktop/Claude/outputs/batch_resize_20260407.py
```

---

## Notes

1. **Gemini thinking tokens are separate from output tokens** — they enable deeper reasoning but don't appear in the final response
2. **Research requires web search** — may need additional Google Search API setup (we'll handle at build time)
3. **Code is always saved as .py files** — small reports come as text
4. **Logging is local only** — all executions tracked in `~/Desktop/Claude/logs/delegator.log`

---

## Next Steps

1. Complete one-shot setup checklist above (can do anytime)
2. Ask Claude: "Build gemini_delegator.py"
3. Test with example commands
4. Use by asking Claude to delegate work

That's it. Simple, focused, powerful.
