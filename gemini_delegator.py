#!/usr/bin/env python3
"""
Gemini Delegator — AI-to-AI delegation for research and coding tasks.
Routes requests to Gemini API with reasoning control and JSON response format.
"""

import sys
import os
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
import google.genai
from google.genai.errors import APIError, ClientError
from google.genai.types import ThinkingConfig, GenerateContentConfig

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "delegator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Model Registry ───────────────────────────────────────────────────────────
# Pass the short alias via --model. Type tells the script which API path to use.
# "text"      → generate_content  (default pipeline)
# "image"     → generate_images   (Imagen models)
# "embedding" → embed_content     (vector embeddings)
# "aqa"       → generate_answer   (grounded Q&A with citations)
MODEL_REGISTRY = {
    # ── Gemini 2.5 ────────────────────────────────────────────────────────────
    "gemini25pro":   {"id": "gemini-2.5-pro",                 "type": "text"},  # default — deep reasoning + thinking
    "gemini25flash": {"id": "gemini-2.5-flash",               "type": "text"},  # fast, large context, research summaries
    "pro":           {"id": "gemini-2.5-pro",                 "type": "text"},  # short alias
    "flash":         {"id": "gemini-2.5-flash",               "type": "text"},  # short alias
    # ── Gemini 2.0 ────────────────────────────────────────────────────────────
    "gemini20flash": {"id": "gemini-2.0-flash",               "type": "text"},  # speed-optimized general tasks
    "gemini20lite":  {"id": "gemini-2.0-flash-lite",          "type": "text"},  # highest-volume simple tasks
    # ── Gemini 3.x (preview) ──────────────────────────────────────────────────
    "gemini3pro":    {"id": "gemini-3-pro-preview",           "type": "text"},  # next-gen deep reasoning
    "gemini3flash":  {"id": "gemini-3-flash-preview",         "type": "text"},  # next-gen fast tasks
    # ── Image generation ──────────────────────────────────────────────────────
    "imagen4":       {"id": "imagen-4.0-generate-001",        "type": "image"}, # photorealistic, high detail
    "imagen4fast":   {"id": "imagen-4.0-fast-generate-001",   "type": "image"}, # fast image generation
    "imagen4ultra":  {"id": "imagen-4.0-ultra-generate-001",  "type": "image"}, # highest quality images
    # ── Embeddings ────────────────────────────────────────────────────────────
    "embedding":     {"id": "gemini-embedding-001",           "type": "embedding"}, # semantic vector search
    # ── Grounded Q&A (requires source file input) ─────────────────────────────
    "aqa":           {"id": "aqa",                            "type": "aqa"},   # cited Q&A against provided documents
}

DEFAULT_MODEL_ALIAS = "gemini25pro"


def resolve_model(alias_or_id: str) -> dict:
    """Return registry entry for alias or raw model ID string."""
    if alias_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias_or_id]
    # Accept raw model IDs (e.g. 'gemini-1.5-flash') passed directly
    return {"id": alias_or_id, "type": "text"}


SYSTEM_INSTRUCTION = (
    "You are a specialized Research and Summarization Assistant. Your role is to process "
    "raw text, chat logs, or research data provided by a Python script. "
    "Always output your response in a clean, structured format (JSON is preferred if the user "
    "asks for data; Markdown if they ask for a report). Assume both reader and advanced AI will "
    "be seeing it and understand at a graduate level for academics. "
    "Do not include conversational filler like 'Here is the summary' or 'I hope this helps.' "
    "If the input is a chat log, identify key themes and action items. "
    "If the input is research, provide a high-level summary followed by technical details."
)


class GeminiDelegator:
    def __init__(self, config_path="gemini_delegator_config.yaml"):
        """Initialize delegator with config and API key."""
        self.config_path = Path(__file__).parent / config_path
        self.config = self._load_config()
        self.api_key = self._load_api_key()
        self.client = self._init_client()
        self.output_dir = Path(__file__).parent / self.config["paths"]["outputs"]
        self.output_dir.mkdir(exist_ok=True)

    def _load_config(self):
        """Load YAML configuration."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_api_key(self):
        """Load API key from JSON file."""
        key_file = Path(__file__).parent / self.config["gemini"]["api_key_file"]
        with open(key_file, "r") as f:
            data = json.load(f)
        return data["api_key"]

    def _init_client(self):
        """Initialize Gemini API client."""
        os.environ["GEMINI_API_KEY"] = self.api_key
        return google.genai.Client()

    def _get_reasoning_budget(self, task_type, level):
        """Get thinking token budget based on task type and level."""
        budgets = self.config["reasoning_budgets"]
        return budgets.get(task_type, {}).get(level, 10000)

    def _build_research_prompt(self, query, level):
        """Build research prompt for Gemini."""
        depth_hints = {
            "basic": "Provide a quick overview with key facts.",
            "intermediate": "Provide structured findings with analysis and sources.",
            "advanced": "Conduct in-depth analysis from multiple angles with detailed sources.",
            "expert": "Perform exhaustive research with synthesis and critical evaluation."
        }
        hint = depth_hints.get(level, depth_hints["intermediate"])
        return f"""Research Query: {query}

{hint}

Provide structured findings with clear sections and citations where applicable.
Return the response as valid JSON with keys: findings (list), sources (list), summary (string)."""

    def _build_code_prompt(self, request, level):
        """Build code generation prompt for Gemini."""
        complexity_hints = {
            "basic": "Write simple, direct code with inline comments.",
            "intermediate": "Write clean, well-structured code with error handling.",
            "advanced": "Write production-quality code with advanced patterns and optimization.",
            "expert": "Write expert-level code with architectural considerations and best practices."
        }
        hint = complexity_hints.get(level, complexity_hints["intermediate"])
        return f"""Code Request: {request}

{hint}

Return the response as valid JSON with keys: code (string - full Python code), explanation (string), notes (string)."""

    def _call_gemini_with_retry(self, prompt, model, thinking_budget, max_retries=3):
        """Call Gemini API with rate limiting and error handling."""
        # Thinking tokens only supported on gemini-2.5-pro
        THINKING_CAPABLE = {"gemini-2.5-pro"}
        supports_thinking = model in THINKING_CAPABLE

        call_config = GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json"
        )
        if supports_thinking and thinking_budget > 0:
            call_config.thinking_config = ThinkingConfig(
                thinking_budget=thinking_budget
            )

        for attempt in range(max_retries):
            try:
                # Rate limiting: 4 seconds between requests (15 RPM = ~4 sec/request)
                time.sleep(4)

                model_name = model  # Variable controlling which Gemini model is called

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=call_config,
                )

                logger.info(f"✓ Gemini response received from {model}")
                return response.text

            except (APIError, ClientError) as e:
                if "429" in str(e) or "Resource Exhausted" in str(e):
                    wait_time = 60
                    logger.warning(f"Rate limit hit (429). Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    print(f"Rate limit reached. Waiting {wait_time}s before retry...", file=sys.stderr)
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                else:
                    logger.error(f"API Error: {e}")
                    raise

    def delegate_research(self, query, level="intermediate", model=None):
        """Delegate research task to Gemini."""
        entry = resolve_model(model or DEFAULT_MODEL_ALIAS)
        model_name = entry["id"]
        thinking_budget = self._get_reasoning_budget("research", level)

        logger.info(f"Research task: {query[:50]}... (level: {level}, model: {model_name})")
        prompt = self._build_research_prompt(query, level)

        response_text = self._call_gemini_with_retry(prompt, model_name, thinking_budget)
        return self._parse_and_format_response(response_text, "research", query)

    def delegate_code(self, request, level="intermediate", model=None):
        """Delegate code generation task to Gemini."""
        entry = resolve_model(model or DEFAULT_MODEL_ALIAS)
        model_name = entry["id"]
        thinking_budget = self._get_reasoning_budget("code", level)

        logger.info(f"Code task: {request[:50]}... (level: {level}, model: {model_name})")
        prompt = self._build_code_prompt(request, level)

        response_text = self._call_gemini_with_retry(prompt, model_name, thinking_budget)
        return self._parse_and_format_response(response_text, "code", request)

    def _parse_and_format_response(self, response_text, task_type, query_summary):
        """Parse JSON response, always save to outputs/ and Desktop, return text in chat."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response_text[:100]}")
            return response_text

        formatted = json.dumps(data, indent=2)

        # Always save to outputs/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_type}_{timestamp}.json"
        outputs_path = self.output_dir / filename
        with open(outputs_path, "w") as f:
            json.dump(data, f, indent=2)

        # Always save a copy to Desktop
        desktop_path = Path.home() / "Desktop" / filename
        with open(desktop_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Output saved to {outputs_path} and {desktop_path}")

        CHAT_LIMIT = 2000
        if len(formatted) <= CHAT_LIMIT:
            chat_body = formatted
        else:
            chat_body = formatted[:CHAT_LIMIT] + "\n... [truncated]"

        return (
            f"{chat_body}\n\n"
            f"💾 Saved to:\n"
            f"  • outputs/{filename}\n"
            f"  • Desktop/{filename}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Delegate research and coding tasks to Gemini API"
    )
    parser.add_argument("--type", choices=["research", "code"], required=True,
                        help="Task type")
    parser.add_argument("--query", type=str,
                        help="Research query (for --type research)")
    parser.add_argument("--request", type=str,
                        help="Code request (for --type code)")
    parser.add_argument("--level", choices=["basic", "intermediate", "advanced", "expert"],
                        default="intermediate",
                        help="Reasoning/complexity level")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        metavar="MODEL",
                        help=(
                            "Model alias. Choices: "
                            + ", ".join(MODEL_REGISTRY.keys())
                            + "  (default: gemini25pro)"
                        ))

    args = parser.parse_args()

    delegator = GeminiDelegator()

    try:
        if args.type == "research":
            if not args.query:
                parser.error("--query required for research tasks")
            result = delegator.delegate_research(args.query, args.level, args.model)
        else:  # code
            if not args.request:
                parser.error("--request required for code tasks")
            result = delegator.delegate_code(args.request, args.level, args.model)

        print(result)

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
