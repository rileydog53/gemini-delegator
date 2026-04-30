#!/usr/bin/env python3
"""
Gemini Delegator — AI-to-AI delegation for research and coding tasks.
Routes requests to Gemini API with reasoning control and JSON response format.
"""

import re
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
# "text"      → generate_content  (research / code tasks)
# "image"     → generate_images   (Imagen models — use --type image)
# "video"     → generate_video    (Veo models — registered, full impl TBD)
# "tts"       → generate_content  (audio output — registered, full impl TBD)
# "audio"     → generate_content  (native audio I/O — registered, full impl TBD)
# "embedding" → embed_content     (vector embeddings)
# "aqa"       → generate_answer   (grounded Q&A with citations)
MODEL_REGISTRY = {
    # ── Gemini 2.5 ────────────────────────────────────────────────────────────
    # tier: confirmed via live testing. "free" = works on free API key. "paid" = limit:0 on free tier.
    "gemini25pro":        {"id": "gemini-2.5-pro",                              "type": "text",  "tier": "paid",    "desc": "Best overall. Deep reasoning + thinking tokens. Ideal for hard research, complex code, long documents.",           "best_for": "Hardest research tasks, expert-level code, massive documents. Best reasoning available — requires paid plan."},
    "gemini25flash":      {"id": "gemini-2.5-flash",                            "type": "text",  "tier": "free",    "desc": "Fast and smart. Best all-rounder — most tasks at a fraction of Pro's latency.",                                   "best_for": "Everyday research, code generation, Q&A, summaries. Best default choice for most tasks on free tier."},
    "pro":                {"id": "gemini-2.5-pro",                              "type": "text",  "tier": "paid",    "desc": "Alias for gemini25pro. Best overall reasoning.",                                                                  "best_for": "Same as gemini25pro. Use that alias instead — requires paid plan."},
    "flash":              {"id": "gemini-2.5-flash",                            "type": "text",  "tier": "unknown", "desc": "Alias for gemini25flash. Fast and capable.",                                                                      "best_for": "Same as gemini25flash. Use that alias instead."},
    "gemini25flashimage": {"id": "gemini-2.5-flash-image",                      "type": "text",  "tier": "paid",    "desc": "2.5 Flash that can generate images inline within a text response.",                                              "best_for": "Text responses that include generated images inline — requires paid plan."},
    "gemini25computeruse":{"id": "gemini-2.5-computer-use-preview-10-2025",     "type": "text",  "tier": "paid",    "desc": "Trained to understand screenshots and plan GUI/desktop interactions.",                                            "best_for": "Describing what to click or type in a UI from a screenshot — requires paid plan."},
    # ── Gemini 2.0 ────────────────────────────────────────────────────────────
    "gemini20flash":      {"id": "gemini-2.0-flash",                            "type": "text",  "tier": "paid",    "desc": "Previous-gen speed model. Reliable, fast, and well-tested for general tasks.",                                    "best_for": "High-volume reliable tasks where 2.5 Flash isn't available — requires paid plan."},
    "gemini20lite":       {"id": "gemini-2.0-flash-lite",                       "type": "text",  "tier": "paid",    "desc": "Fastest and cheapest Gemini model. Best for high-volume simple tasks.",                                          "best_for": "Ultra-cheap simple completions at scale — requires paid plan."},
    # ── Gemini 3.x ────────────────────────────────────────────────────────────
    "gemini3pro":         {"id": "gemini-3-pro-preview",                        "type": "text",  "tier": "unknown", "desc": "Next-gen flagship (preview). Stronger reasoning than 2.5 Pro.",                                                  "best_for": "Most complex reasoning available in the 3.x line — not confirmed on free tier."},
    "gemini3flash":       {"id": "gemini-3-flash-preview",                      "type": "text",  "tier": "free",    "desc": "Next-gen fast model (preview). Better quality than 2.5 Flash.",                                                  "best_for": "When you want newer-generation responses than gemini25flash. Slightly more current behavior."},
    "gemini3proimage":    {"id": "gemini-3-pro-image-preview",                  "type": "text",  "tier": "paid",    "desc": "Gemini 3 Pro with native image generation baked into text responses.",                                           "best_for": "Text + inline images in one response — requires paid plan."},
    # ── Gemini 3.1 ────────────────────────────────────────────────────────────
    "gemini31pro":        {"id": "gemini-3.1-pro-preview",                      "type": "text",  "tier": "unknown", "desc": "Latest flagship (preview). Best available reasoning across all models.",                                          "best_for": "Cutting-edge reasoning tasks — not confirmed on free tier."},
    "gemini31lite":       {"id": "gemini-3.1-flash-lite-preview",               "type": "text",  "tier": "unavailable", "desc": "Ultra-fast and cheap 3.1 model. Currently unavailable (503 error).",                                                "best_for": "[unavailable] Simple lookups that need speed."},
    "gemini31flashimage": {"id": "gemini-3.1-flash-image-preview",              "type": "text",  "tier": "paid",    "desc": "3.1 Flash with native inline image generation capability.",                                                       "best_for": "Inline image generation in text responses — requires paid plan."},
    "gemini31live":       {"id": "gemini-3.1-flash-live-preview",               "type": "text",  "tier": "unavailable", "desc": "Optimized for real-time streaming and live conversation sessions.",                                               "best_for": "[unavailable] Real-time streaming chat — not available on free tier (404 error)."},
    "gemini31customtools":{"id": "gemini-3.1-pro-preview-customtools",          "type": "text",  "tier": "unknown", "desc": "3.1 Pro with extended custom tool-calling support for agentic workflows.",                                        "best_for": "Agentic pipelines with custom tool definitions — not confirmed on free tier."},
    # ── Deep Research ─────────────────────────────────────────────────────────
    "deepresearch":       {"id": "deep-research-preview-04-2026",               "type": "text",  "tier": "paid",    "desc": "Autonomously browses the web and synthesizes multi-step research reports.",                                        "best_for": "Multi-source web research reports with citations — requires paid plan."},
    "deepresearchpro":    {"id": "deep-research-pro-preview-12-2025",           "type": "text",  "tier": "paid",    "desc": "Deep Research at pro-tier thoroughness. More sources, deeper analysis.",                                           "best_for": "Exhaustive research with more sources and deeper cross-referencing — requires paid plan."},
    "deepresearchmax":    {"id": "deep-research-max-preview-04-2026",           "type": "text",  "tier": "paid",    "desc": "Maximum-effort web research. Most thorough available, slowest.",                                                   "best_for": "Maximum-depth research tasks where time doesn't matter — requires paid plan."},
    # ── Stable "latest" aliases ───────────────────────────────────────────────
    "flashlatest":        {"id": "gemini-flash-latest",                         "type": "text",  "tier": "free",    "desc": "Always points to Google's current production Flash release. Never deprecated.",                                    "best_for": "Scripts and automations that need to keep working across Google releases without manual updates."},
    "flashlitelatest":    {"id": "gemini-flash-lite-latest",                    "type": "text",  "tier": "free",    "desc": "Always points to the current production Flash Lite release.",                                                     "best_for": "Lightweight automations that should auto-track the latest lite release indefinitely."},
    "prolatest":          {"id": "gemini-pro-latest",                           "type": "text",  "tier": "unknown", "desc": "Always points to the current production Pro release.",                                                            "best_for": "Pro-quality tasks via a stable alias — not confirmed on free tier."},
    # ── Gemma 3 (open-weight, confirmed free) ────────────────────────────────
    # NOTE: Gemma 3.x models do NOT support system_instruction OR response_mime_type=application/json
    "gemma31b":           {"id": "gemma-3-1b-it",                               "type": "text",  "tier": "free",    "no_system_instruction": True, "no_json_mime": True, "desc": "1B open-weight. Ultra-light; designed for on-device or offline use.",                                            "best_for": "Speed-critical tasks where quality matters less. Instant responses, minimal compute."},
    "gemma34b":           {"id": "gemma-3-4b-it",                               "type": "text",  "tier": "free",    "no_system_instruction": True, "no_json_mime": True, "desc": "4B open-weight. Small but capable for general everyday tasks.",                                                  "best_for": "Simple everyday queries — summaries, basic Q&A, short code snippets."},
    "gemma312b":          {"id": "gemma-3-12b-it",                              "type": "text",  "tier": "free",    "no_system_instruction": True, "no_json_mime": True, "desc": "12B open-weight. Good balance of quality and speed.",                                                            "best_for": "General research and code where you want open-weight quality without the heaviest models."},
    "gemma327b":          {"id": "gemma-3-27b-it",                              "type": "text",  "tier": "free",    "no_system_instruction": True, "no_json_mime": True, "desc": "27B open-weight. Best quality in Gemma 3 — near frontier model performance.",                                    "best_for": "Complex reasoning, detailed writing, hard code — highest free-tier quality in the Gemma 3 line."},
    "gemma3ne2b":         {"id": "gemma-3n-e2b-it",                             "type": "text",  "tier": "free",    "no_system_instruction": True, "no_json_mime": True, "desc": "2B multimodal nano model. Handles audio, image, and text in a tiny footprint.",                                   "best_for": "Mixed-input tasks (text + images or audio) where you need a tiny, efficient model."},
    "gemma3ne4b":         {"id": "gemma-3n-e4b-it",                             "type": "text",  "tier": "free",    "no_system_instruction": True, "no_json_mime": True, "desc": "4B multimodal nano. Same as 3ne2b but larger and more capable.",                                                "best_for": "Multimodal tasks needing slightly more capability than gemma3ne2b."},
    # ── Gemma 4 (open-weight, confirmed free) ────────────────────────────────
    "gemma4":             {"id": "gemma-4-26b-a4b-it",                          "type": "text",  "tier": "free",    "desc": "26B mixture-of-experts; only 4B active at once. Fast, efficient, punches above weight.",                          "best_for": "Demanding tasks needing near-Pro quality with low compute cost. Best efficiency on free tier."},
    "gemma431b":          {"id": "gemma-4-31b-it",                              "type": "text",  "tier": "free",    "desc": "31B dense open-weight model. Highest quality in the Gemma 4 line.",                                             "best_for": "Highest-quality outputs available on free tier. Use for long, complex, or detailed tasks."},
    # ── Gemini Robotics (confirmed free) ─────────────────────────────────────
    "robotics15":         {"id": "gemini-robotics-er-1.5-preview",              "type": "text",  "tier": "unavailable", "desc": "Embodied reasoning for robotics and physical agent planning.",                                                    "best_for": "[unavailable] Spatial reasoning, robotics planning — not found (404 error)."},
    "robotics16":         {"id": "gemini-robotics-er-1.6-preview",              "type": "text",  "tier": "free",    "desc": "Robotics v1.6 — improved 3D spatial and physical understanding.",                                               "best_for": "Spatial reasoning, robotics planning, 3D scene understanding, physical world questions."},
    # ── Experimental ──────────────────────────────────────────────────────────
    "nanobanana":         {"id": "nano-banana-pro-preview",                     "type": "text",  "tier": "paid",    "desc": "Google internal experimental model. Behavior not fully documented.",                                             "best_for": "Curiosity only — undocumented experimental model, requires paid plan."},
    # ── Image generation (Imagen) — confirmed paid-only ───────────────────────
    "imagen4":            {"id": "imagen-4.0-generate-001",                     "type": "image", "tier": "paid",    "desc": "Imagen 4 standard. Photorealistic, high detail. Best default choice.",                                           "best_for": "Photorealistic AI image generation from text prompts — requires paid plan."},
    "imagen4fast":        {"id": "imagen-4.0-fast-generate-001",                "type": "image", "tier": "paid",    "desc": "Imagen 4 Fast. Same quality target as standard, noticeably quicker.",                                           "best_for": "Same as imagen4 but faster — use when you need many images quickly. Requires paid plan."},
    "imagen4ultra":       {"id": "imagen-4.0-ultra-generate-001",               "type": "image", "tier": "paid",    "desc": "Imagen 4 Ultra. Highest possible image quality. Slower and may cost more.",                                      "best_for": "Maximum image quality — hero shots, print-quality output. Requires paid plan."},
    # ── Video generation (Veo) — registered; full --type video impl TBD ───────
    "veo2":               {"id": "veo-2.0-generate-001",                        "type": "video",     "desc": "Veo 2. Generates cinematic HD video clips from a text description."},
    "veo3":               {"id": "veo-3.0-generate-001",                        "type": "video",     "desc": "Veo 3. Video + synchronized native audio (ambient sound, speech, music)."},
    "veo3fast":           {"id": "veo-3.0-fast-generate-001",                   "type": "video",     "desc": "Veo 3 Fast. Quick video clip generation with audio."},
    "veo31":              {"id": "veo-3.1-generate-preview",                    "type": "video",     "desc": "Veo 3.1. Latest and highest quality video generation (preview)."},
    "veo31fast":          {"id": "veo-3.1-fast-generate-preview",               "type": "video",     "desc": "Veo 3.1 Fast. Speed-optimized version of Veo 3.1."},
    "veo31lite":          {"id": "veo-3.1-lite-generate-preview",               "type": "video",     "desc": "Veo 3.1 Lite. Lightweight, short clips with lower compute cost."},
    # ── TTS / Audio — registered; full --type tts/audio impl TBD ─────────────
    "tts25flash":         {"id": "gemini-2.5-flash-preview-tts",                "type": "tts",       "desc": "2.5 Flash TTS. Fast, natural-sounding text-to-speech output."},
    "tts25pro":           {"id": "gemini-2.5-pro-preview-tts",                  "type": "tts",       "desc": "2.5 Pro TTS. Highest quality voice output — best for important audio."},
    "tts31flash":         {"id": "gemini-3.1-flash-tts-preview",                "type": "tts",       "desc": "3.1 Flash TTS. Next-gen voice generation (preview)."},
    "audio25flash":       {"id": "gemini-2.5-flash-native-audio-latest",        "type": "audio",     "desc": "Native audio I/O. Can understand spoken input AND generate spoken output."},
    # ── Embeddings ────────────────────────────────────────────────────────────
    "embedding":          {"id": "gemini-embedding-001",                        "type": "embedding", "desc": "Converts text to a semantic vector for similarity search, clustering, or RAG."},
    "embedding2":         {"id": "gemini-embedding-2",                          "type": "embedding", "desc": "Embedding v2. Improved accuracy over v1 for semantic search tasks."},
    # ── Grounded Q&A ─────────────────────────────────────────────────────────
    "aqa":                {"id": "aqa",                                         "type": "aqa",       "desc": "Grounded Q&A. Answers questions with citations tied to documents you provide."},
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


class PaidTierRequired(Exception):
    """Raised when a paid-tier model is selected but --force was not passed."""


class GeminiDelegator:
    def __init__(self, config_path="gemini_delegator_config.yaml", force=False):
        """Initialize delegator with config and API key."""
        self.config_path = Path(__file__).parent / config_path
        self.config = self._load_config()
        self.api_key = self._load_api_key()
        self.client = self._init_client()
        self.output_dir = Path(__file__).parent / self.config["paths"]["outputs"]
        self.output_dir.mkdir(exist_ok=True)
        self.force = force

    def _check_tier(self, entry):
        """Refuse unavailable and paid-tier models unless --force was passed."""
        tier = entry.get("tier")
        if tier == "unavailable":
            raise ValueError(
                f"Model '{entry['id']}' is not available. "
                f"(It may have been withdrawn or is temporarily down.)"
            )
        if tier == "paid" and not self.force:
            raise PaidTierRequired(
                f"Model '{entry['id']}' is paid-tier on this API key "
                f"(registry tier='paid'). Pass --force to attempt anyway."
            )

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

    def _call_gemini_with_retry(self, prompt, model, thinking_budget, skip_system_instruction=False, skip_json_mime=False, max_retries=1):
        """Call Gemini API with rate limiting and error handling."""
        # Models confirmed to support thinking/reasoning tokens
        THINKING_CAPABLE = {"gemini-2.5-flash", "gemini-3-flash-preview", "gemini-flash-latest"}
        supports_thinking = model in THINKING_CAPABLE

        # Build config conditionally based on model capabilities
        config_kwargs = {}
        if not skip_json_mime:
            config_kwargs["response_mime_type"] = "application/json"
        if not skip_system_instruction:
            config_kwargs["system_instruction"] = SYSTEM_INSTRUCTION

        call_config = GenerateContentConfig(**config_kwargs)

        if supports_thinking and thinking_budget > 0:
            call_config.thinking_config = ThinkingConfig(
                thinking_budget=thinking_budget
            )

        # max_retries=1 means: one initial attempt + one retry on transient 429.
        # Hard-quota (limit: 0) failures fail-fast without retrying.
        attempts = max_retries + 1
        for attempt in range(attempts):
            try:
                # Rate limiting: 4 seconds between requests (15 RPM = ~4 sec/request)
                time.sleep(4)

                model_name = model

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=call_config,
                )

                logger.info(f"✓ Gemini response received from {model}")
                return response.text

            except (APIError, ClientError) as e:
                err_str = str(e)
                is_429 = "429" in err_str or "Resource Exhausted" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if not is_429:
                    logger.error(f"API Error: {e}")
                    raise

                # Hard-quota detection: free tier with limit:0 won't recover via retry.
                if "limit: 0" in err_str or "limit:0" in err_str:
                    logger.error(f"Hard quota (limit: 0) on {model} — paid tier required. Failing fast.")
                    raise

                if attempt >= attempts - 1:
                    logger.error(f"Failed after {attempts} attempts: {e}")
                    raise

                # Honor server-provided retryDelay if present (e.g. 'retryDelay': '49s').
                m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+)s", err_str)
                wait_time = int(m.group(1)) + 1 if m else 30
                logger.warning(f"Rate limit hit (429). Waiting {wait_time}s... (Attempt {attempt+1}/{attempts})")
                print(f"Rate limit reached. Waiting {wait_time}s before retry...", file=sys.stderr)
                time.sleep(wait_time)

    def delegate_research(self, query, level="intermediate", model=None):
        """Delegate research task to Gemini."""
        entry = resolve_model(model or DEFAULT_MODEL_ALIAS)
        self._check_tier(entry)
        model_name = entry["id"]
        thinking_budget = self._get_reasoning_budget("research", level)
        skip_sys = entry.get("no_system_instruction", False)
        skip_json = entry.get("no_json_mime", False)

        logger.info(f"Research task: {query[:50]}... (level: {level}, model: {model_name})")
        prompt = self._build_research_prompt(query, level)

        response_text = self._call_gemini_with_retry(prompt, model_name, thinking_budget, skip_system_instruction=skip_sys, skip_json_mime=skip_json)
        return self._parse_and_format_response(response_text, "research", query)

    def delegate_code(self, request, level="intermediate", model=None):
        """Delegate code generation task to Gemini."""
        entry = resolve_model(model or DEFAULT_MODEL_ALIAS)
        self._check_tier(entry)
        model_name = entry["id"]
        thinking_budget = self._get_reasoning_budget("code", level)
        skip_sys = entry.get("no_system_instruction", False)
        skip_json = entry.get("no_json_mime", False)

        logger.info(f"Code task: {request[:50]}... (level: {level}, model: {model_name})")
        prompt = self._build_code_prompt(request, level)

        response_text = self._call_gemini_with_retry(prompt, model_name, thinking_budget, skip_system_instruction=skip_sys, skip_json_mime=skip_json)
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
                        ))
    parser.add_argument("--force", action="store_true",
                        help="Bypass the paid-tier pre-flight check and attempt the call anyway.")

    args = parser.parse_args()

    # Warn if a non-text model is passed to a text task
    if args.model:
        entry = MODEL_REGISTRY.get(args.model, {})
        model_type = entry.get("type", "text")
        if model_type != "text":
            print(
                f"Note: '{args.model}' is a {model_type} model. "
                f"Full --type {model_type} support is not yet implemented — "
                f"sending as-is; expect an API error.",
                file=sys.stderr
            )

    delegator = GeminiDelegator(force=args.force)

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
