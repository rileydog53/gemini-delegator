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
from google.genai.types import ThinkingConfig, GenerateContentConfig, GenerateImagesConfig

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
    "gemini25pro":        {"id": "gemini-2.5-pro",                              "type": "text"},  # default — deep reasoning + thinking
    "gemini25flash":      {"id": "gemini-2.5-flash",                            "type": "text"},  # fast, large context, research summaries
    "pro":                {"id": "gemini-2.5-pro",                              "type": "text"},  # short alias
    "flash":              {"id": "gemini-2.5-flash",                            "type": "text"},  # short alias
    "gemini25flashimage": {"id": "gemini-2.5-flash-image",                      "type": "text"},  # 2.5 Flash with native image generation
    "gemini25computeruse":{"id": "gemini-2.5-computer-use-preview-10-2025",     "type": "text"},  # GUI/desktop agent — sees and controls screens
    # ── Gemini 2.0 ────────────────────────────────────────────────────────────
    "gemini20flash":      {"id": "gemini-2.0-flash",                            "type": "text"},  # speed-optimized general tasks
    "gemini20lite":       {"id": "gemini-2.0-flash-lite",                       "type": "text"},  # highest-volume simple tasks
    # ── Gemini 3.x ────────────────────────────────────────────────────────────
    "gemini3pro":         {"id": "gemini-3-pro-preview",                        "type": "text"},  # next-gen deep reasoning
    "gemini3flash":       {"id": "gemini-3-flash-preview",                      "type": "text"},  # next-gen fast tasks
    "gemini3proimage":    {"id": "gemini-3-pro-image-preview",                  "type": "text"},  # Gemini 3 Pro + native image gen
    # ── Gemini 3.1 ────────────────────────────────────────────────────────────
    "gemini31pro":        {"id": "gemini-3.1-pro-preview",                      "type": "text"},  # 3.1 flagship — best reasoning
    "gemini31lite":       {"id": "gemini-3.1-flash-lite-preview",               "type": "text"},  # 3.1 lightweight — fast and cheap
    "gemini31flashimage": {"id": "gemini-3.1-flash-image-preview",              "type": "text"},  # 3.1 Flash with native image gen
    "gemini31live":       {"id": "gemini-3.1-flash-live-preview",               "type": "text"},  # 3.1 real-time streaming / live sessions
    "gemini31customtools":{"id": "gemini-3.1-pro-preview-customtools",          "type": "text"},  # 3.1 Pro with extended/custom tool use
    # ── Deep Research ─────────────────────────────────────────────────────────
    "deepresearch":       {"id": "deep-research-preview-04-2026",               "type": "text"},  # autonomous multi-step web research
    "deepresearchpro":    {"id": "deep-research-pro-preview-12-2025",           "type": "text"},  # deep research — pro-tier thoroughness
    "deepresearchmax":    {"id": "deep-research-max-preview-04-2026",           "type": "text"},  # deep research — maximum depth
    # ── Stable "latest" aliases ───────────────────────────────────────────────
    "flashlatest":        {"id": "gemini-flash-latest",                         "type": "text"},  # always points to latest Flash release
    "flashlitelatest":    {"id": "gemini-flash-lite-latest",                    "type": "text"},  # always points to latest Flash Lite release
    "prolatest":          {"id": "gemini-pro-latest",                           "type": "text"},  # always points to latest Pro release
    # ── Gemma 3 (open-weight) ─────────────────────────────────────────────────
    "gemma31b":           {"id": "gemma-3-1b-it",                               "type": "text"},  # 1B — ultra-light, on-device use cases
    "gemma34b":           {"id": "gemma-3-4b-it",                               "type": "text"},  # 4B — small but capable general tasks
    "gemma312b":          {"id": "gemma-3-12b-it",                              "type": "text"},  # 12B — balanced quality/speed
    "gemma327b":          {"id": "gemma-3-27b-it",                              "type": "text"},  # 27B — best open-weight quality
    "gemma3ne2b":         {"id": "gemma-3n-e2b-it",                             "type": "text"},  # 3n 2B — multimodal nano (audio/image/text)
    "gemma3ne4b":         {"id": "gemma-3n-e4b-it",                             "type": "text"},  # 3n 4B — multimodal nano, larger
    # ── Gemma 4 ───────────────────────────────────────────────────────────────
    "gemma4":             {"id": "gemma-4-26b-a4b-it",                          "type": "text"},  # 4 MoE — 26B params, 4B active (fast + efficient)
    "gemma431b":          {"id": "gemma-4-31b-it",                              "type": "text"},  # 4 Dense — 31B full model (highest open quality)
    # ── Gemini Robotics ───────────────────────────────────────────────────────
    "robotics15":         {"id": "gemini-robotics-er-1.5-preview",              "type": "text"},  # embodied reasoning for robots / physical agents
    "robotics16":         {"id": "gemini-robotics-er-1.6-preview",              "type": "text"},  # embodied reasoning v1.6 — improved spatial understanding
    # ── Experimental ──────────────────────────────────────────────────────────
    "nanobanana":         {"id": "nano-banana-pro-preview",                     "type": "text"},  # Google internal experimental model
    # ── Image generation (Imagen) ─────────────────────────────────────────────
    "imagen4":            {"id": "imagen-4.0-generate-001",                     "type": "image"}, # Imagen 4 — photorealistic, high detail
    "imagen4fast":        {"id": "imagen-4.0-fast-generate-001",                "type": "image"}, # Imagen 4 Fast — quick generation
    "imagen4ultra":       {"id": "imagen-4.0-ultra-generate-001",               "type": "image"}, # Imagen 4 Ultra — highest quality
    # ── Video generation (Veo) — registered; full --type video impl TBD ───────
    "veo2":               {"id": "veo-2.0-generate-001",                        "type": "video"}, # Veo 2 — cinematic HD video from text
    "veo3":               {"id": "veo-3.0-generate-001",                        "type": "video"}, # Veo 3 — video + synchronized native audio
    "veo3fast":           {"id": "veo-3.0-fast-generate-001",                   "type": "video"}, # Veo 3 Fast — quick clip generation
    "veo31":              {"id": "veo-3.1-generate-preview",                    "type": "video"}, # Veo 3.1 — latest, highest quality
    "veo31fast":          {"id": "veo-3.1-fast-generate-preview",               "type": "video"}, # Veo 3.1 Fast
    "veo31lite":          {"id": "veo-3.1-lite-generate-preview",               "type": "video"}, # Veo 3.1 Lite — lightweight clips
    # ── TTS / Audio — registered; full --type tts/audio impl TBD ─────────────
    "tts25flash":         {"id": "gemini-2.5-flash-preview-tts",                "type": "tts"},   # 2.5 Flash TTS — fast text-to-speech
    "tts25pro":           {"id": "gemini-2.5-pro-preview-tts",                  "type": "tts"},   # 2.5 Pro TTS — highest quality voice
    "tts31flash":         {"id": "gemini-3.1-flash-tts-preview",                "type": "tts"},   # 3.1 Flash TTS — next-gen voice
    "audio25flash":       {"id": "gemini-2.5-flash-native-audio-latest",        "type": "audio"}, # native audio I/O — understand + generate speech
    # ── Embeddings ────────────────────────────────────────────────────────────
    "embedding":          {"id": "gemini-embedding-001",                        "type": "embedding"}, # embedding v1 — semantic vector search
    "embedding2":         {"id": "gemini-embedding-2",                          "type": "embedding"}, # embedding v2 — improved accuracy
    # ── Grounded Q&A ─────────────────────────────────────────────────────────
    "aqa":                {"id": "aqa",                                         "type": "aqa"},   # cited Q&A against provided documents
}

DEFAULT_MODEL_ALIAS = "gemini25pro"


# ─── Image Style Presets ──────────────────────────────────────────────────────
# Use --style to apply one of these. Multiple styles can be stacked with commas.
STYLE_PRESETS = {
    # Photographic
    "photorealistic":  "photorealistic, ultra-detailed, sharp focus, professional photography, 8k",
    "cinematic":       "cinematic shot, dramatic composition, film grain, anamorphic lens, depth of field",
    "vintage-photo":   "vintage photograph, faded colors, film grain, 1970s aesthetic, kodachrome",
    "polaroid":        "polaroid photo, instant camera, soft focus, slightly washed out, nostalgic",
    "documentary":     "documentary photography, candid, natural light, photojournalism",
    "portrait":        "professional portrait, soft lighting, shallow depth of field, sharp eyes",
    "macro":           "macro photography, extreme close-up, hyperdetailed, shallow depth of field",
    "aerial":          "aerial photography, drone shot, top-down view, vast landscape",

    # Painting / Illustration
    "oil-painting":    "oil painting, visible brush strokes, classical art style, rich textures",
    "watercolor":      "watercolor painting, soft edges, flowing pigments, paper texture",
    "pencil-sketch":   "pencil sketch, graphite, hatching, monochrome, hand-drawn",
    "ink-drawing":     "black ink drawing, bold lines, crosshatching, high contrast",
    "impressionist":   "impressionist painting, visible brush strokes, soft lighting, Monet-inspired",
    "art-nouveau":     "art nouveau, ornate, flowing organic lines, Mucha-inspired",
    "ukiyo-e":         "ukiyo-e woodblock print, Japanese traditional art, flat colors, bold outlines",

    # Animation / Stylized
    "anime":           "anime style, cel shading, vibrant colors, expressive eyes, Japanese animation",
    "studio-ghibli":   "Studio Ghibli style, hand-painted backgrounds, whimsical, soft palette",
    "disney":          "Disney animation style, expressive characters, vibrant, family-friendly",
    "pixar":           "Pixar 3D animation style, expressive, soft lighting, family-friendly",
    "comic-book":      "comic book style, bold outlines, halftone shading, dynamic pose",
    "manga":           "black and white manga, screentone shading, dynamic linework",
    "cartoon":         "cartoon style, bold outlines, flat colors, exaggerated features",

    # Digital / Modern
    "digital-art":     "digital art, vibrant colors, smooth shading, ArtStation trending",
    "concept-art":     "concept art, detailed environment, atmospheric, professional illustration",
    "3d-render":       "3D render, octane, ray tracing, cinema 4D, hyperrealistic materials",
    "low-poly":        "low poly 3D, flat shading, geometric, minimalist 3D",
    "pixel-art":       "pixel art, 16-bit style, limited palette, retro game aesthetic",
    "vaporwave":       "vaporwave aesthetic, neon pink and blue, 80s retro, glitch art",
    "synthwave":       "synthwave, neon grid, sunset, retro-futuristic, 80s aesthetic",
    "cyberpunk":       "cyberpunk, neon-lit, dystopian city, holographic, rain-slicked streets",
    "steampunk":       "steampunk, brass gears, victorian, copper pipes, industrial",

    # Mood / Atmosphere
    "fantasy-art":     "fantasy art, magical, ethereal lighting, mystical atmosphere",
    "dark-fantasy":    "dark fantasy, gothic, moody, dramatic shadows, ominous",
    "film-noir":       "film noir, black and white, dramatic shadows, venetian blind lighting, 1940s",
    "minimalist":      "minimalist, clean composition, negative space, simple color palette",
    "surreal":         "surreal, dreamlike, impossible geometry, Dali-inspired",
    "horror":          "horror aesthetic, eerie, unsettling, dim lighting, macabre",
    "ethereal":        "ethereal, soft glow, dreamlike, otherworldly, luminous",

    # Pop / Graphic
    "pop-art":         "pop art, Lichtenstein style, bold colors, ben-day dots",
    "graffiti":        "graffiti street art, spray paint texture, urban, bold tags",
    "poster":          "movie poster style, dramatic, typography-friendly composition, bold",
    "infographic":     "clean infographic style, flat design, clear visual hierarchy",
}


# ─── Aspect Ratios ────────────────────────────────────────────────────────────
ASPECT_RATIOS = {
    "square":     "1:1",   # 1024x1024 — Instagram posts, profile pics
    "portrait":   "3:4",   # 896x1280 — vertical photos, book covers
    "landscape":  "4:3",   # 1280x896 — classic photo, presentation
    "wide":       "16:9",  # 1408x768 — desktop wallpaper, video thumb
    "tall":       "9:16",  # 768x1408 — phone wallpaper, story
    "1:1":        "1:1",
    "3:4":        "3:4",
    "4:3":        "4:3",
    "16:9":       "16:9",
    "9:16":       "9:16",
}


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

    def _build_image_prompt(self, prompt, style=None, lighting=None, mood=None,
                            quality=None, extra=None):
        """Compose final image prompt by stacking modifiers onto the base description."""
        parts = [prompt.strip()]

        # Stack styles (comma-separated supported)
        if style:
            for s in [x.strip() for x in style.split(",") if x.strip()]:
                if s in STYLE_PRESETS:
                    parts.append(STYLE_PRESETS[s])
                else:
                    parts.append(s)  # custom style string

        if lighting:
            parts.append(f"{lighting} lighting")
        if mood:
            parts.append(f"{mood} mood")
        if quality:
            parts.append(quality)
        if extra:
            parts.append(extra)

        return ", ".join(parts)

    def delegate_image(self, prompt, model=None, style=None, aspect_ratio="1:1",
                       num_images=1, lighting=None, mood=None, quality=None,
                       negative_prompt=None, person_generation="ALLOW_ADULT",
                       extra=None):
        """Delegate image generation to Imagen via Gemini API."""
        entry = resolve_model(model or "imagen4")
        model_name = entry["id"]

        if entry["type"] != "image":
            raise ValueError(f"Model '{model}' is not an image model. Use imagen4, imagen4fast, or imagen4ultra.")

        # Resolve aspect ratio alias
        ratio = ASPECT_RATIOS.get(aspect_ratio, aspect_ratio)

        final_prompt = self._build_image_prompt(
            prompt, style=style, lighting=lighting, mood=mood,
            quality=quality, extra=extra
        )

        logger.info(f"Image task: {prompt[:50]}... (model: {model_name}, ratio: {ratio}, n: {num_images})")
        logger.info(f"Final prompt: {final_prompt[:200]}")

        config_kwargs = {
            "number_of_images": num_images,
            "aspect_ratio": ratio,
            "person_generation": person_generation,
        }
        if negative_prompt:
            config_kwargs["negative_prompt"] = negative_prompt

        try:
            time.sleep(4)  # rate limit
            response = self.client.models.generate_images(
                model=model_name,
                prompt=final_prompt,
                config=GenerateImagesConfig(**config_kwargs),
            )
        except (APIError, ClientError) as e:
            logger.error(f"Image API Error: {e}")
            raise

        # Save each image to outputs/ and Desktop
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        desktop = Path.home() / "Desktop"

        for i, generated in enumerate(response.generated_images, start=1):
            suffix = f"_{i}" if num_images > 1 else ""
            filename = f"image_{timestamp}{suffix}.png"
            outputs_path = self.output_dir / filename
            desktop_path = desktop / filename

            image_bytes = generated.image.image_bytes
            with open(outputs_path, "wb") as f:
                f.write(image_bytes)
            with open(desktop_path, "wb") as f:
                f.write(image_bytes)

            saved_files.append(filename)
            logger.info(f"Image saved to {outputs_path} and {desktop_path}")

        files_listing = "\n".join(
            f"  • outputs/{name}\n  • Desktop/{name}" for name in saved_files
        )
        return (
            f"🖼️  Generated {len(saved_files)} image(s)\n"
            f"Prompt used: {final_prompt}\n"
            f"Aspect ratio: {ratio}  |  Model: {model_name}\n\n"
            f"💾 Saved to:\n{files_listing}"
        )

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
        description="Delegate research, coding, and image tasks to Gemini API"
    )
    parser.add_argument("--type", choices=["research", "code", "image"], required=True,
                        help="Task type")
    parser.add_argument("--query", type=str,
                        help="Research query (for --type research)")
    parser.add_argument("--request", type=str,
                        help="Code request (for --type code)")
    parser.add_argument("--prompt", type=str,
                        help="Image description (for --type image)")
    parser.add_argument("--level", choices=["basic", "intermediate", "advanced", "expert"],
                        default="intermediate",
                        help="Reasoning/complexity level (research/code only)")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        metavar="MODEL",
                        help=(
                            "Model alias. Choices: "
                            + ", ".join(MODEL_REGISTRY.keys())
                            + "  (research/code default: gemini25pro; image default: imagen4)"
                        ))

    # ── Image-specific options ────────────────────────────────────────────────
    img = parser.add_argument_group("image options (use with --type image)")
    img.add_argument("--style", type=str, default=None,
                     help=(
                         "Style preset(s), comma-separated. Custom strings allowed. "
                         "Presets: " + ", ".join(STYLE_PRESETS.keys())
                     ))
    img.add_argument("--aspect", type=str, default="1:1",
                     metavar="RATIO",
                     help=(
                         "Aspect ratio. Aliases: square, portrait, landscape, wide, tall. "
                         "Or pass directly: 1:1, 3:4, 4:3, 16:9, 9:16"
                     ))
    img.add_argument("--num", type=int, default=1, choices=[1, 2, 3, 4],
                     help="Number of images to generate (1-4)")
    img.add_argument("--lighting", type=str, default=None,
                     help="Lighting modifier (e.g. 'golden hour', 'studio', 'neon', 'soft')")
    img.add_argument("--mood", type=str, default=None,
                     help="Mood modifier (e.g. 'moody', 'cheerful', 'dramatic', 'serene')")
    img.add_argument("--quality", type=str, default=None,
                     help="Quality modifier (e.g. '8k', 'hyperdetailed', 'masterpiece')")
    img.add_argument("--negative", type=str, default=None,
                     help="Negative prompt (what to AVOID in the image)")
    img.add_argument("--extra", type=str, default=None,
                     help="Any extra freeform modifiers appended to the prompt")
    img.add_argument("--people", type=str, default="ALLOW_ADULT",
                     choices=["DONT_ALLOW", "ALLOW_ADULT", "ALLOW_ALL"],
                     help="Person generation policy (default: ALLOW_ADULT)")

    args = parser.parse_args()

    # Warn if a non-text/image model is passed to a text/image task
    if args.model:
        entry = MODEL_REGISTRY.get(args.model, {})
        model_type = entry.get("type", "text")
        if model_type in ("video", "tts", "audio") and args.type in ("research", "code", "image"):
            print(
                f"Note: '{args.model}' is a {model_type} model. "
                f"Full --type {model_type} support is not yet implemented — "
                f"sending as-is; expect an API error.",
                file=sys.stderr
            )

    delegator = GeminiDelegator()

    try:
        if args.type == "research":
            if not args.query:
                parser.error("--query required for research tasks")
            result = delegator.delegate_research(args.query, args.level, args.model)
        elif args.type == "code":
            if not args.request:
                parser.error("--request required for code tasks")
            result = delegator.delegate_code(args.request, args.level, args.model)
        else:  # image
            if not args.prompt:
                parser.error("--prompt required for image tasks")
            result = delegator.delegate_image(
                prompt=args.prompt,
                model=args.model,
                style=args.style,
                aspect_ratio=args.aspect,
                num_images=args.num,
                lighting=args.lighting,
                mood=args.mood,
                quality=args.quality,
                negative_prompt=args.negative,
                person_generation=args.people,
                extra=args.extra,
            )

        print(result)

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
