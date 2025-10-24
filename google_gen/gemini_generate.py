#!/usr/bin/env python3
"""Generate an image with Google's Gemini Imagen models from the command line."""

import argparse
from pathlib import Path
from google import genai
from google.genai import types
from google_gen.utils import get_name
from typing import Any, Iterable, Optional


def _first(iterable: Iterable[Any]) -> Optional[Any]:
    for item in iterable:
        return item
    return None


def _extract_image_bytes(response: Any) -> bytes:
    """Pull the first image payload from a google-genai response."""
    candidates = getattr(response, "candidates", None)
    if candidates:
        first_candidate = _first(candidates)
        if first_candidate and getattr(first_candidate, "content", None):
            parts = getattr(first_candidate.content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    return inline_data.data

    raise RuntimeError("No image data returned from the API.")


def main(args) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an image using Google's Gemini image generation models."
    )
    parser.add_argument("prompt", help="The text prompt for the image.")
    parser.add_argument(
        "-o",
        "--output",
        default="gemini.png",
        help="Path to save the generated image (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--image",
        default=None,
        action="append",
        help="Paths to any images to attach",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="3:4",
        choices=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],
        help="Aspect ratio to request (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help="Gemini image model to use (default: %(default)s)",
    )
    args = parser.parse_args(args)
    client = genai.Client()

    contents = [args.prompt]
    if args.image is not None:
        from PIL import Image
        for img in args.image:
            contents.append(Image.open(img))

    try:
        response = client.models.generate_content(
            model=args.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=args.aspect_ratio),
            ),
        )
    except Exception as exc:  # pragma: no cover - surfaced as CLI error text.
        parser.error(f"API call failed: {exc}")

    image_bytes = _extract_image_bytes(response)

    output_path = Path(args.output)
    unique_path = get_name(output_path)
    unique_path.write_bytes(image_bytes)

    print(f"Saved image to {unique_path.resolve()}")
