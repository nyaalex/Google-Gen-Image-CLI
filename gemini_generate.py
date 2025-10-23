#!/usr/bin/env python3
"""Generate an image with Google's Gemini Imagen models from the command line."""

import argparse
import os
from pathlib import Path
from secrets import token_hex
from typing import Any, Iterable, Optional
from PIL import Image
from google import genai
from google.genai import types


def _first(iterable: Iterable[Any]) -> Optional[Any]:
    for item in iterable:
        return item
    return None


def _extract_image_bytes(response: Any) -> bytes:
    """Pull the first image payload from a google-genai response."""

    # generate_images returns a list of `ModelGeneratedImage`; generate_content wraps the
    # data in `response.candidates`. Inspect the first available payload.
    generated = getattr(response, "generated_images", None)
    if generated:
        first_image = _first(generated)
        if first_image and getattr(first_image, "image_bytes", None):
            return first_image.image_bytes

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


def main() -> None:
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
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("Set the GEMINI_API_KEY environment variable before running.")

    client = genai.Client(api_key=api_key)

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
    parent = output_path.parent or Path(".")
    stem = output_path.stem or "image"
    suffix = output_path.suffix or ".png"

    parent.mkdir(parents=True, exist_ok=True)

    unique_path = parent / f"{stem}-{token_hex(4)}{suffix}"
    unique_path.write_bytes(image_bytes)

    print(f"Saved image to {unique_path.resolve()}")


if __name__ == "__main__":
    main()
