#!/usr/bin/env python3
"""Generate an image with Google's Gemini Imagen models from the command line."""

import argparse
import os
from pathlib import Path
from google import genai
from google.genai import types
from google_gen.utils import get_name
from typing import Any, Iterable, Optional
from multiprocessing import Pool


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
        "-r",
        "--retries",
        default=3,
        type=int,
        help="Number of times to retry the prompt in case of errors (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1,
        help="Number of images to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--enhance",
        action="store_true",
        help="Enhance the prompt using gemini-2.5-flash (default: %(default)s)",
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

    threads = os.getenv('THREAD_COUNT', 5)

    args = parser.parse_args(args)
    with Pool(threads) as p:
        p.map(generate, (args for _ in range(args.number)))



def generate(args):
    client = genai.Client()

    prompt = args.prompt
    images= []

    output_path = Path(args.output)
    unique_path = get_name(output_path)

    if args.image is not None:
        from PIL import Image
        for img in args.image:
            images.append(Image.open(img))

    for _ in range(args.retries):
        try:
            if args.enhance:
                print("Enhancing prompt...")
                enhancer_model = "gemini-2.5-flash"
                system_instruction = (
                    "You are an expert image generation prompt writer. "
                    "Please take the following prompt and enhance it to be more descriptive and "
                    "suitable for a text-to-image model. Make it vivid and detailed, but keep it concise. "
                    "Only return the enhanced prompt, without any preamble or explanation."
                )
                response = client.models.generate_content(
                    model=enhancer_model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=[args.prompt] + images,
                )
                prompt = response.text.strip()
                print(f"Prompt for {unique_path}: {prompt}")

            response = client.models.generate_content(
                model=args.model,
                contents=[prompt] + images,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=args.aspect_ratio),
                ),
            )
            image_bytes = _extract_image_bytes(response)
            break

        except Exception as exc:  # pragma: no cover - surfaced as CLI error text.
            print("Failed to generate image:", exc)
    else:
        print("Failed to generate image")
        return


    unique_path.write_bytes(image_bytes)

    print(f"Saved image to {unique_path.resolve()}")
