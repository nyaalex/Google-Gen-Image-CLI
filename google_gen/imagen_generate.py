#!/usr/bin/env python3
"""Generate images with Google's Imagen models from the command line."""

import argparse
from pathlib import Path
from secrets import token_hex

from google import genai
from google.genai import types


def main(args) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an image using Google's Imagen image generation models."
    )
    parser.add_argument("prompt", help="The text prompt for the image.")
    parser.add_argument(
        "-o",
        "--output",
        default="imagen.png",
        help="Path to save the generated image(s) (default: %(default)s). "
        "A number will be appended for multiple images (e.g., imagen-1.png, imagen-2.png)",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1,
        help="Number of images to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="imagen-4.0-generate-001",
        help="Imagen model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
        help="Aspect ratio of the generated image(s) (default: %(default)s)",
    )
    args = parser.parse_args(args)

    client = genai.Client()
    config = types.GenerateImagesConfig(
        number_of_images=args.number,
        aspect_ratio=args.aspect_ratio,
        include_rai_reason=True,
    )

    try:
        response = client.models.generate_images(
            model=args.model,
            prompt=args.prompt,
            config=config,
        )
    except Exception as exc:  # pragma: no cover - surfaced as CLI error text.
        parser.error(f"API call failed: {exc}")

    output_path = Path(args.output)
    token = token_hex(4)
    parent = output_path.parent or Path("..")
    stem = output_path.stem or "image"
    suffix = output_path.suffix or ".png"

    parent.mkdir(parents=True, exist_ok=True)

    generated_images = getattr(response, "generated_images", [])
    if not generated_images:
        parser.error("No images were generated.")

    for i, image in enumerate(generated_images):
        image_bytes = image.image.image_bytes
        if not image_bytes:
            print(image.rai_filtered_reason)
            print(f"Warning: No image data for generated image {i+1}.")
            continue

        unique_path = parent / f"{stem}-{token}-{i+1}{suffix}"

        unique_path.write_bytes(image_bytes)
        print(f"Saved image to {unique_path.resolve()}")