#!/usr/bin/env python3
"""Generate images with Google's Imagen models from the command line."""

import argparse
from pathlib import Path
from google import genai
from google.genai import types

from google_gen.utils import get_name


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
    names = get_name(output_path, multiple=True)

    generated_images = getattr(response, "generated_images", [])
    if not generated_images:
        parser.error("No images were generated.")

    for unique_path, image in zip(names, generated_images):
        image_bytes = image.image.image_bytes
        if not image_bytes:
            print(image.rai_filtered_reason)
            print(f"Warning: No image data for generated image.")
            continue

        unique_path.write_bytes(image_bytes)
        print(f"Saved image to {unique_path.resolve()}")
