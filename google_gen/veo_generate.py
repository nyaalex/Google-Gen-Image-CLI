#!/usr/bin/env python3
"""Generate a video with Google's Veo models from the command line."""

import re
import time
import argparse
import mimetypes
from pathlib import Path
from google import genai
from google.genai import types

from google_gen.utils import get_name


def open_image(image_path):
    img_mime, _ = mimetypes.guess_type(image_path)
    file = open(image_path, "rb")
    img_bytes = file.read()
    file.close()
    return types.Image(image_bytes=img_bytes, mime_type=img_mime)


def main(args) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a video using Google's Veo video generation models."
    )
    parser.add_argument("prompt", help="The text prompt for the video")
    parser.add_argument(
        "-o",
        "--output",
        default="veo.mp4",
        help="Path to save the generated video (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--retries",
        default=3,
        type=int,
        help="Number of times to retry the prompt in case of errors (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        help="Path to a source image",
    )
    parser.add_argument(
        "-l",
        "--last",
        default=None,
        help="Path to a source image for the last generated video",
    )
    parser.add_argument(
        "-a",
        "--asset",
        default=None,
        action="append",
        help="Path to an asset image",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=["16:9", "9:16"],
        help="Aspect ratio of the generated video (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="veo-3.1-generate-preview",
        help="Veo model to use (default: %(default)s)",
    )

    args = parser.parse_args(args)
    client = genai.Client()

    videos_source = types.GenerateVideosSource(prompt=args.prompt)
    config = types.GenerateVideosConfig(aspect_ratio=args.aspect_ratio)

    if args.source:
        videos_source.image = open_image(args.source)

    if args.last:
        config.last_frame = open_image(args.last)

    if args.asset:
        if len(args.asset) > 3:
            parser.error("There can not be more than three image assets")

        config.reference_images = []
        for asset in args.asset:
            config.reference_images.append(types.VideoGenerationReferenceImage(
                image=open_image(asset),
                reference_type=types.VideoGenerationReferenceType.ASSET,
            ))

    for _ in range(args.retries):
        try:
            operation = client.models.generate_videos(
                model=args.model,
                source=videos_source,
                config=config,

            )
        except Exception as exc:  # pragma: no cover - surfaced as CLI error text.
            print(exc)
            continue

        print("Waiting for video generation to complete...")
        while not operation.done:
            time.sleep(10)
            operation = client.operations.get(operation)

        if not operation.response.generated_videos:
            if operation.response.rai_media_filtered_reasons:
                print('\n'.join(operation.response.rai_media_filtered_reasons))
                continue

        break
    else:
        print("Video generation failed.")
        return

    generated_video = operation.response.generated_videos[0]

    output_path = Path(args.output)
    unique_path = get_name(output_path)
    client.files.download(file=generated_video.video)
    generated_video.video.save(str(unique_path))
    print(f"Saved video to {unique_path.resolve()}")

    file_name = re.findall('files/[^:]*', generated_video.video.uri)
    client.files.delete(name=file_name[0])
