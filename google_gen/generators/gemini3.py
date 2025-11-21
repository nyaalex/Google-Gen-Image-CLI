from typing import Generator

from google_gen.generators import BaseGenerator
from PIL import Image
from google.genai import types


class Gemini3(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.image is not None:
            for img in args.image:
                self.images.append(Image.open(img))

    @staticmethod
    def setup_args(parser):
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
            choices=["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"],
            help="Aspect ratio to request (default: %(default)s)",
        )
        parser.add_argument(
            "--resolution",
            default="1K",
            choices=["1K", "2K", "4K"],
            help="Image resolution (default: %(default)s)",
        )
        parser.add_argument(
            "--search",
            action="store_true",
            help="Enable Google Search for image generation (default: %(default)s)",
        )
        parser.add_argument(
            "--thoughts",
            action="store_true",
            help="Show/save thoughts and thought images (default: %(default)s)",
        )

    def _extract_image_bytes(self, response: types.GenerateContentResponse) -> 'Generator[bytes]':
        for part in response.parts:
            if part.thought and self.args.thoughts:
                if part.text:
                    print(part.text)
                elif image := part.as_image():
                    yield image.image_bytes
            elif part.inline_data is not None:
                yield part.inline_data.data

    def generate(self, prompt: str) -> list[tuple[bytes, str]]:

        tools =[]
        if self.args.search:
            tools.append({"google_search": {}})

        response = self.client.models.generate_content(
            model='gemini-3-pro-image-preview',
            contents=[prompt] + self.images,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=self.args.aspect_ratio,
                    image_size=self.args.resolution,
                ),
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                ),
            ),
        )
        results = [(image_bytes, 'png') for image_bytes in self._extract_image_bytes(response)]
        if not results:
            raise Exception(f"No images returned from API.")

        return results
