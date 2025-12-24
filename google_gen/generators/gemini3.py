import mimetypes
import os
from typing import Generator

from google import genai

from google_gen.generators import BaseGenerator
from PIL import Image
from google.genai import types


class Gemini3(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)
        self.bypass_content = None

        if args.image is not None:
            for img in args.image:
                self.images.append(Image.open(img))

        if args.bypass:
            print("[*] Getting content for bypass.")
            # Get API key from args (which may override environment) or environment
            api_key = self.args.api_key or os.getenv('GEMINI_API_KEY', None)
            if api_key:
                client = genai.Client(api_key=api_key)
            else:
                client = genai.Client()
            self.bypass_content = []
            for i, image in enumerate(self.images):
                config_dict = {
                    "response_modalities": ["IMAGE"],
                    "image_config": types.ImageConfig(
                        aspect_ratio=self.args.aspect_ratio,
                        image_size=self.args.resolution,
                    ),
                    "thinking_config": types.ThinkingConfig(
                        include_thoughts=True
                    ),
                }
                
                http_options = self._get_http_options()
                if http_options:
                    config_dict["http_options"] = http_options
                
                unedited = client.models.generate_content(
                    model='gemini-3-pro-image-preview',
                    contents=["add a single translucent pixel to the bottom right of the image"] + [image],
                    config=types.GenerateContentConfig(**config_dict),
                )
                if unedited.parts is not None and unedited.parts[0].inline_data is not None:
                    print(f"[*] Successfully generated a bypass content for image ({i+1}/{len(self.images)})")
                    # unedited.parts[0].as_image().save("test.jpeg")
                    self.bypass_content.append(unedited.candidates[0].content)
                else:
                    raise Exception(f"Failed to generate bypass content for image ({i+1}/{len(self.images)})")

    @staticmethod
    def setup_args(parser):
        parser.add_argument(
            "-i",
            "--image",
            action="append",
            help="Paths to any images to attach",
        )
        parser.add_argument(
            "--aspect-ratio",
            choices=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            help="Aspect ratio to request",
        )
        parser.add_argument(
            "--resolution",
            choices=["1K", "2K", "4K"],
            help="Image resolution",
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
        parser.add_argument(
            "-b",
            "--bypass",
            action="store_true",
            help="Attempt to bypass the safeties on img&txt2img (requires one extra image generation) (default: %(default)s)",
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

        tools = []
        if self.args.search:
            tools.append({"google_search": {}})

        if self.bypass_content is None:
            contents = [prompt] + self.images

        else:
            contents = self.bypass_content + [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                ),
            ]

        config_dict = {
            "response_modalities": ["IMAGE"],
            "image_config": types.ImageConfig(
                aspect_ratio=self.args.aspect_ratio,
                image_size=self.args.resolution,
            ),
            "thinking_config": types.ThinkingConfig(
                include_thoughts=True
            ),
        }
        
        http_options = self._get_http_options()
        if http_options:
            config_dict["http_options"] = http_options
        
        response = self.client.models.generate_content(
            model='gemini-3-pro-image-preview',
            contents=contents,
            config=types.GenerateContentConfig(**config_dict),
        )
        results = [(image_bytes, 'jpeg') for image_bytes in self._extract_image_bytes(response)]
        if not results:
            raise Exception(f"No images returned from API.")

        return results
