import mimetypes

from google_gen.generators import BaseGenerator
from PIL import Image
from google.genai import types


class Gemini(BaseGenerator):

    def __init__(self, args):
        super().__init__(args)

        if args.image is not None:
            for img in args.image:
                with open(img, 'rb') as f:
                    image_part = types.Part.from_bytes(data=f.read(), mime_type=mimetypes.guess_type(img)[0])
                    self.images.append(image_part)

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
            choices=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],
            help="Aspect ratio to request (default: %(default)s)",
        )
        parser.add_argument(
            "-b",
            "--bypass",
            action="store_true",
            help="Attempt to bypass the safeties on img2img (default: %(default)s)",
        )


    @staticmethod
    def _extract_image_bytes(response: types.GenerateContentResponse) -> bytes:
        for part in response.parts:
            if part.inline_data is not None:
                return part.inline_data.data

        raise RuntimeError("No image data returned from the API.")

    def generate(self, prompt: str) -> list[tuple[bytes, str]]:

        if self.args.bypass:
            contents = [
                types.Content(
                    role="model",
                    parts=self.images,
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                ),
            ]
        else:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)] + self.images,
                ),
            ]

        response = self.client.models.generate_content(
            model='gemini-2.5-flash-image',
            contents= contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=self.args.aspect_ratio),
            ),
        )
        image_bytes = self._extract_image_bytes(response)
        return [(image_bytes, 'png')]
