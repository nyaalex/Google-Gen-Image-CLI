import mimetypes
import re
import time

from PIL import Image
from google.genai import types

from google_gen.generators import BaseGenerator


class Veo(BaseGenerator):
    def __init__(self, client, args):
        super().__init__(client, args)

        if args.asset is not None:
            for img in args.image:
                self.images.append(Image.open(img))

        if args.last is not None:
            self.images.append(Image.open(args.last))

        if args.source is not None:
            self.images.append(Image.open(args.source))

    @staticmethod
    def setup_args(parser):
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

    @staticmethod
    def _open_image(image_path):
        img_mime, _ = mimetypes.guess_type(image_path)
        file = open(image_path, "rb")
        img_bytes = file.read()
        file.close()
        return types.Image(image_bytes=img_bytes, mime_type=img_mime)

    def generate(self, prompt: str) -> list[tuple[bytes, str]]:
        videos_source = types.GenerateVideosSource(prompt=prompt)
        config = types.GenerateVideosConfig(aspect_ratio=self.args.aspect_ratio)

        if self.args.source:
            videos_source.image = self._open_image(self.args.source)

        if self.args.last:
            config.last_frame = self._open_image(self.args.last)

        if self.args.asset:
            if len(self.args.asset) > 3:
                raise Exception("There can not be more than three image assets")

            config.reference_images = []
            for asset in self.args.asset:
                config.reference_images.append(types.VideoGenerationReferenceImage(
                    image=self._open_image(asset),
                    reference_type=types.VideoGenerationReferenceType.ASSET,
                ))

        try:
            operation = self.client.models.generate_videos(
                model='veo-3.1-generate-preview',
                source=videos_source,
                config=config,
            )

        except Exception as exc:
            print(exc)
            return []

        print("Waiting for video generation to complete...")
        while not operation.done:
            time.sleep(3)
            operation = self.client.operations.get(operation)

        if not operation.response.generated_videos:
            if operation.response.rai_media_filtered_reasons:
                print('\n'.join(operation.response.rai_media_filtered_reasons))
            return []

        generated_video = operation.response.generated_videos[0]

        video_bytes = self.client.files.download(file=generated_video.video)

        file_name = re.findall('files/[^:]*', generated_video.video.uri)
        self.client.files.delete(name=file_name[0])
        return [(video_bytes, 'mp4')]
