from google.genai import types

from google_gen.generators import BaseGenerator


class Imagen(BaseGenerator):

    @staticmethod
    def setup_args(parser):
        parser.add_argument(
            "-b",
            "--batch",
            type=int,
            default=4,
            help="Number of images to generate at once (default: %(default)s)",
        )
        parser.add_argument(
            "--aspect-ratio",
            default="1:1",
            choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
            help="Aspect ratio of the generated image(s) (default: %(default)s)",
        )

    def generate(self, prompt):
        config = types.GenerateImagesConfig(
            number_of_images=self.args.batch,
            aspect_ratio=self.args.aspect_ratio,
            include_rai_reason=True,
        )

        try:
            response = self.client.models.generate_images(
                model='imagen-4.0-generate-001',
                prompt=prompt,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - surfaced as CLI error text.
            print(f"API call failed: {exc}")
            return []

        generated_images = getattr(response, "generated_images", [])
        if not generated_images:
            print("No images were generated.")
            return []

        output = []
        for image in generated_images:
            image_bytes = image.image.image_bytes
            if not image_bytes:
                print(image.rai_filtered_reason)
                print(f"Warning: No image data for generated image.")
                continue

            output.append((image_bytes, 'png'))

        return output
