from pathlib import Path
from itertools import count
from datetime import datetime
from secrets import token_hex

from google import genai
from google.genai import types


class BaseGenerator:
    def __init__(self, args):
        self.client = None
        self.args = args
        self.images = []

    @staticmethod
    def setup_args(parser):
        pass

    def generate(self, prompt: str) -> list[tuple[bytes, str]]:
        pass

    @staticmethod
    def _get_name(output_path: Path, filetype):
        parent = output_path.parent or Path("..")
        stem = output_path.stem or "file"

        parent.mkdir(parents=True, exist_ok=True)
        uid = datetime.today().strftime("%Y%m%d%H%M%S") + '-' + token_hex(2)
        return parent / f"{stem}-{uid}.{filetype}"

    def _enhance_prompt(self, prompt):
        enhancer_model = "gemini-2.5-flash"
        system_instruction = (
            f"You are an expert content generation prompt writer. "
            f"Please take the following prompt and enhance it to be more descriptive and "
            f"suitable for a text-to-content model. Make it vivid and detailed, but keep it concise. "
            "Only return the enhanced prompt, without any preamble or explanation."
        )

        response = self.client.models.generate_content(
            model=enhancer_model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE)
                ]
            ),
            contents=[prompt] + self.images,
        )
        new_prompt = response.text.strip()
        return new_prompt

    def run(self, _):
        self.client = genai.Client()
        prompt = self.args.prompt
        for _ in range(self.args.retries):
            try:
                if self.args.enhance:
                    prompt = self._enhance_prompt(prompt)
                content = self.generate(prompt)
                break
            except Exception as exc:
                print("Failed to generate image:", exc)
        else:
            return []

        results = []
        for image, filetype in content:
            output_path = Path(self.args.output)
            unique_path = BaseGenerator._get_name(output_path, filetype)
            unique_path.write_bytes(image)
            print(f"Prompt for {unique_path}: {prompt}")
            print(f"Saved image to {unique_path.resolve()}")
            results.append((unique_path, prompt))
        return results
