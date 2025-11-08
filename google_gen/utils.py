from pathlib import Path
from itertools import count
from datetime import datetime
from secrets import token_hex

from google import genai
from google.genai import types


def get_name(output_path: Path, multiple=False):
    parent = output_path.parent or Path("..")
    stem = output_path.stem or "file"
    suffix = output_path.suffix or ".bin"
    parent.mkdir(parents=True, exist_ok=True)
    uid = datetime.today().strftime("%Y%m%d%H%M%S") + '-' + token_hex(2)

    if not multiple:
        return parent / f"{stem}-{uid}{suffix}"

    else:
        return (parent / f"{stem}-{uid}-{i + 1}{suffix}" for i in count())

def enhance_prompt(prompt, gen_type, images=None):
    client = genai.Client()
    if images is None:
        images = []

    enhancer_model = "gemini-2.5-flash"
    system_instruction = (
        f"You are an expert {gen_type} generation prompt writer. "
        f"Please take the following prompt and enhance it to be more descriptive and "
        f"suitable for a text-to-{gen_type} model. Make it vivid and detailed, but keep it concise. "
        "Only return the enhanced prompt, without any preamble or explanation."
    )


    response = client.models.generate_content(
        model=enhancer_model,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        ),
        contents=[prompt] + images,
    )
    new_prompt = response.text.strip()
    return new_prompt