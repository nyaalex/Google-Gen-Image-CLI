import sys
from google_gen import gemini_generate, imagen_generate, veo_generate

USAGE = f"""Usage: {sys.argv[0]} {{imagen|gemini|veo}} [options]

Uses Google's generative AI API to generate different media types, including images and video.
"""

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        return

    submodule = sys.argv[1]
    if submodule == "gemini":
        gemini_generate.main(sys.argv[2:])
    elif submodule == "imagen":
        imagen_generate.main(sys.argv[2:])
    elif submodule == "veo":
        veo_generate.main(sys.argv[2:])
    else:
        print(USAGE)
        raise Exception(f"Unknown submodule: {submodule}")
