import json
import os
import sys
import argparse
from pathlib import Path

from google import genai
from google_gen.generators import *


def add_shared_args(parser):
    parser.add_argument(
        "-o",
        "--output",
        default="generated",
        help="Prefix for the output file, will be appended with the date, random token (to avoid collisions), and file suffix (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--retries",
        default=3,
        type=int,
        help="Number of times to retry the prompt in case of errors (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1,
        help="Number of items to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--enhance",
        action="store_true",
        help="Enhance the prompt using gemini-2.5-flash (default: %(default)s)",
    )
    parser.add_argument("prompt", help="The text prompt for the content.")


def main():
    if os.getenv('GEMINI_API_KEY', None) is None:
        print("GEMINI_API_KEY environment variable not set.")
        return
    client = genai.Client()

    parser = argparse.ArgumentParser(
        description="Generate content using the Google generative AI suite of models."
    )
    subparsers = parser.add_subparsers(dest="submodule", required=True)

    for submodule, name, help_str in SUBMODULES:
        subparser = subparsers.add_parser(name, help=help_str)
        add_shared_args(subparser)
        submodule.setup_args(subparser)

    args = parser.parse_args()

    selected = None
    for submodule, name, _ in SUBMODULES:
        if args.submodule == name:
            selected = submodule(client, args)

    if selected is None:
        raise Exception("Unknown submodule %s" % args.submodule)

    threads = os.getenv('THREAD_COUNT', 5)
    path = Path(args.output).resolve().parent
    hist_file = path / ".prompt_history"

    with open(hist_file, 'a') as hist:
        for i in range(args.number):
            results = selected.run(0)
            for filename, prompt in results:
                if filename is not None:
                    hist.write(json.dumps({
                        'filename': str(filename),
                        'prompt': prompt,
                        'args': sys.argv}) + '\n')


if __name__ == "__main__":
    main()
