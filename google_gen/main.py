import json
import os
import sys
import argparse
from multiprocessing import Pool
from pathlib import Path
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
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="Use the system prompt for enhancing, gets around some safety settings (default: %(default)s)",
    )
    parser.add_argument(
        "--ignore-images",
        action="store_true",
        help="Stops images from being sent to the prompt enhancer (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for Google GenAI (overrides GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        help="Custom HTTP header in format KEY:VALUE (can be specified multiple times). Headers can also be set via GEMINI_HEADERS environment variable as JSON object.",
    )
    parser.add_argument("prompt", help="The text prompt for the content.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate content using the Google generative AI suite of models."
    )
    subparsers = parser.add_subparsers(dest="submodule", required=True)

    for submodule, name, help_str in SUBMODULES:
        subparser = subparsers.add_parser(name, help=help_str)
        add_shared_args(subparser)
        submodule.setup_args(subparser)

    args = parser.parse_args()

    # Check for API key - use argument if provided, otherwise check environment
    api_key = args.api_key or os.getenv('GEMINI_API_KEY', None)
    if api_key is None:
        print("GEMINI_API_KEY environment variable not set and --api-key not provided.")
        return

    selected = None
    for submodule, name, _ in SUBMODULES:
        if args.submodule == name:
            selected = submodule(args)

    if selected is None:
        raise Exception("Unknown submodule %s" % args.submodule)

    threads = os.getenv('THREAD_COUNT', 5)
    path = Path(args.output).resolve().parent
    hist_file = path / ".prompt_history"

    with Pool(threads) as p:
        with open(hist_file, 'a') as hist:
            for results in p.imap_unordered(selected.run, range(args.number)):
                for filename, prompt in results:
                    if filename is not None:
                        hist.write(json.dumps({
                            'filename': str(filename),
                            'prompt': prompt,
                            'args': sys.argv}) + '\n')


if __name__ == "__main__":
    main()
