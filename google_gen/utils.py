from pathlib import Path
from itertools import count
from datetime import datetime
from secrets import token_hex


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
