"""Helper script to create a dummy submission JSON file.

The evaluation server expects a JSON file with the keys ``videos`` and
``predictions``.  ``videos`` should contain the list of test video IDs in the
exact order provided by the challenge organisers, while ``predictions`` should
contain an integer class label for each video.  This script creates a template
submission with all predictions set to ``0``.  It accepts either a JSON file
containing the list of videos directly or a dictionary with a ``videos`` key.
"""

import argparse
import json
from typing import List


def load_video_list(path: str) -> List[str]:
    """Return the list of video identifiers from ``path``.

    The input JSON file may either be a plain list of video identifiers or a
    dictionary containing the key ``"videos"``.  This helper normalises both
    formats so the script can be used with whichever file is available.
    """

    with open(path, "r") as fp:
        data = json.load(fp)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "videos" in data:
        return data["videos"]

    raise ValueError(f"Unrecognised videos file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dummy submission file for the proficiency benchmark"
    )
    parser.add_argument(
        "--videos-file",
        type=str,
        required=True,
        help="Path to the challenge video list JSON",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="test_submission.json",
        help="Destination file for the generated submission",
    )

    args = parser.parse_args()

    videos = load_video_list(args.videos_file)
    predictions = [0] * len(videos)

    with open(args.output_file, "w") as fp:
        json.dump({"videos": videos, "predictions": predictions}, fp)

    print(f"Wrote {args.output_file} with {len(videos)} predictions")


if __name__ == "__main__":
    main()

