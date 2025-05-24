import json
import argparse

parser = argparse.ArgumentParser(description="Generate a dummy submission file")
parser.add_argument('--videos-file', type=str, required=True,
                    help='Path to demonstrator_proficiency_test.json')
parser.add_argument('--output-file', type=str, default='test_submission.json',
                    help='Output JSON filename')
args = parser.parse_args()

with open(args.videos_file, 'r') as f:
    videos = json.load(f)

predictions = [0] * len(videos)

with open(args.output_file, 'w') as f:
    json.dump({"videos": videos, "predictions": predictions}, f)

print(f"Wrote {args.output_file} with {len(videos)} predictions.")
