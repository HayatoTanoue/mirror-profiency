import json

def verify_json_files(submission_file_path, test_file_path):
    try:
        with open(submission_file_path, 'r') as f:
            submission_data = json.load(f)
        with open(test_file_path, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("Error: One or both JSON files not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from one or both files.")
        return

    submission_videos = submission_data.get("videos", [])
    submission_predictions = submission_data.get("predictions", [])
    
    # The test data is a list of videos directly
    test_videos = test_data

    # Check 1: The `videos` list in `my_submission.json` is identical to the list in `demonstrator_proficiency_test.json`.
    check1_passed = (submission_videos == test_videos)
    print(f"Check 1 (Videos lists identical): {'Passed' if check1_passed else 'Failed'}")

    # Check 2: The `predictions` list in `my_submission.json` has the same number of elements as the `videos` list.
    len_submission_videos = len(submission_videos)
    len_submission_predictions = len(submission_predictions)
    check2_passed = (len_submission_predictions == len_submission_videos)
    print(f"Check 2 (Predictions length matches videos length): {'Passed' if check2_passed else 'Failed'}")

    # Check 3: All elements in the `predictions` list are `0`.
    check3_passed = all(p == 0 for p in submission_predictions)
    print(f"Check 3 (All predictions are 0): {'Passed' if check3_passed else 'Failed'}")

    print(f"Length of videos list in submission: {len_submission_videos}")
    print(f"Length of predictions list in submission: {len_submission_predictions}")
    print(f"Length of videos list in test file: {len(test_videos)}")

    if check1_passed and check2_passed and check3_passed:
        print("All checks passed successfully.")
    else:
        print("Some checks failed.")

if __name__ == "__main__":
    submission_file = "demonstrator_proficiency/my_submission.json"
    test_file = "demonstrator_proficiency/demonstrator_proficiency_test.json"
    verify_json_files(submission_file, test_file)
