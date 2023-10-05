import sys


def main(prediction_path, target_path):
    prediction = read_file(prediction_path)
    target = read_file(target_path)

    check_files(prediction, target)
    pos_accuracy(prediction, target)


def read_file(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()

    return data


def check_files(prediction, target):
    for i, val in enumerate(prediction):
        if val.split("\t")[0] != target[i].split("\t")[0]:
            raise ValueError(f"unequal at {i}")


def pos_accuracy(prediction, target):
    correct_tags = 0
    incorrect_tags = 0
    for i, val in enumerate(prediction):
        try:
            prediction_tag = val.split("\t")[5]
            target_tag = target[i].split("\t")[5]
            if prediction_tag == target_tag:
                correct_tags += 1
            else:
                incorrect_tags += 1
        except Exception:
            continue
    total_tags = correct_tags + incorrect_tags
    print(f"Encountered: {total_tags}")
    print(f"Correct: {correct_tags}")
    print(f"Incorrect: {incorrect_tags}")
    print(f"Accuracy: {correct_tags / total_tags}%")


if __name__ == "__main__":
    prediction_path = sys.argv[1]
    target_path = sys.argv[2]
    main(prediction_path, target_path)
