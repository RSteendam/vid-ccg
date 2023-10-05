import sys

import pandas as pd


def main(python_lexicon_path, java_lexicon_path):
    python_lexicon = read_python_lexicon(python_lexicon_path)
    java_lexicon = read_java_lexicon(java_lexicon_path)

    dictionaries_are_the_same = compare_dictionaries(python_lexicon, java_lexicon)
    if dictionaries_are_the_same:
        print("hurray, dictionaries are the same")
    else:
        print("Too bad, dictionaries are not the same")


def compare_dictionaries(python_dict, java_dict):
    checked_keys = []
    dictionaries_are_the_same = True
    for key in python_dict:
        if key not in checked_keys:
            checked_keys.append(key)

            if key in java_dict:
                if python_dict[key] != java_dict[key]:
                    dictionaries_are_the_same = False
                    print(
                        f"1. {key} items in python_dict but not in java_dict: {python_dict[key].difference(java_dict[key])} and items in java_dict but not in python_dict: {java_dict[key].difference(python_dict[key])}"
                    )
            else:
                dictionaries_are_the_same = False
                print(f"2. {key} not in java_dict")

    for key in java_dict:
        if key not in checked_keys:
            checked_keys.append(key)
            if key in python_dict:
                if python_dict[key] != java_dict[key]:
                    dictionaries_are_the_same = False
                    print(
                        f"3. {key} items in python_dict but not in java_dict: {python_dict[key].difference(java_dict[key])} and items in java_dict but not in python_dict: {java_dict[key].difference(python_dict[key])}"
                    )
            else:
                dictionaries_are_the_same = False
                print(f"4. {key} not in python_dict for java_dict: {java_dict[key]}")

    return dictionaries_are_the_same


def read_python_lexicon(lexicon_path):
    with open(lexicon_path, "r") as f:
        data = f.read().strip().split("\n")

    data_dict = {}
    for line in data:
        key, val = line.split("\t")
        tags_set = set(val.split(" "))
        tags_set.discard("CONJ")
        data_dict[key] = tags_set

    return data_dict


def read_java_lexicon(lexicon_path):
    forbidden_categories = {
        "!",
        '"',
        "''",
        "(",
        ")",
        "-",
        "--",
        ".",
        ":",
        "<",
        ">",
        "?",
        "[",
        "]",
        "``",
        "conj",
        "{",
        "}",
        "«",
        "»",
        "“",
        "”",
    }

    fwidths = [20, 600]
    df = pd.read_fwf(lexicon_path, widths=fwidths, names=["category", "tags"])
    df.category.replace(r"\\\.", r"\\", regex=True, inplace=True)
    df.category.replace(r"/\.", "/", regex=True, inplace=True)

    data_dict = df.to_dict()
    new_dict = {}
    for key, val in data_dict["category"].items():
        tags_set = set(data_dict["tags"][key].split(" "))
        tags_set.discard("SCONJ")
        tags_set.discard("CCONJ")
        tags_set.discard("INTJ")
        tags_set.discard("CONJ")
        new_dict[val] = tags_set

    for item in forbidden_categories:
        if item in new_dict:
            new_dict.pop(item)

    return new_dict


if __name__ == "__main__":
    python_lexicon = sys.argv[1]
    java_lexicon = sys.argv[2]
    main(python_lexicon, java_lexicon)
