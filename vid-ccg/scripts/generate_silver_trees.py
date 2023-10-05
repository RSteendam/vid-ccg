import os
import subprocess
import sys

from . import script_utils

DEBUG = False


def main():
    tagged_dataset_path = sys.argv[1]
    filename = os.path.split(tagged_dataset_path)[1]

    auto_output_path = os.path.join("output", "depccg", "auto_" + filename)
    conll_output_path = os.path.join("output", "depccg", "connl_" + filename)
    generate_auto_command = f"depccg_en -f auto -i {tagged_dataset_path}"
    generate_conll_command = f"depccg_en -f conll -i {tagged_dataset_path}"

    with open(auto_output_path, "w") as outfile:
        subprocess.call(generate_auto_command.split(), stdout=outfile)

    clean_output_file(auto_output_path)

    with open(conll_output_path, "w") as outfile:
        subprocess.call(generate_conll_command.split(), stdout=outfile)

    clean_conll(conll_output_path)


def clean_conll(conll_file):
    clean_output_file(conll_file)
    if True:
        with open(conll_file) as f_input:
            data = f_input.readlines()
        new_data = []
        for line in data:
            if line.startswith("# ID="):
                new_data.append("\n")
                continue
            if line.startswith("# log"):
                continue
            id, word, lemma, upos, xpos, _unk1, head, cat, _unk2, subtree = line.split(
                "\t"
            )

            new_line = "\t".join(
                [id, word, lemma, upos, xpos, _unk1, _unk2, head, cat, subtree]
            )
            new_data.append(new_line)

        del new_data[0]
        new_data.append("\n")
        new_data.append("\n")
        new_data = "".join(new_data)

        with open(conll_file, "w") as f_output:
            f_output.write(new_data)


def clean_output_file(output_path):
    with open(output_path) as f_input:
        data = f_input.read().rstrip("\n")

    with open(output_path, "w") as f_output:
        f_output.write(data)


if __name__ == "__main__":
    script_utils.init_log("generate_silver_trees", debug=DEBUG)
    main()
