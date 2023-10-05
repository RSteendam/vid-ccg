import os

LABELED_PTB_SENTENCES = "output/wsj/wsj_sentences_labeled.txt"
CCG_AUTO = "output/wsj/ccg_auto_gold.auto"
PTB_TAGGED_LABELED = "output/wsj/wsj_out_labeled.txt"


def main():
    with open(LABELED_PTB_SENTENCES) as f:
        labeled_ptb = [line for line in f.readlines()]

    with open(PTB_TAGGED_LABELED) as f:
        ptb_tagged = [line for line in f.readlines()]

    with open(CCG_AUTO) as f:
        ccg = [line for line in f.readlines()]

    split_train_test(labeled_ptb, ptb_tagged, ccg)
    os.unlink(CCG_AUTO)
    os.unlink(PTB_TAGGED_LABELED)
    os.unlink(LABELED_PTB_SENTENCES)


def split_train_test(labeled_ptb, ptb_tags, ccg):
    split_train_test_labeled_ptb(labeled_ptb)
    split_train_test_tagged_ptb(ptb_tags)
    split_train_test_ccg(ccg)


def split_train_test_labeled_ptb(ptb):
    train_range = range(2, 22)
    test_range = range(23, 24)

    labeled_train_filename = "output/wsj/wsj_train_sentences.txt"
    labeled_test_filename = "output/wsj/wsj_test_sentences.txt"
    labeled_train_file = open(labeled_train_filename, "w")
    labeled_test_file = open(labeled_test_filename, "w")

    for i, line in enumerate(ptb):
        if i % 2 == 0:
            id_line = line
            id = line.strip()
            section = int(id[4:6])
        else:
            if section in train_range:
                labeled_train_file.write(line)

            if section in test_range:
                labeled_test_file.write(line)

    labeled_train_file.close()
    labeled_test_file.close()


def split_train_test_tagged_ptb(ptb):
    train_range = range(2, 22)
    test_range = range(23, 24)

    tagged_train_filename = "output/wsj/wsj_out_train.txt"
    tagged_test_filename = "output/wsj/wsj_out_test.txt"
    tagged_test_filename_no_punct = "output/wsj/wsj_out_test_np.txt"

    tagged_train_file = open(tagged_train_filename, "w")
    tagged_test_file = open(tagged_test_filename, "w")
    tagged_test_file_no_punct = open(tagged_test_filename_no_punct, "w")
    for line in ptb:
        if line.startswith("wsj"):
            id_line = line
            id = line.strip()
            section = int(id[4:6])
            if section in test_range:
                index = 1
        else:
            if section in train_range:
                tagged_train_file.write(line)

            if section in test_range:
                tagged_test_file.write(line)
                new_line = line
                if "\t" in line:
                    _, word, _, _, _, tag = line.strip().split("\t")
                    if tag != ".":
                        new_line = f"{index}\t{word}\t_\t_\t_\t{tag}\n"
                        index += 1
                    else:
                        continue
                tagged_test_file_no_punct.write(new_line)

    tagged_train_file.close()
    tagged_test_file.close()
    tagged_test_file_no_punct.close()


def split_train_test_ccg(ccg):
    test_range = range(23, 24)

    ccg_auto_test_filename = "output/wsj/ccg_auto_test.auto"
    ccg_auto_test_file = open(ccg_auto_test_filename, "w")

    for i, line in enumerate(ccg):
        if i % 2 == 0:
            id_line = line
            section = int(line.strip()[7:9])
        else:
            if section in test_range:
                ccg_auto_test_file.write(id_line)
                ccg_auto_test_file.write(line)

    ccg_auto_test_file.close()


if __name__ == "__main__":
    main()
