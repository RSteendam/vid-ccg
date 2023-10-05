import os

import tqdm

SENTENCE_LIMIT = 20
PTB_TAGGED_TRAIN_LABELED = "output/wsj/wsj_out_train_labeled.txt"
PTB_TAGGED_TRAIN_LABELED_CLEAN = (
    f"output/wsj/wsj_out_train_labeled_clean_{SENTENCE_LIMIT}.txt"
)
PTB_TAGGED_TRAIN_SENTENCES = "output/wsj/wsj_train_sentences_labeled.txt"
PTB_TAGGED_TRAIN_SENTENCES_CLEAN = (
    f"output/wsj/wsj_train_sentences_labeled_clean_{SENTENCE_LIMIT}.txt"
)


def main():
    with open(PTB_TAGGED_TRAIN_LABELED) as f:
        ptb_tagged_train = [line for line in f.readlines()]

    with open(PTB_TAGGED_TRAIN_SENTENCES) as f:
        ptb_sentences_train = [line for line in f.readlines()]

    long_sentences = get_long_sentences(ptb_tagged_train, sentence_limit=SENTENCE_LIMIT)
    remove_long_sentences(
        long_sentences, ptb_tagged_train, PTB_TAGGED_TRAIN_LABELED_CLEAN
    )
    remove_long_sentences(
        long_sentences, ptb_sentences_train, PTB_TAGGED_TRAIN_SENTENCES_CLEAN
    )


def get_long_sentences(ptb_tagged, sentence_limit=None):
    long_sentence_ids = []
    id = None
    sentence_length = 0
    for line in ptb_tagged:
        if line.startswith("wsj"):
            if sentence_length > sentence_limit:
                long_sentence_ids.append(id)
            id = line.strip()
            sentence_length = 0
        else:
            tag = line.strip().split("\t")[-1]

            if tag != "." and tag != "":
                sentence_length += 1
    if sentence_length > sentence_limit:
        long_sentence_ids.append(id)

    return long_sentence_ids


def remove_long_sentences(long_sentence_ids, ptb_tagged, output_file):
    labeled_train_file = open(output_file, "w")
    write_line = False
    for line in tqdm.tqdm(ptb_tagged):
        if line.startswith("wsj"):
            id = line.strip()
            write_line = id not in long_sentence_ids
        else:
            labeled_train_file.write(line)

    labeled_train_file.close()


def get_ptb_dict(ptb):
    ptb_dict = {}
    for i, line in enumerate(ptb):
        if i % 2 == 0:
            id = line.strip()
        else:
            ptb_dict[id] = line
    return ptb_dict


if __name__ == "__main__":
    main()
