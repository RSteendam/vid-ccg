import os
import re
import sys
import time

import tqdm
from nltk.corpus import ptb
from nltk.tag.mapping import map_tag

from scripts.wsj.forbidden_ids import FORBIDDEN_IDS

FORBIDDEN_TAGS = ["-NONE-"]
FORBIDDEN_WORDS = ["`", "''", "``"]
LABELED_PTB_SENTENCES = "output/wsj/wsj_sentences_labeled.txt"
PTB_TAGGED_LABELED = "output/wsj/wsj_out_labeled.txt"
LABELED_CCG_SENTENCES = "output/wsj/ccg_sentences_labeled.txt"
CCG_AUTO = "output/wsj/ccg_auto_gold.auto"


def main(ccg_auto_path):
    ccg_auto = get_ccg_auto(ccg_auto_path)

    t0 = time.time()
    ccg_sents = get_ccg_sents(ccg_auto)
    print(f"get_ccg_sents took: {time.time()-t0}s")

    t0 = time.time()
    ptb_sents = get_ptb_sents()
    print(f"get_ptb_sents took: {time.time()-t0}s")

    t0 = time.time()
    generate_files(ptb_sents, ccg_auto, ccg_sents)
    print(f"generate_files took: {time.time()-t0}s")

    t0 = time.time()
    check_files()
    print(f"check_files took: {time.time()-t0}s")


# Read CCG AUTO files and save them to a list with tuples of (id, line)
def get_ccg_auto(ccg_auto_path):
    def _get_file_list(ccg_auto_path):
        file_list = []
        for root, _, files in os.walk(ccg_auto_path):
            for file in files:
                if file.startswith("."):
                    continue
                file_list.append(os.path.join(root, file))
        file_list.sort()

        return file_list

    file_list = _get_file_list(ccg_auto_path)
    ccg_auto = []

    for f in tqdm.tqdm(file_list):
        with open(f, "r") as read_file:
            data = read_file.readlines()
            for i, elem in enumerate(data):
                if i % 2 == 0:
                    id = elem.split(" ")[0].split("ID=")[1]
                    continue
                ccg_auto.append((id, elem))

    return ccg_auto


# Gets sentences from ccg_auto list
def get_ccg_sents(ccg_auto):
    def _parse_line(line):
        node_description = re.findall(r"\<(.*?)\>", line)
        words = [n.split(" ")[4] for n in node_description if n.startswith("L")]
        sentence = " ".join(words)

        return sentence

    sentences = []

    for id, sent in ccg_auto:
        sentence = _parse_line(sent)
        sentences.append((id, sentence))

    return sentences


def get_ptb_sents():
    sentences = []
    for file_id in ptb.fileids():
        _, tail = os.path.split(file_id)
        base_id = tail.split(".MRG")[0].lower()

        sents = ptb.tagged_sents(file_id)

        for i, sent in enumerate(sents):
            id = base_id + "." + str(i + 1)

            new_tagged_sent = []
            for word, tag in sent:
                if tag not in FORBIDDEN_TAGS and word not in FORBIDDEN_WORDS:
                    new_tagged_sent.append((word, tag))

            sentences.append((id, new_tagged_sent))
    return sentences


def generate_files(
    ptb_sents,
    ccg_auto,
    ccg_sents,
):
    def _get_missing_ids(ccg_sents, ptb_sents):
        ccg_ids = set(id for id, _ in ccg_sents)
        ptb_ids = set(id for id, _ in ptb_sents)

        return set(ccg_ids).symmetric_difference(set(ptb_ids))

    missing_ids = _get_missing_ids(ccg_sents, ptb_sents)
    missing_ids.update(set(FORBIDDEN_IDS))
    t0 = time.time()
    generate_ptb_files(ptb_sents, missing_ids)
    print(f"generate_ptb_files took: {time.time()-t0}s")

    t0 = time.time()
    generate_ccg_files(ccg_auto, ccg_sents, missing_ids)
    print(f"generate_ccg_files took: {time.time()-t0}s")


def generate_ptb_files(ptb_sents, missing_ids):
    new_tagged_sentences = []
    new_sentences = []
    for i, (ptb_id, ptb_sent) in enumerate(ptb_sents):
        new_tagged_sentence = []
        if ptb_id in missing_ids:
            continue
        for word, tag in ptb_sent:
            new_tag = map_tag("en-ptb", "universal", tag)

            new_tagged_sentence.append((word, new_tag))

        sentence_string = " ".join([word for (word, _) in new_tagged_sentence])

        new_tagged_sentences.append((ptb_id, new_tagged_sentence))
        sentence_string = " ".join([word for (word, _) in new_tagged_sentence])
        new_sentences.append((ptb_id, sentence_string))

    with open(LABELED_PTB_SENTENCES, "w") as f:
        for id, sent in new_sentences:
            f.write(f"{id}\n")
            f.write(f"{sent}\n")

    with open(PTB_TAGGED_LABELED, "w") as f:
        for id, sentence in new_tagged_sentences:
            f.write(f"{id}\n")
            for j, (word, tag) in enumerate(sentence):
                f.write(f"{j+1}\t{word}\t_\t_\t_\t{tag}\n")
            f.write("\n")


def generate_ccg_files(ccg_autos, ccg_sents, missing_ids):
    new_sentences = []
    for ccg_id, ccg_sent in ccg_sents:
        if ccg_id in missing_ids:
            continue

        new_sentences.append((ccg_id, ccg_sent))

    with open(LABELED_CCG_SENTENCES, "w") as f:
        for ccg_id, sent in new_sentences:
            f.write(f"{ccg_id}\n")
            f.write(f"{sent}\n")

    new_autos = []
    for ccg_id, ccg_auto in ccg_autos:
        if ccg_id in missing_ids:
            continue
        new_autos.append((ccg_id, ccg_auto))

    with open(CCG_AUTO, "w") as f:
        for ccg_id, auto in new_autos:
            id_string = f"ID={ccg_id} PARSER=GOLD NUMPARSE=1\n"
            f.write(id_string)
            f.write(auto)


def check_files():
    with open(LABELED_CCG_SENTENCES) as f:
        ccg = f.readlines()

    with open(LABELED_PTB_SENTENCES) as f:
        ptb = f.readlines()

    found_mistakes = []
    sentence_count = 0
    for i, ccg_sent in enumerate(ccg):
        if i % 2 == 0:
            id = ccg_sent.strip()
        if i % 2 == 1:
            sentence_count += 1
            ptb_sent = ptb[i]
            if ccg_sent != ptb_sent:
                found_mistakes.append(id)
                print("not the same")
                print("PTB")
                print(ptb[i])

                print("CCG")
                print(ccg[i])

    if len(found_mistakes) == 0:
        print("Files are equal")
        os.unlink(LABELED_CCG_SENTENCES)
    else:
        print("Files are not equal")
        print("Check if there is an index error")
        print("Else update FORBIDDEN_IDS with list below")
        print(f"Found {len(found_mistakes)} mistakes out of {sentence_count} sentences")
        print(f"{found_mistakes}")


if __name__ == "__main__":
    ccg_auto_path = sys.argv[1]
    main(ccg_auto_path)
    exit()
