import logging
import time
from collections import defaultdict

from grounded_ccg_inducer import constants
from grounded_ccg_inducer.dataloader import Dataloader
from grounded_ccg_inducer.pos_analyzer import POSAnalyzer
from scripts import script_utils

DEBUG = False


def main():
    t0 = time.time()
    logging.info("Loading DiDeMo...")
    dataloader = Dataloader(constants.didemo_path)

    pos_analyzer = POSAnalyzer()
    script_utils.pos_tag_dataset(dataloader, pos_analyzer)

    count_nouns_verbs(dataloader.tagged_dataset)
    calc_frequency_distribution(dataloader.tagged_dataset)
    print_most_frequent(dataloader.tagged_dataset)
    analyze_dataset(dataloader.tagged_dataset)

    t1 = time.time()
    logging.info(f"Program took {t1-t0}")


def calc_frequency_distribution(sentences):
    len_counter = defaultdict(list)
    for i, sent in enumerate(sentences):
        sent_len = len(sent)
        len_counter[sent_len].append(i)

    max_key = max([k for k in len_counter])

    logging.info("Sentence length counter")
    for i in range(0, max_key + 1):
        count = len(len_counter[i])
        if count != 0:
            print(f"{i}\t{len(len_counter[i])}")


def count_nouns_verbs(sentences):
    noun_counter = defaultdict(int)
    verb_counter = defaultdict(int)
    for i, sent in enumerate(sentences):
        for word, tag in sent:
            if tag == "NOUN":
                noun_counter[word] += 1
            if tag == "VERB":
                verb_counter[word] += 1

    sorted_noun_counter = sorted(noun_counter.items(), key=lambda x: x[1], reverse=True)
    logging.info(f"Nr of unique nouns: {len(noun_counter)}")
    logging.info(f"Nr of nouns: {sum(noun_counter.values())}")
    logging.info("Top 10 nouns:")
    for word, count in sorted_noun_counter[:10]:
        print(word, count)

    sorted_verb_counter = sorted(verb_counter.items(), key=lambda x: x[1], reverse=True)
    logging.info(f"Nr of unique verbs: {len(verb_counter)}")
    logging.info(f"Nr of verbs: {sum(verb_counter.values())}")
    logging.info("Top 10 verbs:")
    for word, count in sorted_verb_counter[:10]:
        print(word, count)


def analyze_dataset(sentences):
    open_class_tags = {"ADJ", "ADV", "INTJ", "PROPN"}
    closed_class_tags = {
        "ADP",
        "AUX",
        "CCONJ",
        "DET",
        "NUM",
        "PART",
        "PRT",
        "PRON",
        "SCONJ",
        "CONJ",
    }
    custom_tags = {"object", "action", "OBJECT", "ACTION"}
    tagsets = [custom_tags, closed_class_tags, open_class_tags]

    known_tagset = set()
    for tagset in tagsets:
        known_tagset = known_tagset.union(tagset)
        logging.info(f"tagset: {known_tagset}")
        count_tags(sentences, known_tagset)


def count_tags(sentences, tagset):
    unk_per_sent = defaultdict(int)

    for i, sent in enumerate(sentences):
        for _, tag in sent:
            if tag not in tagset:
                unk_per_sent[i] += 1

    unk_counter = defaultdict(int)
    for _, count in unk_per_sent.items():
        unk_counter[count] += 1

    max_key = max([k for k in unk_counter])

    n_or_less = defaultdict(int)
    n_or_more = defaultdict(int)
    for i in range(1, max_key):
        for j in range(i, max_key):
            n_or_less[j] += unk_counter[i]
        for j in range(i, max_key):
            n_or_more[i] += unk_counter[j]

    print("UNK\tN\t<=N\t>=N")
    for i in range(1, 6):
        print(f"{i}\t{unk_counter[i]}\t{n_or_less[i]}\t{n_or_more[i]}")


def print_most_frequent(sentences):
    sentence_count = len(sentences)
    word_count = 0
    noun_count = 0
    verb_count = 0
    len_counter = defaultdict(list)
    word_dict = defaultdict(int)
    noun_dict = defaultdict(int)
    verb_dict = defaultdict(int)

    logging.info("Looping through sentences...")
    for sentence in sentences:
        sent_len = len(sentence)
        word_count += sent_len
        len_counter[sent_len].append(sentence)
        for word, tag in sentence:
            word_dict[word] += 1
            if tag == "NOUN":
                noun_dict[word] += 1
                noun_count += 1
            if tag == "VERB":
                verb_dict[word] += 1
                verb_count += 1

    logging.info("Total sentences: %s", sentence_count)
    logging.info("Total words: %s", word_count)
    logging.info("Total nouns: %s", noun_count)
    logging.info("Total verbs: %s", verb_count)

    logging.info("WORDS")
    print_top_n_words(word_dict)

    logging.info("NOUNS")
    print_top_n_words(noun_dict)

    logging.info("VERBS")
    print_top_n_words(verb_dict)


def print_top_n_words(word_dict, topn=10):
    sorted_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    total_count = 0
    for word, count in sorted_words[:topn]:
        print(f"{word}\t{count}")
        total_count += count
    other_count = 0

    for word, count in sorted_words[topn:]:
        other_count += count
        total_count += count
    print()
    print(f"other_words\t{other_count}")
    print(f"total_words\t{total_count}")


if __name__ == "__main__":
    script_utils.init_log("analyze_dataset", debug=DEBUG)
    main()
