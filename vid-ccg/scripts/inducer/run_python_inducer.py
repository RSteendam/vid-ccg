import collections
import logging
import os
import pickle
import sys

import coloredlogs
from grounded_ccg_inducer.inducer import Inducer
from grounded_ccg_inducer.parser import Parser

DEBUG = False
ALLOWED_TRAINING_STEPS = {"read_training_files", "induction"}
MAX_SENTENCE_LENGTH = 15


def main(data_path):
    training_steps = "read_training_files,induction,induction,induction"
    training_steps = parse_training_steps(training_steps)

    inducer = None
    dataset = None

    for training_step in training_steps:
        if training_step == "read_training_files":
            logging.info("Reading training files...")
            dataset = load_data(data_path)

        elif training_step == "induction":
            logging.info("Induction step...")
            if inducer is None:
                inducer = Inducer(
                    dataset=dataset,
                    max_sentence_length=MAX_SENTENCE_LENGTH,
                    debug=DEBUG,
                )
            inducer.induction_step()

            logging.debug(f"run induce_grammar lexicon: {inducer.lexicon}")
            category_count = 0
            for key in inducer.lexicon:
                category_count += len(inducer.lexicon[key])
            logging.info(f"categories in lexicon: {category_count}")
            od = collections.OrderedDict(sorted(inducer.lexicon.items()))
            logging.debug("lexicon ORDERED DICT:")
            logging.debug(od)
            inducer.write_lexicon("output/python_lexicon")
        else:
            raise ValueError(
                f"{training_step} not an allowed training step. \
                             Choose one of {ALLOWED_TRAINING_STEPS}"
            )


def inside_outside(parsed_rules):
    get_counts(parsed_rules)


def get_counts(rules):
    count_dict = {}
    for rule in rules:
        if rule[0] not in count_dict:
            count_dict[rule[0]] = 0
        count_dict[rule[0]] += 1
    for rule in rules:
        probability = 1 / count_dict[rule[0]]
        logging.debug(rule, probability)


def parse_grammar(tagged_dataset, lexicon):
    filename = f"output/parsed_grammar--{MAX_SENTENCE_LENGTH}"
    parser = Parser(
        tagged_dataset,
        lexicon=lexicon,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        debug=DEBUG,
    )
    logging.debug("INDUCED GRAMMAR:")
    for rule in parser.induced_grammar:
        logging.debug(rule)
    logging.debug("GRAMMAR:")
    for rule in parser.grammar:
        logging.debug(rule)
    if os.path.exists(filename):
        with open(filename, "rb") as fp:
            parsed_rules = pickle.load(fp)
    else:
        parsed_rules = parser.parse_dataset()
        with open(filename, "wb") as fp:
            pickle.dump(parsed_rules, fp)
    return parsed_rules


def load_data(data_path):
    with open(data_path) as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    data = [line.split("\t") for line in data]

    sentences = []
    sentence = []
    for line in data:
        if len(line) == 1:
            sentences.append(sentence)
            sentence = []
            continue
        _, word, _, _, _, tag = line
        sentence.append((word, tag))
    if len(sentence) != 0:
        sentences.append(sentence)

    return sentences


def parse_training_steps(steps):
    training_steps = steps.replace(" ", "").split(",")
    training_steps = [x for x in training_steps if x != ""]

    for step in training_steps:
        if step not in ALLOWED_TRAINING_STEPS:
            raise ValueError(
                f"{step} not an allowed training step. Choose one of {ALLOWED_TRAINING_STEPS}"
            )

    return training_steps


def init(debug=False):
    level = "INFO"
    if debug:
        level = "DEBUG"
    coloredlogs.install(level=level, fmt="> %(asctime)s %(levelname)-8s %(message)s")


if __name__ == "__main__":
    init(debug=DEBUG)
    data_path = sys.argv[1]
    main(data_path)
