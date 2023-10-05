import logging
from copy import deepcopy

import tqdm

from grounded_ccg_inducer.inducer_utils import (
    init_lexicon,
    left_arg,
    left_mod,
    ok_argument,
    right_arg,
    right_mod,
)


class Inducer:
    def __init__(self, dataset, max_sentence_length=12, debug=False) -> None:
        self.dataset = dataset
        self.lexicon = init_lexicon()
        self.new_lexicon = None
        self.max_sentence_length = max_sentence_length
        self.debug = debug
        self.total_len = len(self.dataset)
        logging.info("total_len: %s", self.total_len)

    def induction_step(self):
        self.new_lexicon = deepcopy(self.lexicon)
        logging.debug("induction_step")
        new_lexicons = []
        dataset = self.dataset

        for sentence in tqdm.tqdm(dataset):
            stripped_s = [(word, tag) for (word, tag) in sentence if tag != "."]

            if len(stripped_s) < self.max_sentence_length:
                new_lexicon = self._induce_sentence(
                    stripped_s, self.lexicon, self.new_lexicon
                )

            if new_lexicon != set():
                logging.debug(f"induction_step.new_lexicon: {new_lexicon}")
                logging.debug("NEW LEXICON OUT INDUCE CLIP")
                logging.debug(new_lexicon)
                new_lexicons.append(new_lexicon)

        result = self.new_lexicon
        logging.debug("resulting lexicons")
        logging.debug(f"induction_step result: {result}")
        logging.debug(f"induction_step new_lexicons: {new_lexicons}")

        for d in new_lexicons:
            for key in d:
                result[key].update(d[key])
        self.new_lexicon = result

        self.lexicon = deepcopy(self.new_lexicon)

    def write_lexicon(self, output_path):
        logging.debug(f"write_lexicon: {self.lexicon}")
        new_lexicon = {}
        for key, val in self.lexicon.items():
            if val != set():
                for v in val:
                    new_lexicon[v] = set()

        for key, val in self.lexicon.items():
            if val != set():
                for v in val:
                    new_lexicon[v].add(key)

        with open(output_path, "w") as f:
            for key, val in new_lexicon.items():
                val = sorted(val)
                print_val = " ".join(val)
                f.write(f"{key}\t{print_val}\n")

    def _inside_outside(self):
        pass

    def _induce_sentence(self, sentence, lexicon, new_lexicon):
        for i in range(len(sentence)):
            word, tag = sentence[i]
            category = self.lexicon[tag]

            if i + 1 < len(sentence):
                next_word, next_tag = sentence[i + 1]
                next_category = self.lexicon[next_tag]

                if next_category != {}:
                    logging.debug("Inducing right with %s and %s", word, next_word)
                    new_lexicon = self._induce_right(
                        tag, next_category, new_lexicon, lexicon
                    )

                if category != {}:
                    logging.debug("Inducing left with %s and %s", word, next_word)
                    logging.debug("Inducing left with %s and %s", tag, next_tag)
                    new_lexicon = self._induce_left(
                        category, next_tag, new_lexicon, lexicon
                    )

        return new_lexicon

    def _induce_left(self, left_category, right_pos, new_lexicon, lexicon):
        logging.debug("induce_left %s %s", left_category, right_pos)
        new_rules = set()
        if lexicon[right_pos] == set():
            for c_l in left_category:
                rule = left_mod(c_l)
                if rule is not None:
                    new_rules.add(rule)
            new_lexicon[right_pos].update(new_rules)
        else:
            for c_r in lexicon[right_pos]:
                for c_l in left_category:
                    # Right_Arg
                    if ok_argument(c_r, c_l):
                        rule = left_arg(c_l, c_r)
                        logging.debug("LEFT ARG RULE: %s", rule)
                        if rule is not None:
                            new_rules.add(rule)
                    # Right Mod
                    rule = left_mod(c_l)
                    logging.debug("LEFT MOD POS: %s", right_pos)
                    logging.debug("LEFT MOD RULE: %s", rule)
                    if rule is not None:
                        new_rules.add(rule)
            logging.debug("LEFT OLD RULES: %s %s", right_pos, new_lexicon[right_pos])
            new_lexicon[right_pos].update(new_rules)
            logging.debug("LEFT NEW RULES: %s %s", right_pos, new_lexicon[right_pos])

        return new_lexicon

    def _induce_right(self, left_pos, right_category, new_lexicon, lexicon):
        new_rules = set()
        if lexicon[left_pos] == set():
            for c_r in right_category:
                logging.debug("induce_right %s %s", left_pos, c_r)
                rule = right_mod(c_r)
                logging.debug("empty left rule: %s", rule)
                if rule is not None:
                    new_rules.add(rule)
            new_lexicon[left_pos].update(new_rules)
        else:
            for c_l in lexicon[left_pos]:
                for c_r in right_category:
                    # Right_Arg
                    if ok_argument(c_l, c_r):
                        rule = right_arg(c_l, c_r)
                        if rule is not None:
                            new_rules.add(rule)
                    # Right Mod
                    logging.debug("induce_right %s %s", left_pos, c_r)
                    rule = right_mod(c_r)
                    logging.debug("non empty left rule: %s", rule)
                    if rule is not None:
                        new_rules.add(rule)
            logging.debug("RIGHT OLD RULES: %s %s", left_pos, new_lexicon[left_pos])
            new_lexicon[left_pos].update(new_rules)
            logging.debug("RIGHT NEW RULES: %s %s", left_pos, new_lexicon[left_pos])

        return new_lexicon
