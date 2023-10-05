import logging

import tqdm
from grounded_ccg_inducer.cyk import CYK


class Parser:
    def __init__(
        self,
        dataset,
        lexicon,
        early_stopping=None,
        debug=False,
        max_sentence_length=None,
    ) -> None:
        self.dataset = dataset
        self.lexicon = lexicon
        self.debug = debug
        self.induced_grammar = self._lexicon_to_induced_grammar(lexicon)
        self.grammar = self._induced_grammar_to_grammar(self.induced_grammar, lexicon)
        self.cyk = CYK(grammar=self.grammar, debug=self.debug)
        self.early_stopping = early_stopping
        self.max_sentence_length = max_sentence_length
        self.count_parsed_sentences = 0
        self.cyk_counter = 0

    def parse_dataset(self):
        dataset = self.dataset
        rules_list = []
        if self.early_stopping is not None:
            dataset = dataset[: self.early_stopping]
        for i, sentence in enumerate(tqdm.tqdm(dataset)):
            rules, parsed_count, cyk_counter = self.parse_sentence(sentence)
            self.cyk_counter += cyk_counter
            self.count_parsed_sentences += parsed_count
            if rules != []:
                rules_list.append(rules)
        logging.info(f"Parsed {self.count_parsed_sentences} sentences")
        logging.info(f"Len: {len(rules_list)}")
        logging.debug(f"Rules List: {rules_list}")
        correct_rules = set()
        raw_count = 0
        for clip in rules_list:
            logging.debug(f"clip: {clip}")
            logging.debug(f"clip len: {len(clip)}")
            for rules in clip:
                logging.debug(f"rules: {rules}")
                logging.debug(f"rules len: {len(rules)}")
                raw_count += len(rules)
                for rule in rules:
                    logging.debug(f"rule: {rule}")
                    logging.debug(f"rule_type: {type(rule)}")
                    if len(rule) == 3:
                        x, y, z = rule
                        new_rule = (repr(x), repr(y), repr(z))
                    else:
                        x, y = rule
                        new_rule = (repr(x), repr(y))
                    logging.debug(f"x_type: {type(x.__repr__())}")
                    logging.debug(f"y_type: {type(y.__repr__())}")
                    if len(rule) == 3:
                        logging.debug(f"z_type: {type(z.__repr__())}")
                    correct_rules.add(new_rule)
        logging.info(f"raw_count: {raw_count}")
        logging.info(f"dedup_count: {len(correct_rules)}")
        logging.info(f"cyk counter: {self.cyk_counter}")
        return correct_rules

    def parse_clip(self, clip):
        rules_list = []
        nr_of_parsed_sentences = 0
        total_cyk_counter = 0
        for sentence in clip:
            rules, parsed_count, cyk_counter = self.parse_sentence(sentence)
            nr_of_parsed_sentences += parsed_count
            total_cyk_counter += cyk_counter
            if rules is not None and rules != set():
                rules_list.append(rules)
        return rules_list, nr_of_parsed_sentences, total_cyk_counter

    def parse_sentence(self, sentence):
        print(sentence)
        stripped_s = [(word, tag) for (word, tag) in sentence if tag != "."]
        print(stripped_s)
        nr_of_parsed_sentences = 0
        rules = None
        cyk_counter = 0
        if len(stripped_s) < self.max_sentence_length:
            nr_of_parsed_sentences += 1
            tags = [tag for _, tag in stripped_s]
            logging.debug(f"parsing {tags}")

            rules, cyk_counter = self.cyk.parse_sentence(tags)
        return rules, nr_of_parsed_sentences, cyk_counter

    def _lexicon_to_induced_grammar(self, lexicon):
        induced_grammar = set()
        for key, val in lexicon.items():
            if val != set():
                for category in val:
                    induced_grammar.add(f"{category.value}, {key}")
        return list(induced_grammar)

    def _induced_grammar_to_grammar(self, induced_grammar, lexicon):
        grammar = set(induced_grammar)
        for key, val in lexicon.items():
            if val != set():
                for category in val:
                    if category.direction == "left":
                        grammar.add(f"{category.base}, {category.arg} {category.value}")
                    elif category.direction == "right":
                        grammar.add(f"{category.base}, {category.value} {category.arg}")
        return list(grammar)
