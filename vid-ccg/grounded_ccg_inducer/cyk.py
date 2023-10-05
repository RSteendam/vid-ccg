import itertools
import logging

from pandas import DataFrame


class Node:
    def __init__(self, symbol, child1, direction=None, child2=None):
        self.symbol = symbol
        self.child1 = child1
        self.direction = direction
        self.child2 = child2

    def __key(self):
        return (self.symbol, self.child1, self.direction, self.child2)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return self.symbol


class CYK:
    def __init__(self, grammar=None, debug=False) -> None:
        self.debug = debug
        self.grammar = []
        self.chart = None
        if isinstance(grammar, list):
            self.grammar_from_list(grammar)
        else:
            self.grammar_from_file(grammar)
        logging.debug(f"rules in grammar: {self.grammar})")
        logging.info(f"nr of rules in grammar: {len(self.grammar)}")
        self.write_grammar_to_file()
        self.counter = 0

    def write_grammar_to_file(self):
        _grammar_file = "output/grammar"
        with open(_grammar_file, "w") as f:
            for item in self.grammar:
                args = " ".join(item[1:])
                f.write(f"{item[0]}\t->\t{args}\n")

    def grammar_from_file(self, grammar):
        with open(grammar, "r") as f:
            raw_grammar = f.read().splitlines()
        self.grammar_from_list(raw_grammar)

    def grammar_from_list(self, grammar):
        for line in grammar:
            line = line.replace(",", "")
            line = line.split()
            self.grammar.append(line)
        self.grammar.sort()
        self.grammar = list(k for k, _ in itertools.groupby(self.grammar))

    def parse_sentence(self, words):
        if self.grammar is None:
            raise ValueError("Please enter either grammar or grammar_file.")
        counter = 0

        sentence_length = len(words)
        self.chart = [
            [set() for x in range(sentence_length)] for y in range(sentence_length)
        ]
        for i in range(0, sentence_length):
            word = words[i]
            for rule in self.grammar:
                if word == rule[1]:
                    self.chart[i][i].add(Node(rule[0], word))
            if self.debug:
                logging.debug(f"\n{DataFrame(self.chart)}")
        for j in range(1, sentence_length):
            for i in reversed(range(0, j)):
                logging.debug("cyk.parse_sentence: step 2.1")
                for k in range(i + 1, j + 1):
                    left_chart = self.chart[i][k - 1]
                    right_chart = self.chart[k][j]
                    for left in left_chart:
                        for right in right_chart:
                            new_rule = valid_rule(left, right)
                            if new_rule:
                                logging.info("new rule: %s", new_rule)
                                counter += 1
                                self.counter += 1
                                self.chart[i][j].add(new_rule)
                logging.debug("cyk.parse_sentence: step 2.2")
        if self.debug:
            logging.debug(f"\n{DataFrame(self.chart)}")
        final_nodes = [x for x in self.chart[0][sentence_length - 1] if x.symbol == "S"]
        logging.debug(f"final_nodes: {len(final_nodes)}")
        trees = []
        global output
        output = set()
        if final_nodes:
            for node in final_nodes:
                recursive(node)

                if self.debug:
                    tree = self.generate_tree(node)
                    trees.append(tree)
                    logging.debug(tree)
                    logging.debug(f"after final_nodes trees len: {len(trees)}")
                    logging.debug(f"output len: {len(output)}")
                    logging.debug(f"correct_rules: {output}")
        return output, counter

    def generate_tree(self, node):
        if node.child2 is None:
            return node.child1 + "_" + node.symbol
        return (
            "["
            + self.generate_tree(node.child1)
            + " "
            + self.generate_tree(node.child2)
            + "]_"
            + node.symbol
        )


def valid_rule(left, right):
    logging.info(f"valid_rule: {left}, {right}")
    rule = fw_apply(left, right)  # L/R   R     --> P
    if rule:
        return rule
    rule = bw_apply(left, right)  # L     L\R   --> P
    if rule:
        return rule
    rule = fw_compose(left, right)  # X/Y   Y/Z   --> X/Z
    if rule:
        return rule
    rule = fw_xcompose(left, right)  # X/Y   Y\Z   --> X\Z
    if rule:
        return rule
    rule = bw_compose(left, right)  # Y\Z   X\Y   --> X\Z
    if rule:
        return rule
    rule = bw_xcompose(left, right)  # Y/Z   X\Y   --> X/Z
    if rule:
        return rule
    return False


def fw_apply(left, right):
    if "/" in left.symbol:
        logging.info("/ in left")
        if left.child2 == right.symbol:
            logging.info("FW APPLY")
            return left.symbol


def bw_apply(left, right):
    if "\\" in right.symbol:
        if right.child1 == left.symbol:
            logging.info("BW APPLY")
            return right.symbol
    return False


def fw_compose(left, right):
    if "/" in left.symbol:
        if "/" in right.symbol:
            if left.child2 == right.child1:
                logging.info("FW COMPOSE")
                return Node(left.child1 + "/" + right.child2, left, right)
    return False


def fw_xcompose(left, right):
    if "/" in left.symbol:
        if "\\" in right.symbol:
            if left.child2 == right.child1:
                logging.info("FW XCOMPOSE")
                return Node(left.child1 + "\\" + right.child2, left, right)
    return False


def bw_compose(left, right):
    if "\\" in left.symbol:
        if "\\" in right.symbol:
            if left.child1 == right.child2:
                logging.info("BW COMPOSE")
                return Node(left.child2 + "\\" + right.child1, left, right)
    return False


def bw_xcompose(left, right):
    if "/" in left.symbol:
        if "\\" in right.symbol:
            if left.child1 == right.child2:
                logging.info("BW XCOMPOSE")
                return Node(left.child2 + "/" + right.child1, left, right)
    return False


output = set()


def recursive(node):
    if node.child2 is None:
        rule = (node.symbol, node.child1)
        if rule not in output:
            output.add(rule)
        return

    rule = (node.symbol, node.child1, node.child2)
    if rule not in output:
        output.add(rule)

    recursive(node.child1)
    recursive(node.child2)
