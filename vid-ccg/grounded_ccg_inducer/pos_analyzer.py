import spacy
from nltk.tag.mapping import map_tag


class POSAnalyzer:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentence(self, sentence: list[str]) -> list[tuple[str, str]]:
        return self.analyze_sentence_petrov(sentence)

    def analyze_sentence_upos(self, sentence: list[str]) -> list[tuple[str, str]]:
        doc = self.nlp(" ".join(sentence))
        return [(token.text, token.pos_) for token in doc]

    def analyze_sentence_petrov(self, sentence: list[str]) -> list[tuple[str, str]]:
        doc = self.nlp(" ".join(sentence))
        return [
            (token.text, map_tag("en-ptb", "universal", token.tag_)) for token in doc
        ]
