import logging
import re

import pandas as pd
from nltk.stem import WordNetLemmatizer
from pyinflect import getAllInflections

import grounded_ccg_inducer.constants as constants


class ActionRecognizer:
    def __init__(self, use_enrichment=False) -> None:
        logging.info("Recognizing actions with %s...", constants._kinetics_filename)
        self.use_enriched = use_enrichment
        self.kinetics_filename = constants.kinetics_path
        self.kinetics_classes = self.get_kinetics_classes()
        logging.info(
            "Nr of deduplicated Kinetics-700 classes: %s", len(self.kinetics_classes)
        )
        self.enriched_kinetics = self.parse_kinetics_classes()
        if self.use_enriched:
            self.enriched_kinetics = self.enrich_kinetics()
            logging.debug(self.enriched_kinetics)

    def parse_kinetics_classes(self):
        kinetics_classes = {}

        for action in self.kinetics_classes:
            actions = set([action])
            # remove everything between ()
            res = re.sub(r"\([^\)]+\)", "", action)
            words = res.split(" ")
            for w in words:
                if w.endswith("ing"):
                    actions.add(w)
            kinetics_classes[action] = actions

        return kinetics_classes

    def get_kinetics_classes(self):
        kinetics_classes = pd.read_csv(self.kinetics_filename)["name"].values.tolist()
        logging.info("Nr of Kinetics-700 classes: %s", len(kinetics_classes))
        cleaned_kinetics_classes = set()
        for action in kinetics_classes:
            res = re.sub(r"\([^\)]+\)", "", action)
            words = res.split(" ")
            for w in words:
                if w.endswith("ing"):
                    cleaned_kinetics_classes.add(w)
        return cleaned_kinetics_classes

    def enrich_kinetics(self):
        enriched_kinetics = {}
        lemmatizer = WordNetLemmatizer()

        for action in self.kinetics_classes:
            inflected_verbs = set([action])
            inf = lemmatizer.lemmatize(action, "v")
            all_inflections = getAllInflections(inf, pos_type="V")
            for key, val in all_inflections.items():
                inflected_verbs.add(val[0])
            enriched_kinetics[action] = inflected_verbs

        return enriched_kinetics
