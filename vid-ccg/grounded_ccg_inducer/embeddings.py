import logging

import gensim.models.keyedvectors as word2vec


class Embeddings:
    def __init__(self, fasttext_filename, topn) -> None:
        self.fasttext_filename = fasttext_filename
        # Only load model upon use, loading takes forever...
        self.model = None
        self.topn = topn

    def most_similar(self, word):
        if self.model is None:
            logging.info("Loading embedding model...")
            try:
                self.model = word2vec.KeyedVectors.load_word2vec_format(
                    self.fasttext_filename, binary=False
                )
            except Exception as e:
                raise SystemExit(f"Error while loading gensim wiki-news: {e}")

        words = [word]
        if "_" in word:
            words = word.split("_")
        if " " in word:
            words = word.split(" ")

        return self.model.most_similar(positive=words, topn=self.topn)
