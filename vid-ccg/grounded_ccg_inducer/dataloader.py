import pickle


class Dataloader:
    def __init__(self, dataset_path, max_length=15) -> None:
        self._dataset_path = dataset_path
        self.max_length = max_length
        self._dataset = self.download_dataset()
        self.tagged_dataset = None
        self.clip_dataset = None

    @property
    def dataset(self):
        return self._dataset

    def download_dataset(self):
        didemo_captions = pickle.load(open(self._dataset_path, "rb"))
        counter = 0
        for clip, sentences in didemo_captions.items():
            counter += len(sentences)
        print(counter)

        for clip, sentences in didemo_captions.items():
            for sentence in sentences:
                if sentence == ["traditional", "dancing"]:
                    sentences.remove(["traditional", "dancing"])
                    print("FOUND IT")
        counter = 0
        for clip, sentences in didemo_captions.items():
            counter += len(sentences)
        print(counter)

        return didemo_captions

    def preprocess_dataset(self, debug=False):
        if debug:
            self.print_statistics()

        self.remove_capitalization()
        if debug:
            self.print_statistics()

        self.remove_punct()
        if debug:
            self.print_statistics()

        self.remove_empty_lines()
        if debug:
            self.print_statistics()

        self.remove_other()
        if debug:
            self.print_statistics()


    def remove_empty_lines(self):
        self.tagged_dataset = [
            [(word, tag) for word, tag in sentence if not word.isspace()]
            for sentence in self.tagged_dataset
        ]
        self.clip_dataset = {
            key: [
                [(word, tag) for word, tag in sentence if not word.isspace()]
                for sentence in val
            ]
            for key, val in self.clip_dataset.items()
        }

    def remove_capitalization(self):
        self.tagged_dataset = [
            [(word.lower(), tag) for word, tag in sentence]
            for sentence in self.tagged_dataset
        ]
        self.clip_dataset = {
            key: [[(word.lower(), tag) for word, tag in sentence] for sentence in val]
            for key, val in self.clip_dataset.items()
        }

    def remove_punct(self):
        self.remove_punct_petrov()

    def remove_punct_petrov(self):
        self.tagged_dataset = [
            [(word, tag) for word, tag in sentence if tag != "."]
            for sentence in self.tagged_dataset
        ]
        self.clip_dataset = {
            key: [
                [(word, tag) for word, tag in sentence if tag != "."]
                for sentence in val
            ]
            for key, val in self.clip_dataset.items()
        }

    def remove_punct_upos(self):
        self.tagged_dataset = [
            [(word, tag) for word, tag in sentence if tag != "PUNCT"]
            for sentence in self.tagged_dataset
        ]
        self.clip_dataset = {
            key: [
                [(word, tag) for word, tag in sentence if tag != "PUNCT"]
                for sentence in val
            ]
            for key, val in self.clip_dataset.items()
        }

    def remove_other(self):
        new_sentences = []
        for sentence in self.tagged_dataset:
            tags = [tag for word, tag in sentence]
            if not ("X" in tags or "SYM" in tags):
                new_sentences.append(sentence)

        self.tagged_dataset = new_sentences

        new_clip_sentences = []
        new_dict = {}
        for key, val in self.clip_dataset.items():
            new_clip_sentences = []
            for sentence in val:
                tags = [tag for word, tag in sentence]
                if not ("X" in tags or "SYM" in tags):
                    new_clip_sentences.append(sentence)
            new_dict[key] = new_clip_sentences

    def remove_long(self):
        self.tagged_dataset = [
            sentence
            for sentence in self.tagged_dataset
            if len(sentence) <= self.max_length
        ]

    def print_capitals(self):
        total_words = 0
        capitalized_words = 0
        for i, sentence in enumerate(self.tagged_dataset):
            for word, tag in sentence:
                total_words += 1
                if not word.islower():
                    if word.isalpha():
                        capitalized_words += 1
        print(
            f"there are {capitalized_words} capitalized words out of {total_words} words."
        )

    def print_punct(self):
        total_words = 0
        punct_words = 0
        for i, sentence in enumerate(self.tagged_dataset):
            for word, tag in sentence:
                total_words += 1
                if tag == ".":
                    punct_words += 1
        print(f"there are {punct_words} punct words out of {total_words} words.")

    def print_other(self):
        total_sentences = 0
        other_sentences = 0
        for i, sentence in enumerate(self.tagged_dataset):
            total_sentences += 1
            tags = [tag for word, tag in sentence]
            if "X" in tags or "SYM" in tags:
                other_sentences += 1
        print(
            f"there are {other_sentences} sentences with other words out of {total_sentences} sentences."
        )

    def print_long(self):
        total_sentences = 0
        long_sentences = 0
        for i, sentence in enumerate(self.tagged_dataset):
            total_sentences += 1
            if len(sentence) > self.max_length:
                long_sentences += 1

        print(
            f"there are {long_sentences} sentences longer than {self.max_length} out of {total_sentences} sentences."
        )

    def print_empty_lines(self):
        total_words = 0
        empty_words = 0
        for i, sentence in enumerate(self.tagged_dataset):
            for word, tag in sentence:
                total_words += 1
                if word.isspace():
                    empty_words += 1

        print(f"there are {empty_words} empty words out of {total_words} words.")

    def set_tagged_dataset(
        self, tagged_data: dict[str, list[list[tuple[str, str]]]]
    ) -> None:
        self.clip_dataset = tagged_data
        sentences = []
        for _, item in tagged_data.items():
            for sent in item:
                sentences.append(sent)

        self.tagged_dataset = sentences

    def print_sentence_count(self):
        total_sentence_count = len(self.tagged_dataset)
        print(f"there are {total_sentence_count} sentences.")

    def print_word_count(self):
        total_word_count = 0
        for sentence in self.tagged_dataset:
            for word, _ in sentence:
                total_word_count += 1
        print(f"there are {total_word_count} words.")

    def print_statistics(self):
        self.print_sentence_count()
        self.print_word_count()
        self.print_capitals()
        self.print_punct()
        self.print_other()
        self.print_empty_lines()
        print()
