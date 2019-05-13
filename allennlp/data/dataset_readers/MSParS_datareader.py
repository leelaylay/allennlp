import glob
import html
import json
import logging
import os
import re
import string
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from nltk.corpus import stopwords
from overrides import overrides


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EMBEDDING_DIM = 100
HIDDEN_DIM = 200


@DatasetReader.register("MSParS")
class SemEvalDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from SemEval2017 Task4 SubsetA.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self.SEPARATOR = "\t"
        # data id set
        self.data = set()
        # stop words list
        self.stop = stopwords.words('english') + list(string.punctuation) + ['rt', 'via']
        # tokenizer
        self._tokenizer = tokenizer or WordTokenizer()
        # token_indexers
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }

    @overrides
    def _read(self, folder_path: str):
        # read files below the folder
        files = glob.glob(os.path.join(folder_path, "*.txt"))

        for file_path in files:
            with open(cached_path(file_path), "r") as data_file:
                logger.info("Reading instances from lines in file at: %s",
                            file_path)
                for index, line in enumerate(data_file):
                    columns = line.rstrip().split(self.SEPARATOR)
                    if not columns:
                        continue
                    if len(columns)<3:
                        logger.info(index)
                        logger.info(columns)
                    tweet_id = columns[0]
                    sentiment = columns[1]
                    text = columns[2:]
                    text = self.clean_text(''.join(text))
                    if tweet_id not in self.data:
                        self.data.add(tweet_id)
                        yield self.text_to_instance(sentiment, text)
                    else:
                        continue

    @overrides
    def text_to_instance(self, sentiment: str,
                         text: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'tokens': text_field}
        if sentiment is not None:
            fields['label'] = LabelField(sentiment)
        return Instance(fields)

    def clean_text(self, text: str):
        """
        Remove extra quotes from text files and html entities
        Args:
            text (str): a string of text
        Returns: (str): the "cleaned" text
        """
        text = text.rstrip()
        if '""' in text:
            if text[0] == text[-1] == '"':
                text = text[1:-1]
            text = text.replace('\\""', '"')
            text = text.replace('""', '"')

        text = text.replace('\\""', '"')

        text = html.unescape(text)

        text = ' '.join(text.split())
        return text

