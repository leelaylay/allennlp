from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mspars")
class MSParSDatasetReader(DatasetReader):
    """
    Read MSParS data.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)
            lines = data_file.readlines()
            lines_num = len(lines)
            for i in range(0, lines_num, 5):
                line_question = lines[i]
                line_logical_form = lines[i+1]
                line_parameters = lines[i+2]
                line_question = lines[i+3]
                source_sequence = line_question.split("\t")[-1]
                target_sequence = line_logical_form.split("\t")[-1]
                pamameter = line_parameters.split("\t")[-1]
                label = line_question.split("\t")[-1]

                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(
                tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})
