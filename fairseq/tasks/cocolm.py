# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os

from omegaconf import MISSING, II, OmegaConf

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset2,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SpanDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES


logger = logging.getLogger(__name__)


# @dataclass
# class COCOLMConfig(FairseqDataclass):
#     data: str = field(
#         default=MISSING,
#         metadata={
#             "help": "colon separated path to data directories list, \
#                             will be iterated upon during epochs in round-robin manner"
#         },
#     )
#     sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
#         default="none",
#         metadata={
#             "help": 'If omitted or "none", fills each sample with tokens-per-sample '
#             'tokens. If set to "complete", splits samples only at the end '
#             "of sentence, but may include multiple sentences per sample. "
#             '"complete_doc" is similar but respects doc boundaries. '
#             'If set to "eos", includes only one sentence per sample.'
#         },
#     )
#     tokens_per_sample: int = field(
#         default=1024,
#         metadata={"help": "max number of tokens per sample for LM dataset"},
#     )
#     mask_prob: float = field(
#         default=0.15,
#         metadata={"help": "probability of replacing a token with mask"},
#     )
#     leave_unmasked_prob: float = field(
#         default=0.0,
#         metadata={"help": "probability that a masked token is unmasked"},
#     )
#     random_token_prob: float = field(
#         default=0.0,
#         metadata={"help": "probability of replacing a token with a random token"},
#     )
#     freq_weighted_replacement: bool = field(
#         default=False,
#         metadata={"help": "sample random replacement words based on word frequencies"},
#     )
#     mask_whole_words: bool = field(
#         default=False,
#         metadata={"help": "mask whole words; you may also want to set --bpe"},
#     )
#     mask_multiple_length: int = field(
#         default=1,
#         metadata={"help": "repeat the mask indices multiple times"},
#     )
#     mask_stdev: float = field(
#         default=0.0,
#         metadata={"help": "stdev of the mask length"},
#     )
#     shorten_method: SHORTEN_METHOD_CHOICES = field(
#         default="none",
#         metadata={
#             "help": "if not none, shorten sequences that exceed --tokens-per-sample"
#         },
#     )
#     shorten_data_split_list: str = field(
#         default="",
#         metadata={
#             "help": "comma-separated list of dataset splits to apply shortening to, "
#             'e.g., "train,valid" (default: all dataset splits)'
#         },
#     )
#     seed: int = II("common.seed")
#     span: float = field(
#         default=0.0,
#         metadata={"help": "span seq length"},
#     )
#     add_span_cls: bool = field(
#         default=False,
#         metadata={"help": "add cls token to span tokens"},
#     )
#     mask_cls: bool = field(
#         default=False,
#         metadata={"help": "mask cls token"},
#     )
#     include_target_tokens: bool = field(
#         default=False,
#         metadata={
#             "help": "include target tokens in model input. this is used for data2vec"
#         },
#     )


@register_task("cocolm")
class COCOLMTask(FairseqTask):

    # args: COCOLMConfig

    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # get mask token
        self.mask_idx = dictionary.index("[MASK]")
        self.args = args
        
    # @staticmethod
    # def add_args(parser):
    #     """Add task-specific arguments to the parser."""
    #     parser.add_argument('--sample-break-mode', default='complete',
    #                         choices=['none', 'complete', 'complete_doc', 'eos'],
    #                         help='If omitted or "none", fills each sample with tokens-per-sample '
    #                              'tokens. If set to "complete", splits samples only at the end '
    #                              'of sentence, but may include multiple sentences per sample. '
    #                              '"complete_doc" is similar but respects doc boundaries. '
    #                              'If set to "eos", includes only one sentence per sample.')
    #     parser.add_argument('--tokens-per-sample', default=512, type=int,
    #                         help='max number of total tokens over all segments '
    #                              'per sample for BERT dataset')
    #     parser.add_argument('--mask-prob', default=0.15, type=float,
    #                         help='probability of replacing a token with mask')
    #     parser.add_argument('--leave-unmasked-prob', default=0.0, type=float,
    #                         help='probability that a masked token is unmasked')
    #     parser.add_argument('--random-token-prob', default=0.0, type=float,
    #                         help='probability of replacing a token with a random token')
    #     parser.add_argument('--freq-weighted-replacement', action='store_true',
    #                         help='sample random replacement words based on word frequencies')
    #     parser.add_argument('--mask-whole-words', default=False, action='store_true',
    #                         help='mask whole words; you may also want to set --bpe')

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.args.mask_whole_words
            else None
        )

        src_dataset, tgt_dataset = MaskTokensDataset2.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
            mask_cls=self.args.mask_cls,
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_dataset))
        
        span_tokens = SpanDataset(dataset, span=self.args.span, seed=self.args.seed + 1)
        # [CLS] could be removed by cropping; add back
        if self.args.add_span_cls:
            span_tokens = PrependTokenDataset(span_tokens, self.source_dictionary.bos())
        span_tokens = RightPadDataset(
            span_tokens,
            pad_idx=self.source_dictionary.pad(),
        )

        target_dataset = RightPadDataset(
            tgt_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        input_dict = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "span_tokens": span_tokens,
            "src_lengths": NumelDataset(src_dataset, reduce=False),
        }
        if self.args.include_target_tokens:
            input_dict["target_tokens"] = target_dataset

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": input_dict,
                    "target": target_dataset,
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
