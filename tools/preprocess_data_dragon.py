# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import argparse
import math
import json
import os
import sys
import time
import gzip
import glob
import multiprocessing
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object
    nltk_available = False

from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer as build_new_tokenizer
from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset

class CustomLanguageVars(PunktLanguageVars):
    _period_context_fmt = r"""
        \S* # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s* # modifié pour préserver les retours à la ligne
        (?=(?P<after_tok>
            %(NonWord)s              
            |
            (?P<next_tok>\S+)     
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Initialisation du tokenizer (Legacy vs Core)
        if hasattr(self.args, 'legacy_tokenizer') and self.args.legacy_tokenizer:
            Encoder.tokenizer = build_tokenizer(self.args)
        else:
            Encoder.tokenizer = build_new_tokenizer(self.args)

        # Configuration du splitter de phrases
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = os.path.join(os.environ.get("NLTK_DATA", "tokenizers"), "punkt", f"{self.args.lang}.pickle")
            url = f"file:{library}" if os.environ.get("NLTK_DATA") else f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars())
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        nb_tokens = 0
        for key in self.args.json_keys:
            text = data[key]
            sentences = text if isinstance(text, list) else [text]
            doc_ids = []
            sentence_lens = []
            
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend([Encoder.tokenizer.eod] +sentence_ids)
                    sentence_lens.append(len(sentence_ids))
                    nb_tokens += len(sentence_ids)
            
            # Ajout du token EOD à la fin si demandé
            if len(doc_ids) > 0 and self.args.append_eod:
                #doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
                nb_tokens += 1
                
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line), nb_tokens

class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed, total_nb_tokens=0):
        if count % self.args.log_interval == 0:
            elapsed = time.time() - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} docs ({count/elapsed:.1f} docs/s, {mbs:.2f} MB/s), "
                  f"Tokens: {total_nb_tokens:_}", file=sys.stderr)

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        fin = open(input_file_name, 'r', encoding='utf-8')
        
        encoder = Encoder(self.args)
        if hasattr(self.args, 'legacy_tokenizer') and self.args.legacy_tokenizer:
            tokenizer = build_tokenizer(self.args)
        else:
            tokenizer = build_new_tokenizer(self.args)

        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        level = "sentence" if self.args.split_sentences else "document"
        builders = {key: indexed_dataset.IndexedDatasetBuilder(
            f"{output_prefix}_{key}_{level}.bin",
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)) 
            for key in self.args.json_keys}

        proc_start = time.time()
        total_bytes_processed = 0
        total_nb_tokens = 0

        for i, (doc, sentence_lens, bytes_processed, nb_tokens) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            total_nb_tokens += nb_tokens
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed, total_nb_tokens)

        fin.close()
        for key in builders:
            builders[key].finalize(f"{output_prefix}_{key}_{level}.idx")


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    # group.add_argument('--legacy-tokenizer', action='store_true',
    #                    help='Use legacy tokenizer system.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1)%args.partitions

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if args.partitions == 1:
            return


    # encode partition files in parallel
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    if args.legacy_tokenizer:
        tokenizer = build_tokenizer(args)
    else:
        tokenizer = build_new_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':

    main()

