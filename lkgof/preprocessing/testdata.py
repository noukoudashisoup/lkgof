import argparse
import pickle
import gensim
import numpy as np
import os
import lkgof.util as util 
import scipy.stats as stats
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from lkgof.config import expr_configs
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging 
logging.basicConfig(level=logging.INFO)

problem = 'arxiv'
categories = ['stat.AP', 'stat.ME', 'stat.TH']
# categories = ['cs.LG', 'stat.ME', 'stat.TH']
# categories = ['stat.ME', 'math.PR', 'stat.TH']
dict_filename = 'dict_{}.pkl'.format('_'.join(categories))

def main():
    dictdir = os.path.join(args.data_dir, 'dicts')
    dict_path = os.path.join(dictdir, dict_filename)
    with open(dict_path, 'rb') as f:
        dct = pickle.load(f) 
    token2id = dct.token2id

    subsample = 100
    seed = 13
    test_category = 'stat.TH'
    doc_data_dir = os.path.join(args.data_dir, 'tokenized')
    path = os.path.join(doc_data_dir, '{}.pkl'.format(test_category))
    with open(path, 'rb') as f:
        docs = pickle.load(f)

    data = []
    with util.NumpySeedContext(seed):
        for text in docs:
            word_ids = np.array([token2id[token] for token in text])
            len_word_ids = len(word_ids)
            if len_word_ids >= subsample:
                ind = np.random.choice(len_word_ids, subsample, replace=False)
                word_ids = word_ids[ind]
                data.append(word_ids)
    data = np.vstack(data).astype(np.int)
    outdir = os.path.join(args.data_dir, 'testdata', '_'.join(categories))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, '{}'.format(test_category))
    logging.info('Saving to {}.npy'.format(outfile))
    print(data.shape)
    np.save(outfile, data)

if __name__ == '__main__':
    dir_problem = expr_configs['problems_path']
    dir_data = os.path.join(dir_problem, problem)
    parser = argparse.ArgumentParser(
        description='Extract arXiv articles of a given cateogry',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    args = parser.parse_args()
 
    main()
