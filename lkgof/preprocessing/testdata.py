import argparse
import pickle
import numpy as np
import os
import lkgof.util as util 
from lkgof.config import expr_configs
from tqdm import tqdm

import logging 
logging.basicConfig(level=logging.INFO)

problem = 'arxiv'
default_categories = ['stat.ME', 'math.PR', 'stat.TH']

def main():
    dictdir = os.path.join(args.data_dir, 'dicts')
    dict_filename = 'dict_{}.pkl'.format('_'.join(args.categories))
    dict_path = os.path.join(dictdir, dict_filename)
    with open(dict_path, 'rb') as f:
        dct = pickle.load(f) 
    token2id = dct.token2id

    subsample = 100
    seed = 13
    test_category = args.testcategory
    doc_data_dir = os.path.join(args.data_dir, 'tokenized')
    path = os.path.join(doc_data_dir, '{}.pkl'.format(test_category))
    with open(path, 'rb') as f:
        docs = pickle.load(f)

    data = []
    with util.NumpySeedContext(seed):
        for text in tqdm(docs):
            word_ids = np.array([token2id[token] for token in text])
            len_word_ids = len(word_ids)
            if len_word_ids >= subsample:
                ind = np.random.choice(len_word_ids, subsample, replace=False)
                word_ids = word_ids[ind]
                data.append(word_ids)
    data = np.vstack(data).astype(np.int)
    outdir = os.path.join(args.data_dir, 'testdata', '_'.join(args.categories))
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
        description='Create a test dataset for a given combionations of arXiv categories',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    parser.add_argument('-c', '--categories', nargs=3, default=default_categories,
                        type=str, help='Category combination for a given problem.')
    parser.add_argument('-t', '--testcategory', default=default_categories[2],
                        type=str, help='Category of the test dataset.')
    args = parser.parse_args()
 
    main()
