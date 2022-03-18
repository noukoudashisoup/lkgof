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

import logging 
logging.basicConfig(level=logging.INFO)

problem = 'arxiv'

def main():
    doc_data_dir = os.path.join(args.data_dir, 'tokenized')
    categories = ['stat.AP', 'stat.ME', 'stat.TH']
    #categories = ['cs.LG', 'stat.ME', 'stat.TH']
    #categories = ['stat.ME', 'math.PR', 'stat.TH']
    alldocs = {}
    dct = Dictionary()
    for c in categories:
        path = os.path.join(doc_data_dir, '{}.pkl'.format(c))
        with open(path, 'rb') as f:
            alldocs[c] = pickle.load(f)
    for key, docs in alldocs.items():
        dct.add_documents(docs)

    outdir = os.path.join(args.data_dir, 'models',
                          '_'.join(categories),
    )
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    n_topics = 100
    passes  = 5
    for c in tqdm(categories):
        logging.info('Training on {}'.format(c))
        model_out_filename = 'LDA_{}'.format(c)
        out_path = os.path.join(outdir, model_out_filename)
        train_data = [dct.doc2bow(text) for text in alldocs[c]]
        lda = LdaModel(id2word=dct, passes=passes, 
                num_topics=n_topics, )
        lda.update(train_data, update_every=1)
        lda.save(out_path)

    outdir = os.path.join(args.data_dir, 'dicts')
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    dict_out_filename = 'dict_{}.pkl'.format('_'.join(categories))
    dict_path = os.path.join(outdir, dict_out_filename)
    with open(dict_path, 'wb') as f:
        pickle.dump(dct, f) 


if __name__ == '__main__':
    dir_problem = expr_configs['problems_path']
    dir_data = os.path.join(dir_problem, problem)
    parser = argparse.ArgumentParser(
        description='Extract arXiv articles of a given cateogry',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    args = parser.parse_args()
 
    main()
