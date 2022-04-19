import argparse
import pickle
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
seed = 3
default_categories = ['stat.ME', 'math.PR', 'stat.TH']

def main():
    doc_data_dir = os.path.join(args.data_dir, 'tokenized')
    categories = args.categories
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
    with util.NumpySeedContext(seed):
        for c in tqdm(categories):
            logging.info('Training on {}'.format(c))
            model_out_filename = 'LDA_{}'.format(c)
            out_path = os.path.join(outdir, model_out_filename)
            train_data = [dct.doc2bow(text) for text in alldocs[c]]
            lda = LdaModel(id2word=dct, passes=passes, 
                    num_topics=n_topics, alpha='auto')
            lda.update(train_data, update_every=1.,)
            lda.save(out_path, separately=['alpha'])

    outdir = os.path.join(args.data_dir, 'dicts')
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    dict_out_filename = 'dict_{}.pkl'.format('_'.join(categories))
    dict_path = os.path.join(outdir, dict_out_filename)
    print(dict_path)
    with open(dict_path, 'wb') as f:
        pickle.dump(dct, f) 


if __name__ == '__main__':
    dir_problem = expr_configs['problems_path']
    dir_data = os.path.join(dir_problem, problem)
    parser = argparse.ArgumentParser(
        description='Train LDA models for a given combination of categories.',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    parser.add_argument('-c', '--categories', nargs=3, default=default_categories,
                        type=str, help='Categories to train models on.')
    args = parser.parse_args()
 
    main()
