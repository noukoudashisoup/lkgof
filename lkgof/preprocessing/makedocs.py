import argparse 
import os
import json
from lkgof.config import expr_configs
import re
from tqdm import tqdm
import pickle
import logging 
logging.basicConfig(level=logging.INFO)
import nltk
from nltk.corpus import stopwords
from nltk.text import TextCollection

nltk.download('stopwords')
nltk.download('punkt')
problem = 'arxiv'
default_category = 'stat.ME'


extra_stop_words = [
    'however', 
    "'s", 'via', 'using', 'study',
    'present', 'paper', 'result', 'results', 'also', 'based',
    'show', 'use', "''", 'consider', 'considers', 'considered',
    'resp.', 
    ]

stop_words = set(stopwords.words('english')+extra_stop_words)

def remove_meaningless_words(text):
    # Fix corrupted characters
    text = text.replace('â\x80\x99', "'")
    text = text.replace('\x7f', "")
    text = text.replace('â\x88\x9e', "'")
    text = text.replace('â\x89¤', "'")
    text = text.replace('â\x80\x94', "'")
    text = text.replace('â\x80\x93', "-")
    # Accented characters
    text = re.sub(r'\\[cglvrHvr\~\"\^\`\']{1}', '', text)
    text = re.sub(r'\\(aa|ss)', '', text)
    # numbers 
    text = re.sub(r'-*\+*\d+\.*\d*%*','xnumx', text)
    text = re.sub(r'[0-9].{0,1}', '', text)
    text = re.sub(r'\([0-9]*\)', '', text)
    # Matches operations between numbers expressed by a single character, such as * - + ^ /
    text = re.sub(r'\(*xnumx\)*(\s*\S{1}\s*\(*xnumx\)*)*', 'xnumx', text)
    # brackets
    text = re.sub(r'[()\[\]\{\}]', '', text)
    return text

def main():
    category = args.category
    # logging.info('Category: {}'.format(category))
    datapath = os.path.join(args.data_dir, 'trimmed', 
            '{}.json'.format(category))
    outdir = os.path.join(args.data_dir, 'tokenized')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data = []
    with open(datapath) as json_file: 
        for line in json_file:
            data.append(json.loads(line)['abstract'])

    
    docs = []
    for abst in tqdm(data):
        if not 'withdrawn' in abst:
            abst = remove_meaningless_words(abst)
            tokenized_abst = nltk.word_tokenize(abst.lower())
            doc = [ token for token in tokenized_abst if 
                        (token not in stop_words) and 
                        (token not in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n')
            ]
            docs.append(doc)

    outpath = os.path.join(outdir, '{}.pkl'.format(category))
    if os.path.exists(outpath):
        logging.info('Removing the existing file.')
        os.remove(outpath)

    with open(outpath, 'wb') as outfile:
        logging.info('Saving to {}'.format(outfile))
        pickle.dump(docs, outfile)




if __name__ == '__main__':
    dir_problem = expr_configs['problems_path']
    dir_data = os.path.join(dir_problem, problem)
    parser = argparse.ArgumentParser(
        description='Extract arXiv articles of a given cateogry',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    parser.add_argument('-c', '--category', default=default_category,
                        type=str, help='Category to be extracted.')
    args = parser.parse_args()
    main()
