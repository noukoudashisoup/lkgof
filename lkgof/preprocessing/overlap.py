import argparse 
import os
import json
from lkgof.config import expr_configs
import re
from tqdm import tqdm
from collections import defaultdict

import logging 
logging.basicConfig(level=logging.INFO)

problem = 'arxiv'
dataname = 'arxiv-metadata-oai-snapshot.json'
default_categories = ['hep-ph', 'quant-ph']

def num_categories():
    datapath = os.path.join(args.data_dir, dataname)
    c1 = args.category[0]
    unilen_dct = defaultdict(int)
    with open(datapath) as json_file:
        cnt = 0
        for line in tqdm(json_file):
            loaded_line = json.loads(line)
            categories = loaded_line['categories']
            split_cat = categories.split(' ')
            if len(split_cat) == 1:
                unilen_dct[split_cat[0]] += 1
                cnt += 1
    # print('Overlapped articles: {}'.format(cnt))
    for k, v in unilen_dct.items():
        print('Number of articles in {}: {}'.format(k, v))

def main():
    datapath = os.path.join(args.data_dir, dataname)
    c1 = args.category[0]
    c2 = args.category[1]
    print(c1, c2)
    with open(datapath) as json_file:
        cnt = 0
        cnt1 = 0
        cnt2 = 0
        for line in tqdm(json_file):
            loaded_line = json.loads(line)
            categories = loaded_line['categories']
            split_cat = categories.split(' ')
            if c1 in categories and c2 in categories:
                cnt += 1
            if c1 in categories:
                cnt1 += 1
            if c2 in categories:
                cnt2 += 1
    print('Overlapped articles: {}'.format(cnt))
    print('Number of articles in {}: {}'.format(c1, cnt1))
    print('Number of articles in {}: {}'.format(c2, cnt2))


if __name__ == '__main__':
    dir_problem = expr_configs['problems_path']
    dir_data = os.path.join(dir_problem, problem)
    parser = argparse.ArgumentParser(
        description='Extract arXiv articles of a given cateogry',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    parser.add_argument('-c', '--category', nargs=2, default=default_categories,
                        type=str, help='Category to be extracted.')
    args = parser.parse_args()
    main()
