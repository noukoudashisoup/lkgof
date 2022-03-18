import argparse 
import os
import json
from lkgof.config import expr_configs
import re
from tqdm import tqdm

import logging 
logging.basicConfig(level=logging.INFO)

problem = 'arxiv'
dataname = 'arxiv-metadata-oai-snapshot.json'
default_category = 'hep-ph'


def main():
    datapath = os.path.join(args.data_dir, dataname)
    c1 = args.category[0]
    c2 = args.category[1]
    print(c1, c2)
    with open(datapath) as json_file:
        cnt = 0
        for line in tqdm(json_file):
            loaded_line = json.loads(line)
            categories = loaded_line['categories'] 
            if c1 in categories and c2 in categories and 'math.PR' in categories:
                cnt += 1
    print('Overlapped articles: {}'.format(cnt))


if __name__ == '__main__':
    dir_problem = expr_configs['problems_path']
    dir_data = os.path.join(dir_problem, problem)
    parser = argparse.ArgumentParser(
        description='Extract arXiv articles of a given cateogry',
        )
    parser.add_argument('--data_dir', default=dir_data,)
    parser.add_argument('-c', '--category', nargs=2, default=default_category,
                        type=str, help='Category to be extracted.')
    args = parser.parse_args()
    main()
