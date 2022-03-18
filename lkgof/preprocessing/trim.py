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
    category = args.category
    print(category)
    datapath = os.path.join(args.data_dir, dataname)
    outdir = os.path.join(args.data_dir, 'trimmed')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, '{}.json'.format(category))
    if os.path.exists(outpath):
        logging.info('Removing the existing file.')
        os.remove(outpath)

    with open(datapath) as json_file, open(outpath, 'a') as outfile:
        for line in tqdm(json_file):
            loaded_line = json.loads(line)
            # lcats = loaded_line['categories'].split(' ')
            if category in loaded_line['categories']:
                abs = loaded_line['abstract']
                abs = re.sub(r'\$.+?\$', 'xmathx', abs)
                abs = abs.replace('\n', ' ').strip()
                data = {
                    "id": loaded_line['id'],
                    "abstract": abs,
                    }    
                json.dump(data, outfile)
                outfile.write('\n')


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
