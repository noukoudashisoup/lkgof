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
    outdir = (os.path.join(args.data_dir, 'trimmed_unique') if args.unique
              else os.path.join(args.data_dir, 'trimmed')
              )
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    target_filename = '{}.json'.format(category)
    outpath = os.path.join(outdir, target_filename)
    if os.path.exists(outpath):
        logging.info('Removing the existing file.')
        os.remove(outpath)
    logging.info('Writing to {}'.format(outpath))
    with open(datapath) as json_file, open(outpath, 'a') as outfile:
        for line in tqdm(json_file):
            loaded_line = json.loads(line)
            catlist = loaded_line['categories'].split(' ')
            if category in loaded_line['categories']:
                if args.unique and not len(catlist) == 1:
                    pass
                else:
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
    parser.add_argument('-u', '--unique', default=False,
                        action='store_true', help=('When True, returns articles '
                                         'belonging only to the specified category '
                                         )
    )
    args = parser.parse_args()
    main()
