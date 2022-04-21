import os
import json
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

path = os.path.dirname(os.path.abspath(__file__)) 
data_path = os.path.join(path, '..', 'dataset')

images_path = os.path.join(data_path, 'train')
annotations_path = os.path.join(data_path, 'train.json')

def main(args):
    with open(annotations_path, 'r') as f:
        train_json = json.loads(f.read())
        images = train_json['images']
        categories = train_json['categories']
        annotations = train_json['annotations']

    annotations_df = pd.DataFrame.from_dict(annotations)
    x = images
    y = [[0] * len(categories) for _ in range(len(images))]
    for anno in annotations:
        y[anno['image_id']][anno['category_id']] += 1

    mskf = MultilabelStratifiedKFold(n_splits=args.n_split, shuffle=True)

    path = args.path

    if not os.path.exists(path):
        os.mkdir(path)

    for idx, (train_index, val_index) in tqdm(enumerate(mskf.split(x, y)), total=args.n_split):
        train_dict = dict()
        val_dict = dict()
        for i in ['info', 'licenses', 'categories']:
            train_dict[i] = train_json[i]
            val_dict[i] = train_json[i]
        train_dict['images'] = np.array(images)[train_index].tolist()
        val_dict['images'] = np.array(images)[val_index].tolist()
        train_dict['annotations'] = annotations_df[annotations_df['image_id'].isin(train_index)].to_dict('records')
        val_dict['annotations'] = annotations_df[annotations_df['image_id'].isin(val_index)].to_dict('records')

        train_dir = os.path.join(path, f'cv_train_{idx + 1}.json')
        val_dir = os.path.join(path, f'cv_val_{idx + 1}.json')
        with open(train_dir, 'w') as train_file:
            json.dump(train_dict, train_file)

        with open(val_dir, 'w') as val_file:
            json.dump(val_dict, val_file)

    print("Done Make files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default=os.path.join(path, '..', 'dataset', 'stratified_kfold'))
    parser.add_argument('--n_split', '-n', type=int, default=10)
    arg = parser.parse_args()
    main(arg)