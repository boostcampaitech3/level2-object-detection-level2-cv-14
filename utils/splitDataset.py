import json
import os
import random

def split_dataset(input_json, output_dir, val_ratio, random_seed):
    random.seed(random_seed)

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    image_ids = [x.get('id') for x in images]
    image_ids.sort()
    random.shuffle(image_ids)

    num_val = int(len(image_ids) * val_ratio)
    num_train = len(image_ids) - num_val

    for index in range(int(1/val_ratio)):
        image_ids_val, image_ids_train = set(image_ids[index*num_val:(index+1)*num_val]), set(image_ids[:index*num_val]+image_ids[(index+1)*num_val:])

        train_images = [x for x in images if x.get('id') in image_ids_train]
        val_images = [x for x in images if x.get('id') in image_ids_val]
        train_annotations = [x for x in annotations if x.get('image_id') in image_ids_train]
        val_annotations = [x for x in annotations if x.get('image_id') in image_ids_val]

        train_data = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': categories,
        }

        val_data = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': categories,
        }

        output_train_json = os.path.join(output_dir, f'train_fold{index}.json')
        output_val_json = os.path.join(output_dir, f'val_fold{index}.json')
        output_train_csv = os.path.join(output_dir, f'train_fold{index}.csv')
        output_val_csv = os.path.join(output_dir, f'val_fold{index}.csv')

        print(f'write {output_train_json}')
        with open(output_train_json, 'w') as train_writer:
            json.dump(train_data, train_writer)

        print(f'write {output_val_json}')
        with open(output_val_json, 'w') as val_writer:
            json.dump(val_data, val_writer)

path = os.path.dirname(os.path.abspath(__file__)) 
path = os.path.join(path, '..', 'dataset')

split_dataset(input_json=os.path.join(path, 'train.json'),
              output_dir=path,
              val_ratio=0.1,
              random_seed=42)