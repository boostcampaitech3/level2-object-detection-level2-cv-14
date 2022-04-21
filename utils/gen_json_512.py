import json
import os
import random
import glob

def gen_512_dataset(input_json, output_dir):

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories'] 

    for i in range(len(images)):
        images[i]['width'] = 512
        images[i]['height'] = 512

    for i in range(len(annotations)):
        annotations[i]['area'] /= 4
        for j in range(4):
            annotations[i]['bbox'][j] /= 2
    
    new_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    with open(output_dir, 'w') as writer:
        json.dump(new_json, writer)


path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, '..', 'dataset')
new_path = os.path.join(path, '..', 'dataset2')

json_list = glob.glob(path + '/*.json') 
for js in json_list:
    name = js.split('/')[-1]
    gen_512_dataset(input_json = js,
                    output_dir = os.path.join(new_path, name))

json_list = glob.glob(path + '/stratified_kfold/*.json')
os.mkdir(new_path+'/stratified_kfold')

for js in json_list:
    direct = js.split('/')[-2]
    name = js.split('/')[-1]
    gen_512_dataset(input_json = js,
                    output_dir = os.path.join(new_path, direct, name))

print("============Finish!==============")



