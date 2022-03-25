import os
import glob
from PIL import Image
from tqdm.auto import tqdm

def gen_image(img_path, out_dir):
    img = Image.open(img_path)
    img = img.resize((512, 512))

    img.save(out_dir)


path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, '..', 'dataset')

os.mkdir(os.path.join(path, '..', 'dataset2'))
os.mkdir(os.path.join(path, '..', 'dataset2', 'train'))

img_list = glob.glob(path + '/train/*.jpg')

for img_path in tqdm(img_list):
    _path = img_path.split('/')
    _path[-3] = _path[-3] + str(2)
    new_path = '/'.join(_path)

    gen_image(img_path, new_path)

print("============Finish!==============")

