import pickle
import lmdb
import os
import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import constants


def to_lmdb(root_path, lmdb_path):

    image_paths = [x.split(".")[0] for x in os.listdir(os.path.join(root_path, "img"))]
    print('#images: ', len(image_paths))
    print("Generate LMDB to %s" % lmdb_path)

    image_size = Image.open(os.path.join(root_path, 'img', f'{image_paths[0]}.png')).size
    pixels = image_size[0] * image_size[1] * len(image_paths)

    print("Pixels in split: ", pixels)

    map_size = pixels * 4 + 1500 * 320 * 240 * 4

    print("Estimated Size: ", map_size / (1024 * 1024 * 1024))

    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir, map_size=map_size, readonly=False, meminit=False, map_async=True, writemap=True)

    txn = db.begin(write=True)

    key_list = []
    # image_paths.remove('remap')
    for idx, path in tqdm(enumerate(image_paths)):
        jpg_path = os.path.join(root_path, 'img', f'{path}.png')
        png_path = os.path.join(root_path, 'lbl', f'{path}.png')
        image = np.array(Image.open(jpg_path).convert('RGB'), dtype=np.uint8)
        label = np.array(Image.open(png_path), dtype=np.uint8)
        # label -= 1
        # label[label == -1] = 255
        # label[label >= 40] = 255
        label[label == 0] = 1    # background
        label[label == 128] = 2  # background
        label[label == 255] = 3  # background

        txn.put(u'{}'.format(path).encode('ascii'), pickle.dumps(np.dstack((image, label)), protocol=3))
        key_list.append(path)

    print('Committing..')
    txn.commit()

    print('Writing keys..')
    keys = [u'{}'.format(k).encode('ascii') for k in key_list]

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys, protocol=3))
        txn.put(b'__len__', pickle.dumps(len(keys), protocol=3))

    print('Syncing..')
    db.sync()
    db.close()


if __name__ == '__main__':
    PATH_TO_SELECTIONS = os.path.join(constants.HDD_DATASET_ROOT, 'train')
    PATH_TO_LMDB = os.path.join(constants.HDD_DATASET_ROOT, 'dataset2.lmdb')
    to_lmdb(PATH_TO_SELECTIONS, PATH_TO_LMDB)
