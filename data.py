import csv
import json
import os
import glob
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud)  # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class Text2Cap(Dataset):
    def __init__(self, args, partition='train'):
        # args.text2shape_csv = './data/text2shape.csv'
        # args.shapenet_dir = './data/shapenetcorev2_hdf5_2048/'
        # args.tokenizer = 'microsoft/prophetnet-base-uncased'
        # args.num_points = 1024
        self.args = args
        self.num_points = args.num_points
        self.split = partition
        self.text2shape = self._load_text2shape_csv(args.text2shape_csv)  # both test and train are in text2shape
        self.shapenet = self._load_shapenet(args.shapenet_dir, split=partition, num_points=args.num_points)
        # assert len(self.text2shape) == len(self.shapenet), 'not all shap in text2shape appears in  shapenetcore'
        self._drop_shape()  # drop shape not in currently split.
        self.caption_id_list = list(self.text2shape.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def __len__(self):
        return len(self.caption_id_list)

    def get_shape_id(self, idx):
        caption_id = self.caption_id_list[idx]
        return self.text2shape[caption_id]['shape_id']

    def __getitem__(self, idx):
        caption_id = self.caption_id_list[idx]
        shape_id = self.text2shape[caption_id]['shape_id']
        pointcloud = self.shapenet[shape_id]
        caption = self.text2shape[caption_id]['description']
        if self.split == 'train':
            pointcloud = random_point_dropout(pointcloud)  # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        caption_toks = self.tokenizer(caption, max_length=self.args.max_length, padding='max_length', truncation=True,
                                      return_tensors='pt')
        caption_ids = caption_toks.input_ids.squeeze(0)
        attention_mask = caption_toks.attention_mask.squeeze(0)
        return pointcloud, caption_ids, attention_mask

    def _drop_shape(self):
        used_shape_ids = set()
        to_be_dropped = []
        for cap_id in self.text2shape:
            shape_id = self.text2shape[cap_id]['shape_id']
            if shape_id not in self.shapenet:
                to_be_dropped.append(cap_id)
            else:
                used_shape_ids.add(shape_id)
        for cap_id in to_be_dropped:
            del self.text2shape[cap_id]

        to_be_dropped = []
        for shape_id in self.shapenet:
            if shape_id not in used_shape_ids:
                to_be_dropped.append(shape_id)
        for shape_id in to_be_dropped:
            del self.shapenet[shape_id]

    @staticmethod
    def _load_text2shape_csv(path):
        samples = {}
        with open(path, 'r', newline='', encoding='utf8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                samples[row['id']] = {
                    'description': row['description'],
                    'shape_id': row['modelId'],
                    'category': row['category'],
                }

        return samples

    def _load_shapenet(self, shapenet_dir, split="train", num_points=1024):
        root_path = Path(shapenet_dir)
        dataset = {}
        shard_count = len(list(root_path.glob(f'{split}*.h5')))
        for idx in range(shard_count):
            id2file_path = root_path / f'{split}{idx}_id2file.json'
            h5_path = root_path / f'{split}{idx}.h5'
            with open(id2file_path, 'r') as f:
                id2file = json.load(f)
            with h5py.File(h5_path, 'r') as f:
                for i, npy_path in enumerate(id2file):
                    obj_id = npy_path.split('/')[-1].split('.')[0]
                    dataset[obj_id] = f['data'][i][:num_points].copy()  # 2048, 3
        return dataset


class ArgMock:
    text2shape_csv = '/mnt/finetune/text2shape/captions.tablechair.csv'
    shapenet_dir = '/mnt/finetune/text2shape/shapenetcorev2_hdf5_2048/'
    tokenizer = 'microsoft/prophetnet-large-uncased'
    num_points = 1024
    batch_size = 32
    max_length = 256
    val_batch_size = 48


if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
    args = ArgMock()
    train = Text2Cap(args, partition='train')
    val = Text2Cap(args, partition='val')
    test = Text2Cap(args, partition='test')
    print(len(train))
    print(len(val))
    print(len(test))

    train_loader = DataLoader(train, num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(len(train_loader))
    sample = next(iter(train_loader))
    print(sample)

    print(len(val_loader))
    sample = next(iter(val_loader))
    print(sample)

    print(len(test_loader))
    sample = next(iter(test_loader))
    print(f"{sample[0].shape} pcl, {sample[1].shape} caption")
