import imp
import os
import numpy as np
import tifffile
from tqdm import tqdm
import json as js
from collections import deque


def convert_images(train_path = "./data/train", layer_ignore = 10):
    """
    Convert images to npy format.
    """
    for root, dirs, files in os.walk(train_path):
        pbar = tqdm(total=len([1 for d in dirs for f in os.listdir(os.path.join(root, d)) if f.endswith(".tif")]), desc="Converting images")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for file in [f for f in os.listdir(dir_path) if f.endswith(".tif")]:
                file_path = os.path.join(dir_path, file)
                img = np.array( tifffile.imread(file_path))[:,:, [i for i in range(13) if i !=layer_ignore]]
                np.save(file_path.replace(".tif", ".npy"), img)
                pbar.update(1)

def count_files(path = "./data/train"):
    """
    Count files in path.
    """
    count = 0
    for root, dirs, files in os.walk(path):
        count += len(files)
    return count	

def create_layer_dist_files(train_path = "./data/train", test_path = "./data/test"):
    """
    Create layer distribution.
    """
    layer_means = {i:deque() for i in range(12)}
    layer_stds = {i:deque() for i in range(12)}
    for data_set in (train_path, test_path):
        dist_file = os.path.join(data_set, "layer_dist.json")
        if data_set is not None:
            for root, dirs, files in os.walk(data_set):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    for file in [f for f in os.listdir(dir_path) if f.endswith(".npy")]:
                        file_path = os.path.join(dir_path, file)
                        img = np.transpose(np.load(file_path))
                        for i, (m,s) in enumerate(zip(np.mean(img, axis = (1,2)), np.std(img, axis = (1,2)))):
                            layer_means[i].append(m)
                            layer_stds[i].append(s)
        dist = {i:{"mean":np.mean(layer_means[i]), "std":np.sqrt(np.mean(layer_stds[i])**2+np.std(layer_means[i])**2)} for i in range(12)}
        with open(dist_file, "w+") as f:
            f.writelines(js.dumps(dist))
                            
                        
              
              