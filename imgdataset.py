import pandas as pd
import numpy as np
import os
import transforms
from torch.utils.data import Dataset
import helpers
import normalizers
import json as js
import torch

class CustomImageDataset(Dataset):
    """
    Custom Dataset class for Training the networks, loading this dataset with a Pytorch DataLoader allows for convenient shuffling and batching of the data.
    """
    def __init__(self, img_labes=None, img_dir=R"./data/train/", transform=None, target_transform=helpers.torch_mapping, normalizer = None, train_test_split=0.5, is_train=True):
        """_summary_

        Args:
            img_labes (pandas.DataFrame, optional): This argument expects a Pandas DataFrame with the information of the train.csv file. Passing it as argument allows for convenient handling of subsets as own datasets. Defaults to None in which case the class tries to load the entire 'train.csv' from a default location.
            img_dir (regexp, optional): The relative path from the current working directory to the location where the train data is saved. Defaults to R"./data/train/".
            transform (Callable, optional): Some function which applies transformations to each of the images. Defaults to None in which case the transformations from 'CustomImageDataset.pipeline' are applied.
            target_transform (Callable, optional): Function that translates the litteral labels into a onehot classification vector. Defaults to helpers.torch_mapping.
            normalizer (Callable, optional): This argument allows to hook a function to the CustomImageDataset instance, which normalizes the image layers. Defaults to None. (During the projekt we tested various ways to normalize the data)
            train_test_split (float, optional): Fraction of labels to keep in the train set when a test set is cut out. Defaults to 0.5.
            is_train (bool, optional): Boolean which determines wether to cut out a Test set from the labels provided to 'img_labels'. Defaults to True.
        """
        if img_labes is None:
            img_labes = pd.read_csv(R"data\train\train.csv")
        self.img_labels = img_labes
        # replace the extension of the image files, we converted them earlier and removed the 10th layer
        self.img_labels[self.img_labels.columns[0]] = self.img_labels.iloc[:,0].str.replace(".tif", ".npy")
        self.img_dir = img_dir
        self.transform = transform if transform is not None else self.pipeline # if no transform is passed, use the default pipeline
        self.target_transform = target_transform
        self.normalizer = normalizer if normalizer is not None else normalizers.DistNormalizer(js.load(open(os.path.join(img_dir, "layer_dist.json")))) # if no normalizer is passed, use the default normalizer
        if is_train:
            self.test_set = self.cut_out(split = train_test_split) # cut out a test set

    def __len__(self):
        """
        Define dunder len, which returns the length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Define dunder getitem, which returns the item at the given index.

        Args:
            idx (int): The pytorch dataloader will call this function to get the item at the given index.

        Returns:
            tuple: The first element is the image, the second element is the label for training the network.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        image = np.load(img_path, allow_pickle=True)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = torch.tensor(np.float64([image[:,:,i] for i in range(image.shape[-1])]), device = "cuda")
        return image, label
    
    def pipeline(self, img):
        """
        Function which applies transformations to each of the images in this dataset. Preprocessing them for better training.

        Args:
            img (np.array): A single image as a 3d numpy array.

        Returns:
            np.array: A transformed image.
        """
        img = self.normalizer(img) # normalize the image layerwise
        if np.random.randn()>0: #50% chance
            img = transforms.horizontal_flip(img)
        if np.random.randn()>0: #50% chance
            img = transforms.vertical_flip(img)
        if np.random.randn()>0: #50% chance
            img = transforms.transpose(img)
        return img
    
    def cut_out(self, split = .5, delete_test_set = True, with_train=False):
        """
        This function cuts out a test set from the labels provided to 'img_labels'.

        Args:
            split (float, optional): The fraction of data remaining in the original dataset. Defaults to .5.
            delete_test_set (bool, optional): Boolean indicating wether to remorve the 'cut out' set form this dataset or not. Defaults to True.
            with_train (bool, optional): Boolean to indicate wether the 'cut out' should generate a test set from it self or not. Defaults to False.

        Returns:
            CustomImageDataset: A new CustomImageDataset instance containing only the random subset set.
        """
        total = len(self.img_labels)
        select = np.random.choice(np.arange(total), int(total*(1-split)))
        sub_set = CustomImageDataset(self.img_labels.iloc[select, :], img_dir=self.img_dir, transform=self.transform, target_transform=self.target_transform, is_train=with_train)
        if delete_test_set:
            self.img_labels = self.img_labels.iloc[~self.img_labels.index.isin(select), :]
        return sub_set
    
    def join(self, dataset):
        """
        Helper function to join two datasets.

        Args:
            dataset (CustomImageDataset): For instance to merge a cut out back into the entire dataset.

        Returns:
            CustomImageDataset: Instance of the dataset whose function 'join' was called.
        """
        self.img_labels = pd.concat([self.img_labels, dataset.img_labels], axis=0)
        return self
    
    def get_label_dataset(self, label, seed=0):
        """
        Create a new dataset where labels only make a binary classification between the given label and all other labels.

        Args:
            label (_type_): _description_
            seed (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        np.random.seed(seed)
        mask = self.img_labels[self.img_labels.columns[1]] == label
        right = self.img_labels[mask]
        false  = self.img_labels[~mask]
        total = len(right)
        random_select = np.random.choice(np.arange(len(false)),  total)
        joint = pd.concat([right, false.iloc[random_select, :]], axis=0)
        def new_target_transform(random_label):
            return torch.tensor(int(self.target_transform(random_label) == self.target_transform(label)))
        return CustomImageDataset(joint, img_dir=self.img_dir, transform=self.transform, target_transform=new_target_transform, is_train=False)
    
    @property
    def full_dataset(self):
        """
        Helper function to return the full dataset of train and test data.

        Returns:
            _type_: _description_
        """
        return self.join(self.test_set)
    
    @property
    def labelbalance(self):
        """
        Helper function to return the label balance of the dataset, indicating the number of samples of each label.

        Returns:
            _type_: _description_
        """
        x = self.img_labels.iloc[:, 1].value_counts()
        return x/x.sum()