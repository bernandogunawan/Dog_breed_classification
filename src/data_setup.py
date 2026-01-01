"""
Contains functionality for creating PyTorch Dataloaders for image classification data.
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
import matplotlib.image as mpimg
from PIL import Image
import os

class CostumDataset(Dataset):
    def __init__(self, path, pair, transform = None):
        self.image_path = path + "/images/Images/"
        self.pair = pair
        self.transform = transform

    def __len__(self):
        return len(self.pair)

    def __getitem__(self,index):
        img_path = os.path.join(self.image_path,self.pair[index][0])
        img = mpimg.imread(img_path)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.pair[index][1]

    def load_image(self, index):
        img_path = os.path.join(self.image_path,self.pair[index][0])
        return Image.open(img_path)

    def get_class(self):
        num = 0
        class_name = []
        for x,y in pair:
            if y == num:
                class_name.append(x.split("-",1)[1].split("/",1)[0])
                num += 1
        return class_name

def load_data(path):

    # get image and file path
    images_path = path + "/images/Images/"
    lists_path = path + "/lists/"

    # load train and test file
    train_file = loadmat(lists_path + "train_list.mat")
    test_file = loadmat(lists_path + "test_list.mat")

    # get train and test data to a list
    file_train_list = [x[0] for x in train_file["file_list"].squeeze()]
    label_train_list = train_file["labels"].squeeze()-1

    file_test_list = [x[0] for x in test_file["file_list"].squeeze()]
    label_test_list = test_file["labels"].squeeze()-1

    return list(zip(file_train_list,label_train_list)),list(zip(file_test_list,label_test_list))

def filter_list(path, pair, image_size):
    temp_list = []
    for x,y in pair:
        temp_path = path + x
        img = mpimg.imread(temp_path)
        if img.shape[0] >= image_size and img.shape[1] >= image_size:
            temp_list.append((x,y))

    return temp_list


def create_dataloader(path, transform, image_size, batch_size):
    train_list,test_list = load_data(path)

    train_list = filter_list(path, train_list, image_size)
    test_list = filter_list(path, test_list, image_size)

    train_dataset = CostumDataset(path,train_list,transform)
    test_dataset = CostumDataset(path,test_list,transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    class_name = train_dataset.get_class()

    return train_dataloader, test_dataloader, class_name
