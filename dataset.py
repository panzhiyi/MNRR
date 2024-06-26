import os
import os.path
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

from IPython.core import debugger
debug = debugger.Pdb().set_trace


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def load_image_label_list_from_npy(img_name_list, data_root):
    cls_labels_dict = np.load('{}/cls_labels.npy'.format(data_root), allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

class Sem_ContourData(Dataset):
    def __init__(self, split, data , data_root=None, data_list=None, transform=None , path_lab='gtFine_trans',path_contour='contour',path_adj='AdjMatrix'):
        self.split = split
        if data=='VOC2012' or data=='SBD':
            self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
            self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
            self.lab_names = ['{}/{}/{}.png'.format(data_root, path_lab, i) for i in self.indices]
            self.contour_names = ['{}/{}/{}.png'.format(data_root, path_contour, i) for i in self.indices]
            #self.adj_names=['{}/{}/{}.png'.format(data_root, path_adj, i) for i in self.indices]
        elif data=='cityscapes':
            self.indices = open('{}/{}'.format(data_root, data_list),'r').read().splitlines()
            self.img_names = ['{}/leftImg8bit/{}/{}_leftImg8bit.png'.format(data_root, split, i) for i in self.indices]
            self.lab_names = ['{}/{}/{}/{}.png'.format(data_root, path_lab, split, i) for i in self.indices]
            self.contour_names = ['{}/{}/{}/{}_gtFine_instanceIds.png'.format(data_root, path_contour, split, i) for i in self.indices]
            #self.adj_names=['{}/{}/{}/{}.png'.format(data_root, path_adj, split, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        contour_path=self.contour_names[index]
        #adj_path=self.adj_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        contour = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != contour.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label, contour = self.transform(image, label, contour)
        '''
        if self.split == 'train':
            adj_matrix = cv2.imread(adj_path, cv2.IMREAD_GRAYSCALE)
            adj_matrix = torch.from_numpy(adj_matrix)
            if not isinstance(adj_matrix, torch.LongTensor):
                adj_matrix = adj_matrix.long()
            return image, label, contour, adj_matrix, image_path
        else:
        '''
        return image, label, contour, image_path

class SemData(Dataset):
    def __init__(self, split, data, data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        if data=='VOC2012' or data=='SBD':
            self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
            self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
            self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        elif data=='cityscapes':
            self.indices = open('{}/{}'.format(data_root, data_list),'r').read().splitlines()
            self.img_names = ['{}/leftImg8bit/{}/{}_leftImg8bit.png'.format(data_root, split, i) for i in self.indices]
            self.lab_names = ['{}/gtFine_trans/{}/{}_gtFine_labelIds.png'.format(data_root, split, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, image_path
        
        
class ClsData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = [i for i in self.indices]
        self.lab_names2 = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.label_list = load_image_label_list_from_npy(self.lab_names, data_root)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path = self.img_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = self.label_list[index]
        label_path=self.lab_names2[index]
        label2 = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            image,label2 = self.transform(image,label2)
        return image, label, label2, image_path
