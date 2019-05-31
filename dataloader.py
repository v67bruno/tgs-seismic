import torch, cv2
from torch.utils.data import Dataset
from random import randint, choice
import numpy as np
from pdb import set_trace as pause

class Load_Dataset(Dataset) :

    def __init__(self, X_img_pt, X_dpt_pt, Y_pt, only_true=True, data_augm=True):
        
        self.X_img = np.load(X_img_pt, mmap_mode='r')
        self.X_dpt = np.load(X_dpt_pt, mmap_mode='r')
        self.Y = np.load(Y_pt, mmap_mode='r')

        self.data_augm = data_augm
        self.only_true = only_true

        if self.data_augm or self.only_true :
            valid = []
            invalid = []

            for i in range(self.X_img.shape[0]):
                data_x = self.X_img[i]
                
                if data_x[data_x == 1].shape[0] == 0 :
                    invalid.append(i)

                else :
                    valid.append(i)
            
            self.valid = valid
            self.invalid = invalid

    def __getitem__(self, index):

        if self.data_augm :
            index_inv = self.invalid[index]

            index_val_r = randint(0, len(self.valid)-1)
            index_val = self.valid[index_val_r]

            X_img_inv = self.X_img[index_inv].copy()
            X_dpt_inv = self.X_dpt[index_inv].copy()
            Y_inv = self.Y[index_inv].copy()

            X_img_val = self.X_img[index_val].copy()
            X_dpt_val = self.X_dpt[index_val].copy()
            Y_val = self.Y[index_val].copy()

            if choice([True, False]) :
                X_img_inv = cv2.flip(X_img_inv, 0)
                X_dpt_inv = cv2.flip(X_dpt_inv, 0)
                Y_inv = cv2.flip(Y_inv, 0)
            
            if choice([True, False]) :
                X_img_inv = cv2.flip(X_img_inv, 1)
                X_dpt_inv = cv2.flip(X_dpt_inv, 1)
                Y_inv = cv2.flip(Y_inv, 1)

            X_img = X_img_inv.copy()
            X_dpt = X_dpt_inv.copy()
            Y = Y_inv.copy()

            data_salt = Y == 1

            if choice([True, False]) :
                X_img[data_salt] = X_img_val[data_salt]
                X_dpt[data_salt] = X_dpt_val[data_salt]
                Y[data_salt] = Y_val[data_salt]

            else :
                X_img = X_img_inv.copy()
                X_dpt = X_dpt_inv.copy()
                Y = Y_inv.copy()

            if choice([True, False]) :
                X_img = cv2.flip(X_img, 0)
                X_dpt = cv2.flip(X_dpt, 0)
                Y = cv2.flip(Y, 0)
            
            if choice([True, False]):
                X_img = cv2.flip(X_img, 1)
                X_dpt = cv2.flip(X_dpt, 1)
                Y = cv2.flip(Y, 1)

            angle = randint(-360, 360)

            X_img = self.rotate_bound(X_img, angle, 0)
            X_dpt = self.rotate_bound(X_dpt, angle, 0)
            Y = self.rotate_bound(Y, angle, 0)
                        
            X = np.stack((X_img, X_dpt), axis=0)
            X = torch.tensor(X)

            Y = torch.tensor(Y)
            
            return tuple((X, Y))

        else:

            if self.only_true :
                 index = self.valid[index]

            X_img = self.X_img[index].copy()
            X_dpt = self.X_dpt[index].copy()
            Y = self.Y[index].copy()

            X = np.stack((X_img, X_dpt), axis=0)
            X = torch.from_numpy(X)

            Y = torch.from_numpy(Y)
            
            return tuple((X, Y))

    def __len__(self) :
        if self.data_augm :
            return len(self.invalid)
        elif self.only_true :
            return len(self.valid)
        else :
            return self.X_img.shape[0]

    def rotate_bound(self, image, angle, bkg_val):
        rows, cols = image.shape[:2]

        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

        return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=bkg_val)

################################################################
################################################################
################################################################