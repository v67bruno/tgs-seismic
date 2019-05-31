import cv2, torch
import numpy as np
from torchvision import models
from torch.utils.data import Dataset
from random import randint, choice
from pdb import set_trace as pause

class SARDataset(Dataset) :

    def __init__(self, VV_path, VH_path, WIND_path, OIL_path, LAND_path, ND_path, only_with_spill=True, data_augm=True, spill_area_thresh=1000):
        
        self.VV = np.load(VV_path, mmap_mode='r')
        self.VH = np.load(VH_path, mmap_mode='r')
        self.WIND = np.load(WIND_path, mmap_mode='r')
        self.OIL = np.load(OIL_path, mmap_mode='r')
        self.LAND = np.load(LAND_path, mmap_mode='r')
        self.ND = np.load(ND_path, mmap_mode='r')
        
        self.data_augm = data_augm
        self.only_with_spill = only_with_spill

        if self.data_augm or self.only_with_spill:
            valid = []
            invalid = []
            for i in range(self.OIL.shape[0]):
                gt = self.OIL[i]

                if gt[gt == 1].shape[0] > spill_area_thresh :
                    valid.append(i)

                elif gt[gt == 1].shape[0] == 0 :
                    invalid.append(i)
            
            self.valid = valid
            self.invalid = invalid
            #print("valid", len(valid))
        # pause()

    def __getitem__(self, index):

        if self.data_augm :
            index_inv = self.invalid[index]

            index_val_r = randint(0, len(self.valid)-1)
            index_val = self.valid[index_val_r]

            VV_inv = self.VV[index_inv].copy()
            VH_inv = self.VH[index_inv].copy()
            WIND_inv = self.WIND[index_inv].copy()
            OIL_inv = self.OIL[index_inv].copy()
            LAND_inv = self.LAND[index_inv].copy()
            ND_inv = self.ND[index_inv].copy()

            VV_val = self.VV[index_val].copy()
            VH_val = self.VH[index_val].copy()
            WIND_val = self.WIND[index_val].copy()
            OIL_val = self.OIL[index_val].copy()
            LAND_val = self.LAND[index_val].copy()
            ND_val = self.ND[index_val].copy()


            if choice([True, False]) :
                VV_inv = cv2.flip(VV_inv, 0)
                VH_inv = cv2.flip(VH_inv, 0)
                WIND_inv = cv2.flip(WIND_inv, 0)
                OIL_inv = cv2.flip(OIL_inv, 0)
                LAND_inv = cv2.flip(LAND_inv, 0)
                ND_inv = cv2.flip(ND_inv, 0)
            
            if choice([True, False]) :
                VV_inv = cv2.flip(VV_inv, 1)
                VH_inv = cv2.flip(VH_inv, 1)
                WIND_inv = cv2.flip(WIND_inv, 1)
                OIL_inv = cv2.flip(OIL_inv, 1)
                LAND_inv = cv2.flip(LAND_inv, 1)
                ND_inv = cv2.flip(ND_inv, 1)

            VV = VV_inv.copy()
            VH = VH_inv.copy()
            WIND = WIND_inv.copy()
            OIL = OIL_inv.copy()
            LAND = LAND_inv.copy()
            ND = ND_inv.copy()


            gt_spill = OIL_val == 1
            gt_land = LAND == 2
            #pause()

            if choice([True, False]) :
            #if gt_land < 1.5 :
                VV[gt_spill] = VV_val[gt_spill]
                VV[gt_land] = VV_inv[gt_land]

                VH[gt_spill] = VH_val[gt_spill]
                VH[gt_land] = VH_inv[gt_land]

                WIND[gt_spill] = WIND_val[gt_spill]
                WIND[gt_land] = WIND_inv[gt_land]

                OIL[gt_spill] = OIL_val[gt_spill]
                OIL[gt_land] = OIL_inv[gt_land]

                LAND[gt_spill] = LAND_val[gt_spill]
                LAND[gt_land] = LAND_inv[gt_land]

                ND[gt_spill] = ND_val[gt_spill]
                ND[gt_land] = ND_inv[gt_land]

            else :
                VV = VV_val.copy()
                VH = VH_val.copy()
                WIND = WIND_val.copy()
                OIL = OIL_val.copy()
                LAND = LAND_val.copy()
                ND = ND_val.copy()

            if choice([True, False]) :
                VV = cv2.flip(VV, 0)
                VH = cv2.flip(VH, 0)
                WIND = cv2.flip(WIND, 0)
                OIL = cv2.flip(OIL, 0)
                LAND = cv2.flip(LAND, 0)
                ND = cv2.flip(ND, 0)
            
            if choice([True, False]):
                VV = cv2.flip(VV, 1)
                VH = cv2.flip(VH, 1)
                WIND = cv2.flip(WIND, 1)
                OIL = cv2.flip(OIL, 1)
                LAND = cv2.flip(LAND, 1)
                ND = cv2.flip(ND, 1)

            angle = randint(-360, 360)

            VV = self.rotate_bound(VV, angle, 0)
            VH = self.rotate_bound(VH, angle, 0)
            WIND = self.rotate_bound(WIND, angle, 0)
            OIL = self.rotate_bound(OIL, angle, 3)
            LAND = self.rotate_bound(LAND, angle, 3)
            ND = self.rotate_bound(ND, angle, 3)
            
            X = np.stack((VV, VH, WIND), axis=0)
            X = torch.tensor(X)
            
            ND[OIL == 1] = 1
            ND[LAND == 2] = 2
            Y = torch.tensor(ND.astype('int64'))
            
            return tuple((X, Y))

        else:

            if self.only_with_spill:
                 index = self.valid[index]

            VV = self.VV[index].copy()
            VH = self.VH[index].copy()
            WIND = self.WIND[index].copy()
            OIL = self.OIL[index].copy()
            LAND = self.LAND[index].copy()
            ND = self.ND[index].copy()

            #VV = np.expand_dims(VV, axis=0)
            #VH = np.expand_dims(VH, axis=0)

            X = np.stack((VV, VH, WIND), axis=0)
            X = torch.from_numpy(X)
            
            ND[OIL == 1] = 1
            ND[LAND == 2] = 2
            Y = torch.from_numpy(ND)
            
            return tuple((X, Y))

    def __len__(self):
        if self.data_augm:
            return len(self.invalid)
        elif self.only_with_spill:
            return len(self.valid)
        else:
            return self.VV.shape[0]

    def rotate_bound(self, image, angle, bkg_val):
        rows,cols = image.shape[:2]

        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

        return cv2.warpAffine(image, M, (cols,rows), flags=cv2.INTER_NEAREST, borderValue= bkg_val)
################################################################
################################################################
################################################################