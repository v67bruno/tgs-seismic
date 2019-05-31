import os, shutil, csv
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as pause

# Data Loader
if False :

    p_t_img = './train/images'
    p_t_msk = './train/masks'
    csv_depth = './depths.csv'

    csv_file = open(csv_depth)
    csv_reader = csv.reader(csv_file, delimiter=',')
    dic_depth = {}

    with open(csv_depth) as csv_file :
        for i, row in enumerate(csv_reader) :
            if i > 0 :
                dic_depth[row[0]] = int(row[1])
                
    _, _, train_ids = next(os.walk(p_t_img))

    #data = ((1, 101, 101))
    x_s = np.array([], dtype=np.float32)
    y_s = np.array([], dtype=np.int64)
    x_d = np.array([], dtype=np.int64)

    #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    rgb2gray = lambda rgb : np.dot(rgb, [0.2989, 0.5870, 0.1140])

    for k, ids in enumerate(train_ids) :

        t_img = np.expand_dims(rgb2gray(plt.imread(p_t_img + '/' + ids)), axis=0)
        t_msk = np.expand_dims(plt.imread(p_t_msk + '/' + ids), axis=0)
        t_dpt = dic_depth[ids[:-4]]

        if k == 0 :
            x_s = t_img
            y_s = t_msk
            x_d = t_dpt

        if k > 0 :
            x_s = np.row_stack((x_s, t_img))
            y_s = np.row_stack((y_s, t_msk))
            x_d = np.row_stack((x_d, t_dpt))

    np.save('./train/X_img', x_s)
    np.save('./train/X_dpt', x_d)
    np.save('./train/Y_labels', y_s)

# Load Data
X_img = np.load('./train/X_img.npy') 
X_dpt = np.load('./train/X_dpt.npy')
Y_pt = np.load('./train/Y_labels.npy')

## Data Expl
#plt.hist(X_img.reshape(-1))
#plt.show()
#plt.hist(X_img[Y_pt == 1])
#plt.show()

# Data Norm
xper = np.percentile(X_img, 99)
xmax = X_img.max()
X_img /= xmax

X_dpt /= X_dpt.max()