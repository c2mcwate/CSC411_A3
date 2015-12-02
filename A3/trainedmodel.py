import scipy.io as spio
import numpy as np
from skimage import data

def main():
    test = spio.loadmat("unlabeled_images.mat").get('unlabeled_images')
    np.reshape(test,(-1,1,32,32))
    img = data.lena(test)


    return


if __name__ == "__main__":
    main()