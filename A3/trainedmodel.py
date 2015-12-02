import scipy.io as spio
import numpy as np
from skimage import io
import time
def main():
    labeled_images_data = spio.loadmat("labeled_images.mat")
    identities = labeled_images_data.get("tr_identity")
    faces = labeled_images_data.get("tr_images")
    faces = faces.transpose(2, 0, 1)

    for i in range(len(faces)):
        io.imshow(faces[i])
        io.show()




    return


if __name__ == "__main__":
    main()