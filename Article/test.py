import numpy
import datetime
import imageio
from skimage.transform import resize



def load_data(sample_g):
    imgs_A = []
    imgs_B = []
    # paired dataset path
    path = 'C:/Users/mirim/PycharmProjects/STST_replication/H_HG/test/' #used to be H_AG... fixed it?

    img = imageio.imread(path + '%d.tiff' % (sample_g + 1)).astype(numpy.float)
    h, w, _ = img.shape
    _w = int(w / 2)
    img_A, img_B = img[:, :_w, :], img[:, _w:, :]
    img_A = resize(img_A, img_res)
    img_B = resize(img_B, img_res)

    imgs_A.append(img_A)
    imgs_B.append(img_B)
    imgs_A = numpy.array(imgs_A) / 127.5 - 1.
    imgs_B = numpy.array(imgs_B) / 127.5 - 1.

    return imgs_A, imgs_B


