from keras.models import load_model
import scipy.misc
import numpy as np
import datetime
import imageio
from skimage.transform import resize



def load_data(sample_g):
    imgs_A = []
    imgs_B = []
    # paired dataset path
    path = 'C:/Users/mirim/PycharmProjects/STST_replication/H_HG/test/' #used to be H_AG... fixed it?

    img = imageio.imread(path + '%d.tiff' % (sample_g + 1)).astype(np.float)
    h, w, _ = img.shape
    _w = int(w / 2)
    img_A, img_B = img[:, :_w, :], img[:, _w:, :]
    #img_A = scipy.misc.imresize(img_A, img_res)
    #img_B = scipy.misc.imresize(img_B, img_res)
    img_A = resize(img_A, img_res)
    img_B = resize(img_B, img_res)

    imgs_A.append(img_A)
    imgs_B.append(img_B)
    imgs_A = np.array(imgs_A) / 127.5 - 1.
    imgs_B = np.array(imgs_B) / 127.5 - 1.

    return imgs_A, imgs_B


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img, patche):
    src_im = np.squeeze(np.array(src_img))
    gen_im = np.squeeze(np.array(gen_img))
    tar_im = np.squeeze(np.array(tar_img))

    path_image = 'C:/Users/mirim/PycharmProjects/STST_replication/results/'

    #scipy.misc.imsave(path_image + 'Input/%d.tiff' % (patche + 1), src_im)
    #scipy.misc.imsave(path_image + 'Generated/%d.tiff' % (patche + 1), gen_im)
    #scipy.misc.imsave(path_image + 'Original/%d.tiff' % (patche + 1), tar_im)

    imageio.imwrite(path_image + 'Input/%d.tiff' % (patche + 1), src_im)
    imageio.imwrite(path_image + 'Generated/%d.tiff' % (patche + 1), gen_im)
    imageio.imwrite(path_image + 'Original/%d.tiff' % (patche + 1), tar_im)
# __________________________________

img_rows = 256
img_cols = 256
channels = 3
img_res = (img_rows, img_cols)

# load model
model = load_model('C:/Users/mirim/PycharmProjects/STST_replication/models/model_15_2500.h5')
# load dataset
start_time = datetime.datetime.now()
for sample in range(500):
    [tar_image, src_image] = load_data(sample)
    print('Patche', sample + 1)
    # generate a batch of fake samples
    gen_image = model.predict(src_image)
    # plot all three images
    plot_images(src_image, gen_image, tar_image, sample)

elapsed_time = datetime.datetime.now() - start_time
print('time: ', elapsed_time)