import numpy
from keras.models import load_model
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


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img, patche):
    src_im = numpy.squeeze(numpy.array(src_img))
    gen_im = numpy.squeeze(numpy.array(gen_img))
    tar_im = numpy.squeeze(numpy.array(tar_img))

    path_image = 'C:/Users/mirim/PycharmProjects/STST_replication/results/'

    imageio.imwrite(path_image + 'Input/%d.tiff' % (patche + 1), src_im)
    imageio.imwrite(path_image + 'Generated/%d.tiff' % (patche + 1), gen_im)
    imageio.imwrite(path_image + 'Original/%d.tiff' % (patche + 1), tar_im)
# __________________________________

def test(model, dataset, img_shape):
    start_time = datetime.datetime.now()
    for index, (tar_image, src_image) in enumerate(dataset):
        print(f"Testing image #{index+1}")
        gen_image = model.predict(src_image)
        # plot_images(src_image, gen_image, tar_image, sample)
    print('time: ', datetime.datetime.now() - start_time)