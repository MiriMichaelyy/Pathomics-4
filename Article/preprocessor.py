import os
import shutil
from PIL import Image

def create_patches(image, size):
    img_width, img_height   = image.size
    width, height, channels = size
    for row in range(0, img_height, height):
        for col in range(0, img_width, width):
            if col + width > img_width or row + height > img_height:
                continue
            yield image.crop((col, row, col + width, row + height))

def split_dataset(path, num_of_samples, split):
    train_samples = num_of_samples * split[0]
    test_samples  = num_of_samples * split[1]
    val_samples   = num_of_samples * split[2]

    if not os.path.exists(f"{path}/train"):
        os.makedirs(f"{path}/train")
    if not os.path.exists(f"{path}/test"):
        os.makedirs(f"{path}/test")
    if not os.path.exists(f"{path}/val"):
        os.makedirs(f"{path}/val")

    for index, name in enumerate(os.listdir(f"{path}/color")):
        if index < train_samples:
            shutil.copyfile(f"{path}/color/{name}",     f"{path}/train/color_{name}")
            shutil.copyfile(f"{path}/grayscale/{name}", f"{path}/train/grayscale_{name}")
        elif index < train_samples + test_samples:
            shutil.copyfile(f"{path}/color/{name}",     f"{path}/test/color_{name}")
            shutil.copyfile(f"{path}/grayscale/{name}", f"{path}/test/grayscale_{name}")
        else:
            shutil.copyfile(f"{path}/color/{name}",     f"{path}/val/color_{name}")
            shutil.copyfile(f"{path}/grayscale/{name}", f"{path}/val/grayscale_{name}")

    return train_samples, test_samples, val_samples

def preprocess_image(dump, image, size, offset):
    with Image.open(image) as img:
        for index, crop_img in enumerate(create_patches(img, size)):

            # Save new patch.
            name = os.path.basename(os.path.splitext(image)[0])
            crop_img.save(os.path.join(dump, "color", f"{offset + index + 1}.tiff"))

            # Save new grayscale.
            gray_img = crop_img.convert('LA')
            gray_img.save(os.path.join(dump, "grayscale", f"{offset + index + 1}.tiff"))

            # Save new combined.
            crop_img.thumbnail(crop_img.size)
            gray_img.thumbnail(gray_img.size)

            w, h     = crop_img.size
            comb_img = Image.new("RGB", (w*2, h))
            comb_img.paste(crop_img, (0, 0, 1*w, h))
            comb_img.paste(gray_img, (w, 0, 2*w, h))
            comb_img.save(os.path.join(dump, "combined", f"{offset + index + 1}.tiff"))

    return index + 1

def preprocess(path, dump, split, size):
    if not os.path.exists(dump):
        os.makedirs(dump)
        os.makedirs(f"{dump}/color")
        os.makedirs(f"{dump}/grayscale")
        os.makedirs(f"{dump}/combined")

    sample_count = 0
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        if os.path.isfile(img_path):
            sample_count += preprocess_image(dump, img_path, size, sample_count)
    sample_count = len(os.listdir(f"{dump}/color"))
    return split_dataset(dump, sample_count, split)

if __name__ == "__main__":
    path  = os.path.abspath("../Datasets/Article/x20")
    dump  = os.path.abspath("../Datasets/Article_Processed/x20")
    split = (0.6, 0.2, 0.2)
    size  = (256, 256, 3)
    preprocess(path, dump, split, size)