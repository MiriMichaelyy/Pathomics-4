import os
import shutil
from PIL import Image

def create_patches(image, length=256):
    width, height = image.size
    for row in range(0, height, length):
        for col in range(0, width, length):
            if col + length > width or row + length > height:
                continue
            yield image.crop((col, row, col + length, row + length))

def split_dataset(path, num_of_samples, train=0.60, test=0.20, val=0.20):
    train_samples = num_of_samples * train
    test_samples  = num_of_samples * test
    val_samples   = num_of_samples * val

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

def preprocess_image(image, dump, num_of_samples):
    with Image.open(image) as img:
        for index, crop_img in enumerate(create_patches(img)):

            # Save new patch.
            name = os.path.basename(os.path.splitext(image)[0])
            crop_img.save(os.path.join(dump, "color", f"{num_of_samples + index + 1}.tiff"))

            # Save new grayscale.
            gray_img = crop_img.convert('LA')
            gray_img.save(os.path.join(dump, "grayscale", f"{num_of_samples + index + 1}.tiff"))

            # Save new combined.
            crop_img.thumbnail(crop_img.size)
            gray_img.thumbnail(gray_img.size)

            w, h     = crop_img.size
            comb_img = Image.new("RGB", (w*2, h))
            comb_img.paste(crop_img, (0, 0, 1*w, h))
            comb_img.paste(gray_img, (w, 0, 2*w, h))
            comb_img.save(os.path.join(dump, "combined", f"{num_of_samples + index + 1}.tiff"))

    return index + 1

if __name__ == "__main__":
    path = os.path.abspath("../Datasets/Article/x20")
    dump = os.path.abspath("../Datasets/Article (Processed)/x20")
    if not os.path.exists(dump):
        os.makedirs(dump)
        os.makedirs(f"{dump}/color")
        os.makedirs(f"{dump}/grayscale")
        os.makedirs(f"{dump}/combined")

    num_of_samples = 0
    for img in os.listdir(path):
        full_path = os.path.join(path, img)
        if os.path.isfile(full_path):
            num_of_samples += preprocess_image(full_path, dump, num_of_samples)
    num_of_samples = len(os.listdir(f"{dump}/color"))
    split_dataset(dump, num_of_samples)