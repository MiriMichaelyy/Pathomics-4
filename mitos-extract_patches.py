# Extract patches 256*256 from each dataset image
#run this first

from __future__ import print_function
import os

from PIL import Image

height = 256
width = 256

imgwidth = 1536
imgheight = 1280

# After each run, change the num value to the last number in the Save folder
num = 12421
# After a run for each folder, change the range to the number of images in the desired folder 
for k in range(10):
    # After each run, change the A, B, C, D character in the path
    img = Image.open('C:/Users/mirim/PycharmProjects/STST_replication/H16/frames/x20/H16_0%dD.tiff' % (k))
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)
            patch = img.crop(box)
            patch.save(os.path.join('C:/Users/mirim/PycharmProjects/STST_replication/H/%d.tiff' % (num)))
            num = num + 1


#file path for miri's project:
#"C:/Users/mirim/PycharmProjects/STST_replication"












