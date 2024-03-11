# Create test and train set with random patches
# run this third

import random
from PIL import Image

list = []

list = random.sample(range(12720), 12720)

k = 0
for i in range(12720):
    j = list[i]
    img = Image.open('C:/Users/mirim/PycharmProjects/STST_replication/H_HG/%d.tiff' % j)
    if i < 2999:
        img.save('C:/Users/mirim/PycharmProjects/STST_replication/train/%d.tiff' % (i + 1))
    else:
        img.save('C:/Users/mirim/PycharmProjects/STST_replication/test/%d.tiff' % (k + 1))
        k = k + 1
