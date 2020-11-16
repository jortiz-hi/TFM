import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir', type=str, default='./')
args = parser.parse_args()

vid = args.frames_dir
filename = os.listdir(vid)
try:
    fld = vid + '_png'
    if not os.path.exists(fld):
        os.makedirs(fld)
except OSError:
    print(r'Error: Creating directory of data')
for i in filename:
    img1 = vid + '\\' + i
    im1 = Image.open(img1)
    im1.save(fld + '\\' + i.replace('.pgm', '_fmt.png'))