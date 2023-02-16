import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
rootrdir = os.path.dirname(currentdir)
commo_dir = os.path.join(rootrdir,'common')
sys.path.append(commo_dir)

import numpy as np
import json
import argparse
import cv2
from PIL import Image
from SimulationData import *


parser = argparse.ArgumentParser(
    description='Draw Diffuse Image')
parser.add_argument('-d', help='Data path for Diffuse Image',
                    type=str, nargs=1, required=True)
args = parser.parse_args()

data_folder = args.d[0]


for f in sorted(os.listdir(data_folder)):
    if f[-2:] != "h5":
        continue
    data_path = os.path.join(data_folder, f)
    state = SimulationState(data_path)
    
    x = np.asarray(state.x)[:,0].reshape(256, 256)
    q = np.asarray(state.q)[:,0].reshape(256, 256)
    q[q>1] = 1
    q[q<0] = 0
    img = np.asarray(q * 255, dtype = np.uint8)
    figure_path = os.path.join(data_folder, f[:-2] + "png")
    cv2.imwrite(figure_path, img)
    print("Save figure to {}".format(figure_path))

frames = []
for f in sorted(os.listdir(data_folder)):
    if f[-3:] != "png":
        continue
    data_path = os.path.join(data_folder, f)
    new_frame = Image.open(data_path)
    frames.append(new_frame)
gif_path = os.path.join(data_folder, "1DiffuseImage.gif")
frames[0].save(gif_path, format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=0)
print("Save GIF to {}".format(gif_path))