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
import matplotlib.pyplot as plt
from PIL import Image
from SimulationData import *

parser = argparse.ArgumentParser(
    description='Draw Burger')
parser.add_argument('-d', help='Data path for Burger',
                    type=str, nargs=1, required=True)
args = parser.parse_args()

data_folder = args.d[0]


for f in sorted(os.listdir(data_folder)):
    if f[-2:] != "h5":
        continue
    data_path = os.path.join(data_folder, f)
    state = SimulationState(data_path)
    
    x = np.asarray(state.x)[:,0]
    q = np.asarray(state.q)[:,0]
    
 
    
    fig, ax = plt.subplots(figsize=(16, 12), dpi=50)
    ax.set_ylim(0, 4.25*1.3)
    im1, = ax.plot(x, q, "#ff7f00", linewidth=5)
    ax.set_xlabel('x', fontsize=28)
    ax.set_ylabel('u', fontsize=28)
    
    figure_path = os.path.join(data_folder, f[:-2] + "png")
    plt.savefig(figure_path)
    print("Save figure to {}".format(figure_path))
    
    plt.close()


frames = []
for f in sorted(os.listdir(data_folder)):
    if f[-3:] != "png":
        continue
    data_path = os.path.join(data_folder, f)
    new_frame = Image.open(data_path)
    frames.append(new_frame)
gif_path = os.path.join(data_folder, "1Burger.gif")
frames[0].save(gif_path, format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=40, loop=0)
print("Save GIF to {}".format(gif_path))




    