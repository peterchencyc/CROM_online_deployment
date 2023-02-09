import os
from PIL import Image
import argparse



def frames2gif(src_dir, save_path, fps=40):
    print("Convert frames to gif...")
    filenames = sorted([x for x in os.listdir(src_dir) if x.endswith('.png')])
    img_list = [Image.open(os.path.join(src_dir, name)) for name in filenames]
    img = img_list[0]
    img.save(fp=save_path, append_images=img_list[1:],
            save_all=True, duration=1 / fps * 1000, loop=0)
    print("Done.")
    
    
parser = argparse.ArgumentParser(
    description='PNG to GIF')
parser.add_argument('-folder', help='folder',
                    type=str, required=True) 
parser.add_argument('-fps', help='fps', default=40,
                    type=int, required=False)                
                    
args = parser.parse_args()
                    
frames2gif(args.folder, os.path.join(args.folder, "1Karman.gif"), args.fps)
                    
