import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda',
                    type=str)
args = parser.parse_args()

device = args.device
if device == 'cpu':
    device_pt = '_cpu'
elif device == 'cuda':
    device_pt = '' 
else:
    exit('invalid device')

run_format = 'python online/online.py -md weights/Diffuse_Image/case2_dec{device_pt}.pt -me weights/Diffuse_Image/case2_enc{device_pt}.pt -o output -exp diffuse_image -config data/Diffuse_Image/F_3_00.json -ini_cond data/Diffuse_Image/h5_f_0000000000.h5 -proj_type nonlinear -diff_threshold 1e-10 -step_size_threshold 1e-10 -nsteps 20 -device {device} '

command = run_format.format(device = device, device_pt=device_pt)

os.system(command)

    
command = 'python common/draw_image.py -d output/Diffuse_Image'

os.system(command)