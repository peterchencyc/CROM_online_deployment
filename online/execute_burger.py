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

run_format = 'python online/online.py -md weights/Burger/epoch=3099-step=40300_dec{device_pt}.pt -me weights/Burger/epoch=3099-step=40300_enc{device_pt}.pt -o output -exp burger -f_path data/Burger/F.json -ini_cond data/Burger/h5_f_0000000000.h5 -proj_type nonlinear -diff_threshold 1e-15 -step_size_threshold 1e-15 -nsteps 200 -device {device} '

command = run_format.format(device = device, device_pt=device_pt)

os.system(command)


if device == 'cpu':
    device_pt = 'cpu'
elif device == 'cuda':
    device_pt = 'cuda'
else:
    exit('invalid device')
    
run_format = 'python common/draw_burger.py -d output/Burger/nonlinear/{device_pt}/full/Burger'

command = run_format.format(device_pt = device_pt)

os.system(command)