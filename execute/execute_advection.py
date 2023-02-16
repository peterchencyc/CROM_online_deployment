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

run_format = 'python online/online.py -md weights/Advection/epoch=5999-step=24000_dec{device_pt}.pt -me weights/Advection/epoch=5999-step=24000_enc{device_pt}.pt -o output -exp advection -ini_cond data/Advection/h5_f_0000000000.h5 -proj_type nonlinear -diff_threshold 1e-15 -step_size_threshold 1e-15 -nsteps 60 -device {device} '

command = run_format.format(device = device, device_pt=device_pt)

os.system(command)


command = 'python common/draw_advection.py -d output/Advection'

os.system(command)