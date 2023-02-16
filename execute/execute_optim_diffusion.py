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

run_format = 'python optimal_sampling/run.py -exp diffusion -md weights/Diffusion/case3_dec{device_pt}.pt -me weights/Diffusion/case3_enc{device_pt}.pt -config data/Diffusion/config.h5 -ini_cond data/Diffusion/h5_f_0000000000.h5 -data_folder data/Optimal_Sampling/Diffusion -proj_type nonlinear -diff_threshold 1e-15 -step_size_threshold 1e-15 -device {device}'

command = run_format.format(device = device, device_pt=device_pt)

os.system(command)