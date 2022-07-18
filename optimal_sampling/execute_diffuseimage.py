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

run_format = 'python optimal_sampling/run.py -exp diffuseimage -md weights/DiffuseImage/case2_dec{device_pt}.pt -me weights/DiffuseImage/case2_enc{device_pt}.pt -ini_cond data/DiffuseImage/h5_f_0000000000.h5 -data_folder optimal_sampling/training_data/DiffuseImage -proj_type nonlinear -diff_threshold 1e-15 -step_size_threshold 1e-15 -device {device}'

command = run_format.format(device = device, device_pt=device_pt)

os.system(command)