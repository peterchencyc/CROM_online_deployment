import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-device',
                    type=str, required=True)
args = parser.parse_args()

device = args.device
if device == 'cpu':
    device_pt = '_cpu'
elif device == 'cuda':
    device_pt = '' 
else:
    exit('invalid device')

run_format = 'python3 online/online.py -md weights/20220510-081900/epoch=1599-step=91199_dec{device_pt}.pt -mdg weights/20220510-081900/epoch=1599-step=91199_dec_func_grad{device_pt}.pt -me weights/20220510-081900/epoch=1599-step=91199_enc{device_pt}.pt -o output -exp elasticity_fem -config data/sim_seq_data_mu=6250.0_lamb=62500.0/config.h5 -ini_cond data/sim_seq_data_mu=6250.0_lamb=62500.0/h5_f_0000000000.h5 -proj_type linear -diff_threshold 1e-10 -step_size_threshold 1e-10 -dt_div 1 -nsteps 100 -dis_or_pos dis -mesh_file data/tetWild/pig_long_l-0.01.h5 -device {device} -num_sample_interior 20 -num_sample_bdry 20 -we 100'

command = run_format.format(device = device, device_pt=device_pt)

os.system(command)