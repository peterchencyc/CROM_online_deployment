import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
rootrdir = os.path.dirname(currentdir)
commo_dir = os.path.join(rootrdir,'common')
sys.path.append(commo_dir)
online_dir = os.path.join(rootrdir,'online')
sys.path.append(online_dir)

import numpy as np
import json
import argparse
import torch
import time

from Encoder import *
from Decoder import *
from NonlinearSolver import *
from Experiments.DiffuseImage import *
from Experiments.Diffusion import *
from SimulationData import *



parser = argparse.ArgumentParser(
    description='Neural Representation evolving')
parser.add_argument('-md', help='path to the decoder',
                    type=str, nargs=1, required=True)
parser.add_argument('-me', help='path to the encoder',
                    type=str, nargs=1, required=True)
parser.add_argument('-exp', help='experiment',
                    type=str, nargs=1, required=True)
parser.add_argument('-diff_threshold', help='',
                    type=float, nargs=1, required=False)
parser.add_argument('-step_size_threshold', help='',
                    type=float, nargs=1, required=False)
parser.add_argument('-config', help='config path',
                    type=str, nargs=1, required=False)
parser.add_argument('-ini_cond', help='initila condition',
                    type=str, nargs=1, required=False)
parser.add_argument('-proj_type', help='projection type',
                    type=str, nargs='*', required=True)
parser.add_argument('-data_folder', help='path to the data',
                    type=str, nargs=1, required=True)
parser.add_argument('-device', help='device',
                    type=str, nargs=1, required=False)
args = parser.parse_args()

md = args.md[0]
me = args.me[0]
exp = args.exp[0]

if args.config:
    config = args.config[0]
if args.ini_cond:
    ini_cond = args.ini_cond[0]

net_enc_jit_load = torch.jit.load(me)
net_dec_jit_load = torch.jit.load(md)
net_dec_grad_jit_load = None

encoder = Encoder(net_enc_jit_load)
decoder = Decoder(net_dec_jit_load, md, net_dec_grad_jit_load)

diff_threshold = 1e-6 if args.diff_threshold is None else args.diff_threshold[0]
step_size_threshold = 1e-3 if args.step_size_threshold is None else args.step_size_threshold[0]

proj_type = args.proj_type[0]

folder_path = args.data_folder[0]

device_str = 'cuda' if args.device is None else args.device[0] 
device = torch.device(device_str)

def one_sim(st, param_path, sample_point):
    

    if exp == 'diffusion':
        problem = Diffusion(config, param_path, ini_cond, device)
    elif exp == 'diffuseimage':
        problem = DiffuseImage(param_path, ini_cond, device)
    nonlinear_solver = NonlinearSolver(problem, decoder, diff_threshold, step_size_threshold)

    q0_gt = problem.getPhysicalInitialCondition().float()
    xhat = encoder.forward(q0_gt)
    q0_gt = q0_gt.view(-1, q0_gt.size(2))
   
    
    problem.updateSampleIndicesWithProblemSpeficifInfo(sample_point)
    problem.jac_sample = problem.getJacobianSample(xhat, decoder, sample_point)
    problem.updateStateSampleAll(xhat, decoder, sample_point, torch.zeros_like(xhat))
    xhat = nonlinear_solver.solve(xhat, q0_gt[sample_point.indices], sample_point, 1, 20, print_info=False) # effectively serves as warm start for the rest of the gpu operations
    problem.updateState(xhat, decoder)
    problem.updateSampleIndicesWithProblemSpeficifInfo(sample_point)
    problem.jac_sample = problem.getJacobianSample(xhat, decoder, sample_point)
    problem.updateStateSample(xhat, decoder, sample_point)
    
    data_path = os.path.dirname(param_path)
    data_list = [d for d in sorted(os.listdir(data_path)) if d[-2:] == "h5" and d[:6] != "config"]
    state = []
    for step in range(problem.Nstep+1):
        state += [SimulationState(os.path.join(data_path, data_list[step]))]


    t = 0
    step = 0
    residual_0 = None
    for step in range(1, problem.Nstep+1):
        res = problem.getResidualSample(xhat, decoder, sample_point, step)
    
        xhat_prev = xhat.detach().clone()
        if proj_type == 'linear':
            jac = problem.getJacobianSampleProj(xhat, decoder, sample_point)
            vhat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac)).matmul(jac.transpose(1, 0)).matmul(res.view(-1, 1))
            vhat = vhat.view(1,1,-1)
            xhat += vhat*problem.dt
        elif proj_type == 'nonlinear':
            q_target = problem.q_sample.view_as(res) + res * problem.dt
            xhat_initial = xhat
            xhat = nonlinear_solver.solve(xhat_initial, q_target, sample_point, step_size = 1, max_iters = 10, print_info = False)
        else:
            exit('invalid proj_type')
        
        # Run every step
        if step % 1 == 0:
            q_now = problem.updateStateSampleAll(xhat, decoder, None)[0].cpu()
            q_actual = torch.tensor(state[step].q).view_as(q_now)
            if residual_0 is None:
                residual_0 = abs(q_actual - q_now).numpy()[:,0]
            else:
                residual_0 += abs(q_actual - q_now).numpy()[:,0]
        
        t += problem.dt

    q_now = problem.updateStateSampleAll(xhat, decoder, None)[0].cpu()
    q_actual = torch.tensor(state[step].q).view_as(q_now)
    residual_0 += abs(q_actual - q_now).numpy()[:,0]
    
    return residual_0

def all_sim(sample_point):
    
    folder_list = sorted(os.listdir(folder_path))
    json_list = []
    for folder in folder_list:
        for f in os.listdir(os.path.join(folder_path, folder)):
            if f[-4:] == "json":
                json_list += [os.path.join(folder_path, folder, f)]

    residual_total = None
    for st in range(len(folder_list)):
        residual_0 = one_sim(st, json_list[st], sample_point)
        
        if residual_total is None:
            residual_total = abs(residual_0)
        else:
            residual_total += abs(residual_0)
        #print(np.mean(residual_0))
    
    return residual_total

def cal_metric(x):
    return np.mean(x**2) + np.max(x**2)


ini_state = SimulationState(ini_cond)
n_points = ini_state.x.shape[0]

sample_style = 'random_'

if sample_style == 'random':
    select_index = [i for i in np.random.choice(np.arange(n_points), 1)]
elif sample_style == 'random_':
    if exp == 'diffusion':
        select_index = np.load("data/Diffusion/optimal_start.npy").tolist()
    elif exp == 'diffuseimage':
        select_index = np.load("data/Diffuse_Image/optimal_start.npy").tolist()
else:
    select_index = np.arange(n_points).tolist()

select_index = [i for i in select_index]

metric_min = 100000
total_test = 10

class SamplePoint:
    pass

sample_point = SamplePoint

start_time = time.time()

# empirical value
if exp == 'diffusion':
    metric_target = 4
elif exp == 'diffuseimage':
    metric_target = 150

while metric_min > metric_target:
    
    sample_point.indices = np.asarray(select_index)
    residual_total = all_sim(sample_point)
    metric = cal_metric(residual_total)
    print("#{}, Metric now {}, Time used {}s".format(len(select_index), metric, round(time.time()-start_time, 2)))
    
    if len(sample_point.indices) >= n_points:
        sys.exit("Already use all points. Exit.")    
    
    index_res = np.asarray([x[0] for x in sorted(enumerate(abs(residual_total)), key=lambda x: x[1], reverse=True)])
    
    metric_all = []
    metric_index = []
    num_test = 0
    for i in range(len(index_res)):
        if index_res[i] not in select_index:
            sample_point.indices = np.asarray(select_index + [index_res[i]])
            residual_total_test = all_sim(sample_point)
            metric = cal_metric(residual_total_test)
            metric_all += [metric]
            metric_index += [index_res[i]]
            print("Try index {}, Metric {}, Time used {}s".format(index_res[i], metric, round(time.time()-start_time, 2)))
            
            num_test += 1
            if num_test >= total_test:
                break
    in_index = np.argmin(metric_all)
    metric_min = metric_all[in_index]
    select_index += [metric_index[in_index]]
    print("Add index {}, Metric {}, Time used {}s".format(metric_index[in_index], metric_min, round(time.time()-start_time, 2)))

print("Optimal Selection:")
print(select_index)

if exp == 'diffusion':
    file_name = "data/Diffusion/optimal_selection_test.npy"
elif exp == 'diffuseimage':
    file_name = "data/Diffuse_Image/optimal_selection_test.npy"
np.save(file_name, np.asarray(select_index))
print("Save index to {}".format(file_name))
    