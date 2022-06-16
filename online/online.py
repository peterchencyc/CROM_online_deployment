import argparse
from SamplePoint import *
import torch
from Encoder import *
from Decoder import *
from NonlinearSolver import *
from Experiments.ElasticityFem import *
from util.IOHelper import *
from ProjTypeScheduler import *
import os

from timer_cm import Timer

parser = argparse.ArgumentParser(
    description='Neural Representation evolving')
parser.add_argument('-md', help='path to the decoder',
                    type=str, nargs=1, required=True)
parser.add_argument('-mdg', help='path to the decoder with func and grad',
                    type=str, nargs=1, required=False)
parser.add_argument('-me', help='path to the encoder',
                    type=str, nargs=1, required=True)
parser.add_argument('-o', help='output path',
                    type=str, nargs=1, required=True)
parser.add_argument('-exp', help='output path',
                    type=str, nargs=1, required=True)
parser.add_argument('-config', help='config path',
                    type=str, nargs=1, required=False)
parser.add_argument('-ini_cond', help='initila condition',
                    type=str, nargs=1, required=False)
parser.add_argument('-proj_type', help='config path',
                    type=str, nargs='*', required=True)
parser.add_argument('-proj_steps', help='config path',
                    type=int, nargs='*', required=False)
parser.add_argument('-nonlinear_initial_guess', help='config path',
                    type=str, nargs='*', required=False)
parser.add_argument('-diff_threshold', help='',
                    type=float, nargs=1, required=False)
parser.add_argument('-step_size_threshold', help='',
                    type=float, nargs=1, required=False)
parser.add_argument('-nsteps', help='',
                    type=int, nargs=1, required=False)
parser.add_argument('-dt_div', help='',
                    type=int, nargs=1, required=False)
parser.add_argument('-debug_print', help='debug_print',
                    action='store_true')
parser.add_argument('-dx_fd', help='grid size for finite difference',
                    type=float, nargs=1, required=False)
parser.add_argument('-dis_or_pos', help='elasticty formulation dis or pos',
                    type=str, nargs=1, required=False)
parser.add_argument('-exact_ini', help='exact initial condition (currently coded for only elasticity fem)',
                    action='store_true')
parser.add_argument('-dx_fem', help='grid size for finite difference',
                    type=float, nargs=1, required=False)
parser.add_argument('-mesh_file', help='mesh_file',
                    type=str, nargs=1, required=True)
parser.add_argument('-remesh_file', help='remesh_file',
                    type=str, nargs='*', required=False)
parser.add_argument('-remesh_step', help='remesh_step',
                    type=int, nargs='*', required=False)
parser.add_argument('-vis_mesh_file', help='vis_mesh_file',
                    type=str, nargs=1, required=False)
parser.add_argument('-device', help='device',
                    type=str, nargs=1, required=False)
parser.add_argument('-num_sample_interior', help='',
                    type=int, nargs=1, required=False)
parser.add_argument('-num_sample_bdry', help='',
                    type=int, nargs=1, required=False)
parser.add_argument('-we', help='write_every',
                    type=int, nargs=1, required=False)
parser.add_argument('-write_sample', help='',
                    action='store_true')
parser.add_argument('-num_cpu_thread', help='',
                    type=int, nargs=1, required=False)
args = parser.parse_args()

md = args.md[0]
mdg = None if not args.mdg else args.mdg[0]
me = args.me[0]
output = args.o[0]
exp = args.exp[0]
if args.config:
    config = args.config[0]
if args.ini_cond:
    ini_cond = args.ini_cond[0]

dis_or_pos = 'pos' if not args.dis_or_pos else args.dis_or_pos[0]
proj_type = args.proj_type
proj_steps = [None] if not args.proj_steps else args.proj_steps
nonlinear_initial_guess = ['prev'] if not args.nonlinear_initial_guess else args.nonlinear_initial_guess
proj_type_scheduler = ProjTypeScheduler(proj_type, proj_steps, nonlinear_initial_guess)

diff_threshold = 1e-6 if args.diff_threshold is None else args.diff_threshold[0]
step_size_threshold = 1e-3 if args.step_size_threshold is None else args.step_size_threshold[0]

net_enc_jit_load = torch.jit.load(me)
net_dec_jit_load = torch.jit.load(md)
net_dec_grad_jit_load = None if mdg is None else torch.jit.load(mdg)

encoder = Encoder(net_enc_jit_load)
decoder = Decoder(net_dec_jit_load, md, net_dec_grad_jit_load)

device_str = 'cuda' if args.device is None else args.device[0] 
device = torch.device(device_str)

num_sample_interior = -1 if not args.num_sample_interior else args.num_sample_interior[0]
num_sample_bdry = -1 if not args.num_sample_bdry else args.num_sample_bdry[0]
write_every = 1 if not args.we else args.we[0]
num_cpu_thread = None if not args.num_cpu_thread else args.num_cpu_thread[0]

sample_str = 'sample-interior_{interior}_bdry_{bdry}'.format(interior = num_sample_interior, bdry = num_sample_bdry)

output = os.path.join(output,os.path.basename(os.path.dirname(md)), proj_type[0], device_str, sample_str, os.path.basename(os.path.dirname(config)),"h5_f_{:010d}.h5")

if exp == 'diffusion':
    problem = Diffusion(config, device)
elif exp == 'diffuse_image':
    problem = DiffuseImage(config, ini_cond, device)
elif exp == 'elasticity_fem':
    problem = ElasticityFem(config, ini_cond, dis_or_pos, device)
else:
    exit('invalid experiment')

if args.nsteps:
    problem.Nstep = args.nsteps[0]

if args.dt_div:
    problem.updatedt(args.dt_div[0])


problem.dx_fd = None if not args.dx_fd else args.dx_fd[0]
problem.dx_fem = None if not args.dx_fem else args.dx_fem[0]
problem.debug_print = True if args.debug_print else False
write_sample = True if args.write_sample else False
problem.remesh_step = None if not args.remesh_step else args.remesh_step
problem.remesh = False if problem.remesh_step == None else True
problem.exact_ini = True if args.exact_ini else False
problem.mesh_file = None if not args.mesh_file else args.mesh_file[0]
problem.remesh_file = None if not args.remesh_file else args.remesh_file

nonlinear_solver = NonlinearSolver(problem, decoder, diff_threshold, step_size_threshold)

q0_gt = problem.getPhysicalInitialCondition()
xhat = encoder.forward(q0_gt)
q0_gt = q0_gt.view(-1, q0_gt.size(2))
sample_style = 'random'
sample_point = SamplePoint(problem)
sample_point.initialize(sample_style, -1, -1, decoder, xhat, torch.zeros_like(xhat))
xhat = nonlinear_solver.solve(xhat, q0_gt, sample_point, 1, 20) # effectively serves as warm start for the rest of the gpu operations
print('initial xhat: ', xhat)
problem.updateState(xhat, decoder)
writeInitialLabel(xhat, md)

if args.vis_mesh_file: problem.vis_mesh_file = args.vis_mesh_file[0]

if  problem.__class__.__name__ == 'ElasticityFem':
    problem.initialSetup(xhat, decoder)

sample_point = SamplePoint(problem)
sample_point.initialize(sample_style, num_sample_interior, num_sample_bdry, decoder, xhat, torch.zeros_like(xhat))

# warm start
num_wm = 100
timer = Timer("warm start")
for i in range(num_wm):
    with timer.child('call 1'):
        problem.updateStateSampleAll(xhat, decoder, sample_point, torch.zeros_like(xhat))
for i in range(num_wm):
    with timer.child('call 2'):
        problem.updateStateSampleAll(xhat, decoder, sample_point, torch.zeros_like(xhat))
timer.print_results()

timer = Timer("online")
problem.timer = timer
nonlinear_solver.timer = timer
decoder.timer = timer

if num_cpu_thread: torch.set_num_threads(num_cpu_thread)

t = 0
step = 0
if not write_sample:
    problem.write2file(t, output.format(step), xhat)
else:
    problem.writeSample2file(sample_point, t, output.format(step), xhat)

for step in range(1, problem.Nstep+1):
    proj_type, nonlinear_initial_guess = proj_type_scheduler.getProjType(step)
    with timer.child('Residual'):
        res = problem.getResidualSample(xhat, decoder, sample_point, step)
    
    with timer.child('Projection'):
        xhat_prev = xhat.detach().clone()
        if proj_type == 'linear':
            with timer.child('Projection').child('linear solve'):
                jac = problem.getJacobianSampleProj(xhat, decoder, sample_point)
                vhat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac)).matmul(jac.transpose(1, 0)).matmul(res.view(-1, 1))
                vhat = vhat.view(1,1,-1)
                xhat += vhat*problem.dt
        elif proj_type == 'nonlinear':
            q_target = problem.q_sample.view_as(res) + res * problem.dt
            with timer.child('Projection').child('nonlinear solve'):
                if nonlinear_initial_guess == 'encoder':
                    q_target4enc = q_target.view(1, q_target.size(0), q_target.size(1))
                    xhat_initial = encoder.forward(q_target4enc)
                elif nonlinear_initial_guess == 'prev':
                    xhat_initial = xhat
                xhat = nonlinear_solver.solve(xhat_initial, q_target, sample_point, step_size = 1, max_iters = 10)
            with timer.child('Projection').child('vhat update'):
                vhat = None
                if  problem.__class__.__name__ == 'ElasticityFem':
                    jac = problem.jac_sample
                    vhat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac)).matmul(jac.transpose(1, 0)).matmul(res.view(-1, 1))
                    vhat = vhat.view(1,1,-1)
        else:
            exit('invalid proj_type')
        with timer.child('Projection').child('update state sample'):
            problem.updateStateSampleAll(xhat, decoder, sample_point, vhat)
            t += problem.dt

    with timer.child('updateWholeStateAndWrite'):
        if step%write_every==0:
            problem.updateState(xhat, decoder)
            if not write_sample:
                problem.write2file(t, output.format(step), xhat)
            else:
                problem.writeSample2file(sample_point, t, output.format(step), xhat)

    with timer.child('callBack'):
        remesh_flag = problem.callBack(step, xhat, decoder, vhat, xhat_prev)
        if remesh_flag:
            sample_point = SamplePoint(problem)
            sample_point.initialize(sample_style, num_sample_interior, num_sample_bdry, decoder, xhat, vhat)

timer.print_results()