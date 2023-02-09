import sys
sys.path.append("online/")
sys.path.append("offline/process_data/")

import numpy as np
import json
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import time
from timeit import default_timer as timer

from Decoder_Karman import *
from Encoder import *
from NonlinearSolver import *
from Experiments.Karman import *

import matplotlib.cm as cm
from scipy.special import erf
from PIL import Image

from functools import partial
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import factorized, spsolve
from scipy import sparse

import argparse

from tqdm import tqdm

KARMAN_VEL = 0.5

def karman_velocity(coords: np.ndarray, init_vel: float=KARMAN_VEL):
    vel = np.zeros_like(coords)
    vel[..., 1] = init_vel # constant horizontal velocity
    return vel


def sphere_obstacle(coords: np.ndarray):
    center = np.array([np.pi / 2, np.pi / 4])
    radius = np.pi / 15
    center = center.reshape(*[1]*len(coords.shape[:-1]), 2)
    sign_dist = np.linalg.norm(coords - center, 2, axis=-1) - radius
    return sign_dist


def karman_boundary(d_grid, u_grid, v_grid, h):
    """d, u, v from stagger grid. h: grid size"""
    grid_indices = np.indices(d_grid.shape)

    def _transform_coords(coords, offset):
        """transform coords in grid space to original domain"""
        minxy = np.array([0, 0])[None, None]
        offset = np.array(offset)[None, None]
        coords = coords.transpose((1, 2, 0)).astype(float) + offset
        coords = coords * h + minxy
        return coords

    # solid sphere obstacle
    u_coords = _transform_coords(grid_indices[:, :, :-1], [-0.5, 0])
    v_coords = _transform_coords(grid_indices[:, :-1], [0, -0.5])
    mask_u = sphere_obstacle(u_coords) < 0
    mask_v = sphere_obstacle(v_coords) < 0
    u_grid[mask_u] = 0
    v_grid[mask_v] = 0

    # domain boundary: open right side
    u_grid[0, :] = 0
    u_grid[-1, :] = 0
    u_grid[:, 0] = KARMAN_VEL

    v_grid[0, :] = 0
    v_grid[-1, :] = 0
    v_grid[:, 0] = 0
    v_grid[:, -1] = 0

    d_grid[0, :] = 0
    d_grid[-1, :] = 0
    d_grid[:, 0] = 0
    d_grid[:, -1] = 0
    

def build_laplacian_matrix(M: int, N: int):
    """build laplacian matrix with neumann boundary condition,
    i.e., gradient at boundary equals zero.
    This eliminates one degree of freedom.

    Args:
        M (int): number of rows
        N (int): number of cols

    Returns:
        np.array: laplacian matrix 
    """
    main_diag = np.full(M * N, -4)
    main_diag[[0, N - 1, -N, -1]] = -2
    main_diag[[*range(1, N - 1), *range(-N + 1, -1), 
               *range(N, (M - 2) * N + 1, N), *range(2 * N - 1, (M - 1) * N, N)]] = -3
    side_diag = np.ones(M * N - 1)
    side_diag[[*range(N - 1, M * N - 1, N)]] = 0
    data = [np.ones(M * N - N), side_diag, main_diag, side_diag, np.ones(M * N - N)]
    offsets = [-N, -1, 0, 1, N]
    mat = sparse.diags(data, offsets)
    return mat
    
    
class StableFluids(object):
    def __init__(self, N: int, dt: float, domain: list=[[0, 1], [0, 1]], visc: float=0, diff: float=0, boundary_func=None):

        """Stable Fluids solver with stagger grid discretization.
        Density(dye) and pressure values are stored at the center of grids.
        Horizontal (u) and vertical (v) velocity values are stored at edges.
        The velocity along (i, j) indexing directions are (v, u).
        A layer of boundary is warpped outside.
        TODO: support different pressure solvers
        TODO: support external force
        TODO: support arbitrary rho. now assume rho=1 everywhere.

        ---v-----v---
        |     |     |
        u  d  u  d  u
        |     |     |
        ---v-----v---
        |     |     |
        u  d  u  d  u
        |     |     |
        ---v-----v---

        Args:
            N (int): grid resolution along the longest dimension.
            dt (float): timestep size
            domain (list, optional): 2D domain ([[x_min, x_max], [y_min, y_max]]). 
                Defaults to [[0, 1], [0, 1]].
            visc (float, optional): viscosity coefficient. Defaults to 0.
            diff (float, optional): diffusion coefficient. Defaults to 0.
            boundary_func (function): function to set boundary condition, 
                func(d_grid, u_grid, v_grid) -> None. Defaults to None, using solid boundary.
        """
        len_x = (domain[0][1] - domain[0][0])
        len_y = (domain[1][1] - domain[1][0])
        self.h = max(len_x, len_y) / N
        print("h", self.h)
        self.M = int(len_x / self.h)
        self.N = int(len_y / self.h)

        self.dt = dt
        self.visc = visc
        self.diff = diff
        self.domain = domain
        self.timestep = 0
        
        self._d_grid = np.zeros((self.M + 2, self.N + 2))
        self._u_grid = np.zeros((self.M + 2, self.N + 1))
        self._v_grid = np.zeros((self.M + 1, self.N + 2))

        # grid coordinates
        self._grid_indices = np.indices(self._d_grid.shape)

        # interpolation function
        self.interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

        # boundary condition function
        if boundary_func is None: # assume solid boundary by default
            self.boundary_func = self._set_solid_boundary
        else:
            self.boundary_func = boundary_func
            
        print(self.M,self.N)

        # linear system solver
        print("Build pre-factorized linear system solver. Could take a while.")
        self.lap_mat = build_laplacian_matrix(self.M, self.N) 
        self.pressure_solver = factorized(self.lap_mat)    

        if self.diff > 0:
            self.diffD_solver = factorized(np.identity(self.M * self.N) - 
                diff * dt / self.h / self.h * build_laplacian_matrix(self.M, self.N))
        if self.visc > 0:
            self.diffU_solver = factorized(np.identity(self.M * (self.N + 1)) - 
                visc * dt / self.h / self.h * build_laplacian_matrix(self.M, self.N + 1))
            self.diffV_solver = factorized(np.identity((self.M + 1) * self.N) - 
                visc * dt / self.h / self.h * build_laplacian_matrix(self.M + 1, self.N))

    @property
    def grid_density(self):
        """density values at grid centers"""
        return self._d_grid[1:-1, 1:-1]
    
    @property
    def grid_velocity(self):
        """velocity values at grid centers"""
        u = (self._u_grid[1:-1, 1:] + self._u_grid[1:-1, :-1]) / 2
        v = (self._v_grid[1:, 1:-1] + self._v_grid[:-1, 1:-1]) / 2
        vel = np.stack([v, u], axis=-1)
        return vel
    
    @property
    def grid_curl(self):
        """curl(vorticity) values at grid centers"""
        curl = compute_curl(self.grid_velocity, self.h)
        return curl

    def _transform_coords(self, coords, offset):
        """transform coords in grid space to original domain"""
        minxy = np.array([self.domain[0][0], self.domain[1][0]])[None, None]
        offset = np.array(offset)[None, None]
        coords = coords.transpose((1, 2, 0)).astype(float) + offset
        coords = coords * self.h + minxy
        return coords

    def add_source(self, attr: str, source_func):
        """add source to density(d) or velocity field(u, v)

        Args:
            attr (str): "velocity" or "density"
            source_func (function): attr(x) = source_func(x)

        Raises:
            ValueError: _description_
        """
        if source_func is None:
            return

        if attr == "velocity":
            u_indices = self._transform_coords(self._grid_indices[:, :, :-1], [-0.5, 0])
            self._u_grid += source_func(u_indices)[..., 1]

            v_indices = self._transform_coords(self._grid_indices[:, :-1], [0, -0.5])
            self._v_grid += source_func(v_indices)[..., 0]
        elif attr == "density":
            d_indices = self._transform_coords(self._grid_indices, [-0.5, -0.5])
            self._d_grid += source_func(d_indices)
        else:
            raise ValueError(f"attr must be velocity or density, but got {attr}.")
        self.boundary_func(self._d_grid, self._u_grid, self._v_grid, self.h)

    def step(self):
        """Integrates the system forward in time by dt."""
        since = time.time()
        
        self._velocity_step()
        self._density_step()
        self.timestep += 1
        
        timecost = time.time() - since
        return timecost
    
    def _density_step(self):
        """update density field by one timestep"""
        # diffusion
        if self.diff > 0:
            self._diffuseD()

        # advection
        self._advectD()

    def _velocity_step(self):
        """update density field by one timestep"""
        # external force
        pass

        # advection
        self._advectVel()

        # diffusion
        if self.visc > 0:
            self._diffuseU()
            self._diffuseV()

        # projection
        self._project()

    def _diffuseD(self):
        """diffusion step for d ([1, M], [1, N]) using implicit method"""
        self._d_grid[1:-1, 1:-1] = self.diffD_solver(self._d_grid[1:-1, 1:-1].flatten()).reshape(self.M, self.N)

    def _diffuseU(self):
        """diffusion step for u ([1, M], [0, N]) using implicit method"""
        self._u_grid[1:-1, :] = self.diffU_solver(self._u_grid[1:-1].flatten()).reshape(self.M, self.N + 1)

    def _diffuseV(self):
        """diffusion step for v ([0, M], [1, N]) using implicit method"""
        self._v_grid[:, 1:-1] = self.diffV_solver(self._v_grid[:, 1:-1].flatten()).reshape(self.M + 1, self.N)

    def _advectD(self):
        """advect density for ([1, M], [1, N])"""
        i_back = self._grid_indices[0, 1:-1, 1:-1] - self.dt / self.h * (
            self._v_grid[:-1, 1:-1] + self._v_grid[1:, 1:-1]) / 2
        j_back = self._grid_indices[1, 1:-1, 1:-1] - self.dt / self.h * (
            self._u_grid[1:-1, :-1] + self._u_grid[1:-1, 1:]) / 2
        self._d_grid[1:-1, 1:-1] = self.interpolate(self._d_grid, np.stack([i_back, j_back]))

    def _advectVel(self):
        """addvect velocity field"""
        new_u_grid = self._advectU()
        new_v_grid = self._advectV()
        self._u_grid[1:-1, :] = new_u_grid
        self._v_grid[:, 1:-1] = new_v_grid

    def _advectU(self):
        """advect horizontal velocity (u) for ([1, M], [0, N])"""
        i_back = self._grid_indices[0, 1:-1, :-1] - self.dt / self.h * (
            self._v_grid[:-1, :-1] + self._v_grid[1:, :-1] + self._v_grid[:-1, 1:] + self._v_grid[1:, 1:]) / 4
        j_back = self._grid_indices[1, 1:-1, :-1] - self.dt / self.h * self._u_grid[1:-1]
        i_back = np.clip(i_back, 0.5, self.M + 0.5)
        new_u_grid = self.interpolate(self._u_grid, np.stack([i_back, j_back]))
        return new_u_grid

    def _advectV(self):
        """advect vertical velocity (v) for ([0, M], [1, N])"""
        i_back = self._grid_indices[0, :-1, 1:-1] - self.dt / self.h * self._v_grid[:, 1:-1]
        j_back = self._grid_indices[1, :-1, 1:-1] - self.dt / self.h * (
            self._u_grid[:-1, :-1] + self._u_grid[1:, :-1] + self._u_grid[:-1, 1:] + self._u_grid[1:, 1:]) / 4
        j_back = np.clip(j_back, 0.5, self.N + 0.5)
        new_v_grid = self.interpolate(self._v_grid, np.stack([i_back, j_back]))
        return new_v_grid

    def _solve_pressure(self):
        """solve pressure field for laplacian(pressure) = divergence(velocity)"""
        # compute divergence of velocity field
        h = self.h
        div = (self._u_grid[1:-1, 1:] - self._u_grid[1:-1, :-1] + 
            self._v_grid[1:, 1:-1] - self._v_grid[:-1, 1:-1]) / h
        
        # solve poisson equation as a linear system
        rhs = div.flatten() * h * h
        p_grid = self.pressure_solver(rhs).reshape(self.M, self.N)
        return p_grid

    def _project(self):
        """projection step to enforce divergence free (incompressible flow)"""
        # set solid boundary condition
        self.boundary_func(self._d_grid, self._u_grid, self._v_grid, self.h)

        p_grid = self._solve_pressure()

        # apply gradient of pressure to correct velocity field
        # ([1, M], [1, N-1]) for u, ([1, M-1], [1, N]) for v. FIXME: this range holds only for solid boundary.
        self._u_grid[1:-1, 1:-1] -= (p_grid[:, 1:] - p_grid[:, :-1]) / self.h
        self._v_grid[1:-1, 1:-1] -= (p_grid[1:, :] - p_grid[:-1, :]) / self.h

        self.boundary_func(self._d_grid, self._u_grid, self._v_grid, self.h)

    def _set_solid_boundary(self, d_grid, u_grid, v_grid, h):
        """set solid (zero Dirichlet) boundary condition"""
        for grid in [d_grid, u_grid, v_grid]:
            grid[0, :] = 0
            grid[-1, :] = 0
            grid[:, 0] = 0
            grid[:, -1] = 0

    def draw(self, attr: str, save_path: str):
        """draw a frame"""
        since = time.time()

        if attr == "velocity":
            draw_velocity(self.grid_velocity, save_path)
        elif attr == "curl":
            # draw_curl(self.grid_curl, save_path)
            
            if self.timestep >= 200 and self.timestep < 900:
                t = self.timestep - 200
                draw_curl_h5(self.grid_curl, t * self.dt, t, save_path)
                draw_vel_u(self._u_grid, t * self.dt, t, save_path)
                draw_vel_v(self._v_grid, t * self.dt, t, save_path)
        elif attr == "density":
            draw_density(self.grid_density, save_path)
        elif attr == "mix":
            draw_mix(self.grid_curl, self.grid_density, save_path)
        else:
            raise NotImplementedError

        timecost = time.time() - since
        return timecost
    
    def load_uv(self, u, v):
        self._u_grid = u
        self._v_grid = v
        
        
def grid_velocity(_u_grid, _v_grid):
    """velocity values at grid centers"""
    u = (_u_grid[1:-1, 1:] + _u_grid[1:-1, :-1]) / 2
    v = (_v_grid[1:, 1:-1] + _v_grid[:-1, 1:-1]) / 2
    vel = np.stack([v, u], axis=-1)
    return vel

def compute_curl(u, v, h):
    
    velocity_field = grid_velocity(u, v)
    
    dvy_dx = np.gradient(velocity_field[..., 1], h)[0]
    dvx_dy = np.gradient(velocity_field[..., 0], h)[1]
    curl = dvy_dx - dvx_dy
    return curl  
        
parser = argparse.ArgumentParser(
    description='Neural Representation evolving')
parser.add_argument('-folderu', help='path to the weight',
                    type=str, required=True)
parser.add_argument('-nameu', help='path to the weight',
                    type=str, required=True)
parser.add_argument('-datau', help='path to the initial data',
                    type=str, required=True)           

parser.add_argument('-folderv', help='path to the weight',
                    type=str, required=True)
parser.add_argument('-namev', help='path to the weight',
                    type=str, required=True)
parser.add_argument('-datav', help='path to the initial data',
                    type=str, required=True)
                    
parser.add_argument('-resolution', help='resolution',
                    type=int, required=True, default=64)  
parser.add_argument('-total_steps', help='resolution',
                    type=int, required=True, default=650)        
parser.add_argument('-save_name', help='save name',
                    type=str, required=True, default="")         

parser.add_argument('-umax', help='umax',
                    type=float, required=True, default=1.0)       
parser.add_argument('-umin', help='umin',
                    type=float, required=True, default=-1.0)      
parser.add_argument('-vmax', help='vmax',
                    type=float, required=True, default=1.0)       
parser.add_argument('-vmin', help='vmin',
                    type=float, required=True, default=-1.0)                      

args = parser.parse_args()

total_steps = args.total_steps

date = args.folderu
name = args.nameu
me = "{}/{}_enc.pt".format(date, name)
md = "{}/{}_dec.pt".format(date, name)
md_grad = "{}/{}_dec_func_grad.pt".format(date, name)
net_enc_jit_load = torch.jit.load(me, map_location="cuda")
net_dec_jit_load = torch.jit.load(md, map_location="cuda")
net_dec_grad_jit_load = torch.jit.load(md_grad, map_location="cuda")
encoder_u = Encoder(net_enc_jit_load)
decoder_u = Decoder(net_dec_jit_load, md, net_dec_grad_jit_load)

date = args.folderv
name = args.namev
me = "{}/{}_enc.pt".format(date, name)
md = "{}/{}_dec.pt".format(date, name)
md_grad = "{}/{}_dec_func_grad.pt".format(date, name)
net_enc_jit_load = torch.jit.load(me, map_location="cuda")
net_dec_jit_load = torch.jit.load(md, map_location="cuda")
net_dec_grad_jit_load = torch.jit.load(md_grad, map_location="cuda")
encoder_v = Encoder(net_enc_jit_load)
decoder_v = Decoder(net_dec_jit_load, md, net_dec_grad_jit_load)


umax = args.umax
umin = args.umin
vmax = args.vmax
vmin = args.vmin

print("umax", umax)
print("umin", umin)
print("vmax", vmax)
print("vmin", vmin)


diff_threshold = 1e-15
step_size_threshold = 1e-15


setup = {
    "domain": [[0, np.pi], [0, 2 * np.pi]],
    "vsource": karman_velocity,
    "dsource": None,
    "src_duration": 1,
    "boundary_func": karman_boundary
}

cfg = {
    "tag": "karman2",
    "example": "karman",
    "N": args.resolution,
    "dt": 0.1,
    "T": 900,
    "diff": 0,
    "visc": 0,
    "draw": "curl",
    "fps": 40
}


SF = StableFluids(cfg["N"], cfg["dt"], setup["domain"], cfg["visc"], cfg["diff"], setup["boundary_func"])

h = SF.h
M = SF.M
N = SF.N

file_path_u = args.datau
file_path_v = args.datav

save_path_u = "output/Karman/vel_u_{}".format(args.save_name)
save_path_v = "output/Karman/vel_v_{}".format(args.save_name)
save_path = "output/Karman/curl_{}".format(args.save_name)

os.makedirs(save_path_u, exist_ok=True)
os.makedirs(save_path_v, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

print("save_path_u", save_path_u)
print("save_path_v", save_path_v)
print("save_path", save_path)


problem_u = Karman(file_path_u)
problem_v = Karman(file_path_v)

nonlinear_solver_u = NonlinearSolver(problem_u, decoder_u, diff_threshold, step_size_threshold)
nonlinear_solver_v = NonlinearSolver(problem_v, decoder_v, diff_threshold, step_size_threshold)

sample_point_u = np.arange((M+2) * (N+1))
sample_point_v = np.arange((M+1) * (N+2))

def karman_boundary_torch(d_grid, u, v, h):
    """d, u, v from stagger grid. h: grid size"""
    grid_indices = np.indices(d_grid.shape)
    
    u_grid = u
    v_grid = v

    def _transform_coords(coords, offset):
        """transform coords in grid space to original domain"""
        minxy = np.array([0, 0])[None, None]
        offset = np.array(offset)[None, None]
        coords = coords.transpose((1, 2, 0)).astype(float) + offset
        coords = coords * h + minxy
        return coords

    # solid sphere obstacle
    u_coords = _transform_coords(grid_indices[:, :, :-1], [-0.5, 0])
    v_coords = _transform_coords(grid_indices[:, :-1], [0, -0.5])
    mask_u = sphere_obstacle(u_coords) < 0
    mask_v = sphere_obstacle(v_coords) < 0
    u_grid[mask_u] = 0
    v_grid[mask_v] = 0

    # domain boundary: open right side
    u_grid[0, :] = 0
    u_grid[-1, :] = 0
    u_grid[:, 0] = KARMAN_VEL

    v_grid[0, :] = 0
    v_grid[-1, :] = 0
    v_grid[:, 0] = 0
    v_grid[:, -1] = 0

    d_grid[0, :] = 0
    d_grid[-1, :] = 0
    d_grid[:, 0] = 0
    d_grid[:, -1] = 0
    
    return u_grid, v_grid

def draw_uv(i, suffix=""):

    img_u = vel_u / 2 + 0.5
    img_v = vel_v / 2 + 0.5

    img_u = Image.fromarray((img_u * 255).astype('uint8'))
    img_v = Image.fromarray((img_v * 255).astype('uint8'))

    save_path_u_img = os.path.join(save_path_u, "f{}_{}.png".format(suffix, str(i).zfill(10)))
    save_path_v_img = os.path.join(save_path_v, "f{}_{}.png".format(suffix, str(i).zfill(10)))

    img_u.save(save_path_u_img)
    img_v.save(save_path_v_img)

    curl = compute_curl(vel_u_ori, vel_v_ori, h)

    # curl = (curl - -9)/(9 - -9)
    curl = (1 / (np.exp(-0.1*curl) + 1) - 0.5) * 3 + 0.5
    
    curl[curl > 1] = 1
    curl[curl < 0] = 0

    img = cm.bwr(curl)

    img = Image.fromarray((img * 255).astype('uint8'))

    save_path_img = os.path.join(save_path, "f{}_{}.png".format(suffix, str(i).zfill(10)))

    img.save(save_path_img)
    

with torch.no_grad():
    q0_gt_u = problem_u.getPhysicalInitialCondition()
    q0_gt_v = problem_v.getPhysicalInitialCondition()
    

    xhat_u = encoder_u.forward(q0_gt_u)
    xhat_v = encoder_v.forward(q0_gt_v)

    q_u = problem_u.updateState(xhat_u, decoder_u)
    q_v = problem_v.updateState(xhat_v, decoder_v)
    
    vel_u = q_u.view(M+2, N+1).cpu().numpy()
    vel_v = q_v.view(M+1, N+2).cpu().numpy()

    vel_u_ori = (vel_u + 1) / 2 * (umax - umin) + umin
    vel_v_ori = (vel_v + 1) / 2 * (vmax - vmin) + vmin
    
    vel_u_ori, vel_v_ori = karman_boundary_torch(SF._d_grid, vel_u_ori, vel_v_ori, h)

    draw_uv(0)

    for i in tqdm(range(1, args.total_steps)):
        
        SF.load_uv(vel_u_ori, vel_v_ori)
        
        SF.step()
        
        new_u = SF._u_grid
        new_v = SF._v_grid
        
        
        new_u = torch.tensor(new_u).cuda().view(1, -1, 1).float()
        new_v = torch.tensor(new_v).cuda().view(1, -1, 1).float()
        
        new_u = (new_u - umin) / (umax - umin) * 2 - 1
        new_v = (new_v - vmin) / (vmax - vmin) * 2 - 1
        
        
        if i < 60 or i % 22 == 0:
            xhat_u = encoder_u.forward(new_u)
        xhat_v = encoder_v.forward(new_v)
        
        xhat_u = nonlinear_solver_u.solve(xhat_u, new_u, sample_point_u, step_size = 1, max_iters = 10)
        
        q_u = problem_u.updateState(xhat_u, decoder_u)
        q_v = problem_v.updateState(xhat_v, decoder_v)
        
        vel_u = q_u.view(M+2, N+1).cpu().numpy()
        vel_v = q_v.view(M+1, N+2).cpu().numpy()
        
        vel_u_ori = (vel_u + 1) / 2 * (umax - umin) + umin
        vel_v_ori = (vel_v + 1) / 2 * (vmax - vmin) + vmin
        
        vel_u_ori, vel_v_ori = karman_boundary_torch(SF._d_grid, vel_u_ori, vel_v_ori, h)
        
        draw_uv(i)
        
        
        



        
        


