import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
rootrdir = os.path.dirname(parentdir)
commo_dir = os.path.join(rootrdir,'common')
sys.path.append(commo_dir)

from Experiment import *
import torch
import numpy as np
import math
from SimulationData import *
from torch import linalg as LA
import json

class DiffuseImage(Experiment):
    def callBack(self, step, xhat, decoder, vhat=None, xhat_prev=None):
        pass

    def __init__(self, config_path, ini_cond, device):

        with open(config_path, 'r') as f:
            gamma = json.load(f)
    
    
        self.dx = 1
        self.dt = 1
        self.Nstep = 20
        self.kappa = 50
        state = SimulationState(ini_cond)
        self.x = torch.from_numpy(state.x).to(device).view(1, -1, 2)
        self.q0_gt = torch.from_numpy(state.q).to(self.x.device).view(1, -1, 1)
        self.gamma = torch.tensor(gamma).to(self.x.device).view(-1, 1)

        # find boundaries
        [xdim, ydim] = state.x.max(axis=0)
        xdim += 1
        ydim += 1
        self.bdry_ind = torch.zeros(xdim, ydim)

        for x in range(xdim):
            self.bdry_ind[x,0]=1.
            self.bdry_ind[x,ydim-1]=1.

        for y in range(ydim):
            self.bdry_ind[0,y]=1.
            self.bdry_ind[xdim-1,y]=1.
        
        self.xdim = xdim
        self.ydim = ydim
        
        self.bdry_ind = self.bdry_ind.view(-1).to(self.x.device)
    
    def getPhysicalInitialCondition(self):
        return self.q0_gt.float()
    
    def write2file(self, time, filename, xhat):
        input_x = self.x[0,:,:].detach().cpu().numpy()
        input_q = self.q[0,:,:].detach().cpu().numpy()
        input_t = np.array([[time]])
        state = SimulationState(filename, False, input_x, input_q, input_t)
        state.write_to_file()

    def updateState(self, xhat, decoder, index=None, vhat=None, xhat_prev=None, save_q=False):
        if index is None:
            index = np.arange(self.x.size(1))
            
            
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:, index, :]), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        q = decoder.forward(x)
        sq = q.view(1, q.size(0), q.size(1))

        if save_q:
            self.q = sq
        return sq.detach()

    def getResidual(self, xhat, decoder, index=None):
        if index is None:
            index = np.arange(self.x.size(1))
        
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:, index, :]), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        _, hessian_part = decoder.hessianPart(x, lbllength, 'sec')
        # Perona–Malik diffusion        
        hessian_part = torch.sum(hessian_part, dim=(1,2,3))

        bdry_ind = self.bdry_ind.view(-1, 1)[index,:]
        hessian_part = hessian_part.view(-1, 1)
        hessian_part = torch.where(bdry_ind < 1., hessian_part, torch.zeros_like(hessian_part)) # dirichlet zero boundary condition
        
        hessian_part *= self.gamma[index]

        return hessian_part
    
    def getResidualFD(self, xhat, decoder):
        q = self.updateState(xhat, decoder)
        q_backup = q.view(q.size(1), q.size(2))
        q = q.view(self.xdim, self.ydim)

        deltas = [torch.zeros_like(q) for _ in range(q.dim())]
        # calculate the diffs
        for i in range(q.dim()):
            slicer = [slice(None, -1) if j == i else slice(None) for j in range(q.dim())]
            diff_local = torch.diff(q, axis=i)
            deltas[i][tuple(slicer)] = diff_local

        voxelspacing = tuple([1.] * q.dim())

        # multiply c
        matrices = [delta for delta, spacing in zip(deltas, voxelspacing)]

        # second derivative
        for i in range(q.dim()):
            slicer = [slice(1, None) if j == i else slice(None) for j in range(q.dim())]
            matrices[i][tuple(slicer)] = torch.diff(matrices[i], axis=i)

        matrices = matrices[0] + matrices[1] # 2d image
        matrices = matrices.view_as(q_backup)
        return matrices

        
    
    def getJacobian(self, xhat, decoder, index=None):
        if index is None:
            index = np.arange(self.x.size(1))
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:, index, :]), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        jacobian_part = decoder.jacobianPart(x, lbllength, 'fir')
        jacobian_part = jacobian_part.view(-1, jacobian_part.size(2))
        return jacobian_part
    
    def getResidualSample(self, xhat, decoder, sample_point, step):
        return self.getResidual(xhat, decoder, sample_point.indices)
    
    def getJacobianSample(self, xhat, decoder, sample_point):
        return self.getJacobian(xhat, decoder, sample_point.indices)
    
    def getJacobianSampleProj(self, xhat, decoder, sample_point):
        return self.getJacobianSample(xhat, decoder, sample_point.indices)
    
    def updateStateSample(self, xhat, decoder, sample_point, vhat=None):
        self.q_sample = self.updateState(xhat, decoder, sample_point.indices, save_q=False)
        return self.q_sample

    def updateStateSampleAll(self, xhat, decoder, sample_point, v_hat=None):
        q_all = self.updateState(xhat, decoder, None, save_q=True)
        return q_all
    
    def updateSampleIndicesWithProblemSpeficifInfo(self, sample_point):
        pass
