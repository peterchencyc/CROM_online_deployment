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

import json

class Diffusion(Experiment):
    def loadConfig(self, config_path):
        with h5py.File(config_path, 'r') as h5_file:
            T = h5_file['/T'][()]
            Nx = int(h5_file['/Nx'][()])
            mu = h5_file['/mu'][()]
        return T, Nx, mu

    def __init__(self, config_path, F_path, ini_cond, device, a=1):
        T, Nx, mu = self.loadConfig(config_path)
        with open(F_path, 'r') as f:
            d = json.load(f)
            F = [2*i for i in d["F"]]
        
        L = 1.        
        sigma=0.05
        self.a = a
        F0 = 0.5
        self.alpha = torch.tensor(a*F).cuda().view(-1,1,1,1)
        x = np.linspace(0, L, Nx+1)   # mesh points in space
        dx = x[1] - x[0]
        self.dx = dx
        self.dt = F0*dx**2/a
        self.Nstep = int(round(T/float(self.dt)))
        
        state = SimulationState(ini_cond)
        self.x = torch.from_numpy(state.x).to(device).view(1, -1, 1).float()
        self.q0_gt = torch.from_numpy(state.q).to(self.x.device).view(1, -1, 1).float()

    
    def getPhysicalInitialCondition(self):
        return self.q0_gt.float()
    
    def write2file(self, time, filename, xhat):
        input_x = self.x[0,:,:].detach().cpu().numpy()
        input_q = self.q[0,:,:].detach().cpu().numpy()
        input_t = np.array([[time]])
        state = SimulationState(filename, False, input_x, input_q, input_t)
        state.write_to_file()

    def updateState(self, xhat, decoder, index=None, vhat=None, xhat_prev=None, save_q=True):
        if index is None:
            index = np.arange(self.x.size(1))
            
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:,index, :]), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        q = decoder.forward(x)
        sq = q.view(1, q.size(0), q.size(1))
        if save_q:
            self.q = sq
        return sq.detach()


    def getResidual(self, xhat, decoder, index=None):
        if index is None:
            index = np.arange(self.x.size(1))
        n_points = self.x.size(1)
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:, index, :]), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        _, hessian_part = decoder.hessianPart(x, lbllength, 'sec')
        hessian_part *= self.alpha[index,:,:,:]
        hessian_part = hessian_part.view(-1, 1)
        for i in range(len(index)):
            if index[i] == 0 or index[i] == n_points-1:
                hessian_part[i] = 0
        return hessian_part
    
    def getJacobian(self, xhat, decoder, index):
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), len(index), xhat.size(2))
        x = torch.cat((xhat_all, self.x[:,index, :]), 2)
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