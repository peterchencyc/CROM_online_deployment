import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
rootrdir = os.path.dirname(parentdir)
commo_dir = os.path.join(rootrdir,'common')
sys.path.append(commo_dir)

sys.path.append(currentdir)

from Experiment import *
import torch
import numpy as np
import math
from SimulationData import *

def f(w):
    return 0.5 * w ** 2

class Burger(Experiment):

    def __init__(self, ini_cond, params=None):
       
        if params is None:
            self.mu1 = 5.15
            self.mu2 = 0.0285
            self.dt = 0.07
            self.alpha = 1 
            self.dx = 100/255
        else:
            self.mu1 = params["mu1"]
            self.mu2 = params["mu2"]
            self.dt = params["dt"] * params["run_dt"]
            self.alpha = params["vel"]
            self.dx = params["dx"]

        state = SimulationState(ini_cond)
        self.x = torch.FloatTensor(state.x).cuda().view(1, -1, 1)
        self.q0_gt = torch.from_numpy(state.q).cuda().view(1, -1, 1)
    
    def getPhysicalInitialCondition(self):
        return self.q0_gt
    
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
        sq[0,0,0] = self.mu1
        if save_q:
            self.q = sq
        return sq.detach()
        
    def updateStateBase(self, xhat, decoder_base, vhat=None, jac=None):
        q = decoder_base.forward(xhat)
        self.q = q.view(1, q.size(0), q.size(1))

        if vhat is not None:
            self.v = torch.matmul(jac, vhat.view(-1,1)).view_as(self.v)
        return self.q.detach()

    
    def getResidual(self, xhat, decoder, index):
        q = self.updateState(xhat, decoder).view(1, -1, 1)
        # print("q", q[:,:50].flatten())
        
        q_pad = torch.zeros_like(q)
        q_pad = torch.cat([q_pad, q_pad[:,:2]], 1)
        q_pad[:, 1:-1] = q
        q_ = self.alpha *( - (f(q_pad[:, 1:-1]) - f(q_pad[:, 0:-2])) / self.dx + 0.02 * torch.exp(self.mu2 * self.x))
        q_[:,0,:] = 0.
        return q_
        
    def getResidualFDBase(self, xhat, decoder_base, index):
        q = self.updateStateBase(xhat, decoder_base).view(1, -1, 1)
        
        q_pad = torch.zeros_like(q)
        q_pad = torch.cat([q_pad, q_pad[:,:2]], 1)
        q_pad[:, 1:-1] = q
        q_ = self.alpha *( - (f(q_pad[:, 1:-1]) - f(q_pad[:, 0:-2])) / self.dx + 0.02 * torch.exp(self.mu2 * self.x))
        q_[:,0,:] = 0.

        return q_
    
    
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