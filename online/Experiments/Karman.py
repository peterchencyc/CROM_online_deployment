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



class Karman(Experiment):

    def __init__(self, ini_cond):
    
        state = SimulationState(ini_cond)
        self.x = torch.FloatTensor(state.x).cuda().view(1, -1, 2).float()
        self.q0_gt = torch.from_numpy(state.q).cuda().view(1, -1, 1).float()
    
    def getPhysicalInitialCondition(self):
        return self.q0_gt
    
    def write2file(self, time, filename):
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
        
    def updateStateBase(self, xhat, decoder_base, vhat=None, jac=None):
        q = decoder_base.forward(xhat)
        self.q = q.view(1, q.size(0), q.size(1))

        if vhat is not None:
            self.v = torch.matmul(jac, vhat.view(-1,1)).view_as(self.v)
        return self.q.detach()
    
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
    
    def getJacobianSample(self, xhat, decoder, sample_point):
        return self.getJacobian(xhat, decoder, sample_point)
    
    def updateStateSample(self, xhat, decoder, sample_point, save_q=False, save_sample=False):
        q_sample = self.updateState(xhat, decoder, sample_point, save_q=save_q)
        if save_sample:
            self.q_sample = q_sample
        return q_sample
        
    def getJacobianBase(self, xhat, decoder_base):
        jacobian = decoder_base.jacobian(xhat)
        return jacobian 
        