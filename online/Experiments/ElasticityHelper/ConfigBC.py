import torch
import numpy as np
from abc import abstractmethod
from .HandleCollision import *

class ConfigBC(object):
    def __init__(self, test_number, device):
        self.bc = None
        self.device = device
        if test_number == 8:
            self.bc = self.config8()
        elif test_number == 9:
            self.bc = self.config9()
        elif test_number == 10:
            self.bc = self.config9()
        elif test_number == 11:
            self.bc = self.config11()
        elif test_number == 12:
            self.bc = self.config12()
        elif test_number == 13:
            self.bc = self.config13()
        else:
            exit('invalid test number for bc')

    def config8(self):
        bc_hard = BoundaryConditionHard()
        bc_hard.append(Plane(torch.Tensor([2,-2.,0]).to(self.device), torch.Tensor([0,1,0]).to(self.device), [0.5, 2]))
        bc_hard.append(Plane(torch.Tensor([-2,-2.,0]).to(self.device), torch.Tensor([0,1,0]).to(self.device), [0.5, 2]))
        return bc_hard
    
    def config9(self):
        bc_hard = BoundaryConditionHard()
        vel_multiplier = 4
        omega_multiplier = 6
        bc_hard.append(Plane(torch.Tensor([-1.75,0.,0]).to(self.device), torch.Tensor([1,0,0]).to(self.device), vel=[-vel_multiplier,0,0], omega = [-omega_multiplier, 0, 0]))
        bc_hard.append(Plane(torch.Tensor([1.75,0.,0]).to(self.device), torch.Tensor([-1,0,0]).to(self.device), vel=[vel_multiplier,0,0], omega = [omega_multiplier, 0, 0]))
        return bc_hard
    
    def config11(self):
        bc_hard = BoundaryConditionHard()
        vel_multiplier = 4
        omega_multiplier = 11.3097
        bc_hard.append(Plane(torch.Tensor([-3,0.,0]).to(self.device), torch.Tensor([1,0,0]).to(self.device), vel=[-vel_multiplier,0,0], omega = [-omega_multiplier, 0, 0]))
        bc_hard.append(Plane(torch.Tensor([-1,0.,0]).to(self.device), torch.Tensor([-1,0,0]).to(self.device), vel=[vel_multiplier,0,0], omega = [omega_multiplier, 0, 0]))
        return bc_hard
    
    def config12(self):
        bc_hard = BoundaryConditionHard()
        bc_hard.append(Plane(torch.Tensor([2,-2.,0]).to(self.device), torch.Tensor([0,1,0]).to(self.device), [1, 2]))
        bc_hard.append(Plane(torch.Tensor([-2,-2.,0]).to(self.device), torch.Tensor([0,1,0]).to(self.device), [1, 2]))
        return bc_hard
    
    def config13(self):
        def activate(step):
            if step >= 130:
                return True
            else:
                return False
        width = 0.25
        bc_hard = BoundaryConditionHard()
        bc_hard.append(Plane(torch.Tensor([2-width,-1.,0]).to(self.device), torch.Tensor([-1,0,0]).to(self.device), activate=activate))
        bc_hard.append(Plane(torch.Tensor([-2+width,-1.,0]).to(self.device), torch.Tensor([1,0,0]).to(self.device), activate=activate))
        return bc_hard