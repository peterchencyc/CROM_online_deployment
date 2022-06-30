import torch
import numpy as np

class SamplePoint(object):
    def __init__(self, problem):
        self.problem = problem
    
    def updateIndicesRandom(self, num_sample_interior, num_sample_bdry, seed=0):
        torch.manual_seed(seed)
        indices_bdry, _ = self.problem.computeCollisionIndicesAndVelocities(self.problem.x[0, :, :], step=0)
        selection_interior = torch.ones((self.problem.x.size(1))).to(self.problem.x.device)
        selection_interior[indices_bdry] = 0
        indices_interior = selection_interior.nonzero().view(-1)

        if num_sample_interior== -1 and num_sample_bdry==-1:
            self.indices = torch.tensor(list(range(self.problem.x.size(1))))
        else:
            if num_sample_interior!=-1:
                ind_rand = torch.randperm(indices_interior.size(0))[:num_sample_interior]
                indices_interior_select = indices_interior[ind_rand]
            else:
                indices_interior_select = indices_interior
            if num_sample_bdry!=-1:
                ind_rand = torch.randperm(indices_bdry.size(0))[:num_sample_bdry]
                indices_bdry_select = indices_bdry[ind_rand]
            else:
                indices_bdry_select = indices_bdry

            self.indices = torch.cat((indices_interior_select, indices_bdry_select))
        self.indices = self.indices.to(self.problem.x.device)
        
        self.indices_xyz = torch.stack([3*self.indices, 3*self.indices+1, 3*self.indices+2]).t().reshape(-1).to(self.problem.x.device)
        
        
    
    def initialize(self, sample_style, num_sample_interior, num_sample_bdry, decoder, xhat, vhat):
        if sample_style == 'random':
            self.updateIndicesRandom(num_sample_interior, num_sample_bdry)
        self.problem.updateSampleIndicesWithProblemSpeficifInfo(self)
        self.problem.jac_sample = self.problem.getJacobianSample(xhat, decoder, self)
        self.problem.updateStateSampleAll(xhat, decoder, self, vhat)
    
    def initialize_diffusion(self, sample_style, num_sample, decoder, xhat):
        if self.problem.__class__.__name__ == 'DiffuseImage':
            if sample_style == 'random':
                self.indices = np.random.choice(np.arange(self.problem.x.shape[1]), num_sample)
            elif sample_style == 'optimal':
                self.indices = np.asarray([31998, 45714, 28942, 23539, 43573, 64003, 26601,  2461, 28917,
                                           61036, 51421, 45607, 57811, 51065, 45619, 26052,  5698, 53996,
                                           61626, 34154, 25595, 57847,   391,  1274, 28683, 39167, 54956,
                                           20622, 61846, 12096,  1404, 33973, 64733,  3495, 59022, 62242,
                                           33768, 61202, 20508, 49453, 39291, 59354, 15486, 58285, 35930,
                                           27477, 61957,  8745, 20925, 48365, 19538, 19616, 39803, 20051,
                                           12419, 22138, 65142, 24822, 37456, 21202, 19153, 28892, 62582])
            elif sample_style == 'uniform':
                x, y = np.meshgrid(np.asarray(np.linspace(16,240,8), dtype=np.int32),np.asarray(np.linspace(16,240,8), dtype=np.int32))
                self.indices = np.asarray([x.flatten()[i]+256*y.flatten()[i] for i in range(len(x.flatten()))])
            else:
                self.indices = np.arange(self.problem.x.shape[1])
        
        elif self.problem.__class__.__name__ == 'Diffusion':
            if sample_style == 'random':
                self.indices = np.random.choice(np.arange(self.problem.x.shape[1]), num_sample)
            elif sample_style == 'optimal':
                self.indices = np.asarray([51, 205, 315, 261, 214, 144, 356, 267, 115, 271, 244, 309, 62, 364, 412, 467, 249, 403, 175, 328, 105, 354])
            elif sample_style == 'uniform':
                self.indices = np.append(12+np.arange(20)*25, np.asarray([0,500]))
            else:
                self.indices = np.arange(self.problem.x.shape[1])
        
        
