import torch
import numpy as np

class SamplePoint(object):
    def __init__(self, problem):
        self.problem = problem
    
    def updateIndicesRandom(self, num_sample_interior, num_sample_bdry=-1, seed=0):
        torch.manual_seed(seed)
        if self.problem.__class__.__name__ == 'ElasticityFem':
            indices_bdry, _ = self.problem.computeCollisionIndicesAndVelocities(self.problem.x[0, :, :], step=0)
            selection_interior = torch.ones((self.problem.x.size(1))).to(self.problem.x.device)
            selection_interior[indices_bdry] = 0
            indices_interior = selection_interior.nonzero().view(-1)

            if num_sample_interior== -1 and num_sample_bdry==-1:
                self.updateIndicesFull()
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
        
        elif self.problem.__class__.__name__ == 'DiffuseImage' or self.problem.__class__.__name__ == 'Diffusion':
            if num_sample_interior == -1:
                self.updateIndicesFull()
            else:
                self.indices = torch.randperm(indices_interior.size(0))[:num_sample_interior]
    
    def updateIndicesUniform(self):
        torch.manual_seed(seed)
        if self.problem.__class__.__name__ == 'DiffuseImage':
            x, y = np.meshgrid(np.asarray(np.linspace(16,240,8), dtype=np.int32),np.asarray(np.linspace(16,240,8), dtype=np.int32))
            self.indices = torch.tensor([x.flatten()[i]+256*y.flatten()[i] for i in range(len(x.flatten()))])
        elif self.problem.__class__.__name__ == 'Diffusion':
            self.indices = torch.tensor(np.append(12+np.arange(20)*25, np.asarray([0,500])))
            
    def updateIndicesFull(self):
        self.indices = torch.tensor(list(range(self.problem.x.size(1))))
        
        
    def updateIndicesOptimal(self):
        if self.problem.__class__.__name__ == 'DiffuseImage':
            self.indices = torch.tensor(np.load("data/DiffuseImage/optimal_selection.npy"))
        elif self.problem.__class__.__name__ == 'Diffusion':
            self.indices = torch.tensor(np.load("data/Diffusion/optimal_selection.npy"))
    
    def initialize(self, sample_style, num_sample_interior, num_sample_bdry, decoder, xhat, vhat):
        if self.problem.__class__.__name__ == 'ElasticityFem':
            if sample_style == 'random':
                self.updateIndicesRandom(num_sample_interior, num_sample_bdry)
            self.problem.updateSampleIndicesWithProblemSpeficifInfo(self)
            self.problem.jac_sample = self.problem.getJacobianSample(xhat, decoder, self)
            self.problem.updateStateSampleAll(xhat, decoder, self, vhat)
        elif self.problem.__class__.__name__ == 'DiffuseImage':
            if sample_style == 'random':
                self.updateIndicesRandom(num_sample_interior)
            elif sample_style == 'optimal':
                self.updateIndicesOptimal()
            elif sample_style == 'uniform':
                self.updateIndicesUniform()
            else:
                self.updateIndicesFull()
        elif self.problem.__class__.__name__ == 'Diffusion':
            if sample_style == 'random':
                self.updateIndicesRandom(num_sample_interior)
            elif sample_style == 'optimal':
                self.updateIndicesOptimal()
            elif sample_style == 'uniform':
                self.updateIndicesUniform()
            else:
                self.updateIndicesFull()
        
        self.indices = torch.tensor(self.indices, dtype=torch.long)
    

        
        
        
