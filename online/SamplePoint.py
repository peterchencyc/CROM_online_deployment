import torch

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
