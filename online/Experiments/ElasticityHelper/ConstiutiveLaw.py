import torch
import math
from torch import linalg as LA
from abc import abstractmethod

class ConstiutiveLaw(object):
    @abstractmethod
    def update(self, def_grad):
        pass

    @abstractmethod
    def computedPdF(self, def_grad):
        pass

class LinearCorotated(ConstiutiveLaw):
    def __init__(self, mu, lamb):
        self.mu = mu
        self.lamb = lamb
        self.sim_dim = 3
        
    
    def print(self):
        # print('mu: ', self.mu)
        # print('lamb: ', self.lamb)
        print('def_grad: ', self.def_grad)
        # print('S: ', self.S)
        # print('R: ', self.pd_l)
        # print('F-R: ', self.def_grad-self.pd_l)
        print('P: ', self.first_piola)

    
    def update(self, def_grad):
        self.def_grad = def_grad
        U, S, Vh = torch.linalg.svd(def_grad)
        self.S = S
        
        # https://en.wikipedia.org/wiki/Polar_decomposition#Relation_to_the_SVD
        self.pd_r = Vh.transpose(1, 2) @ torch.diag_embed(S) @ Vh #S
        self.pd_l = U @ Vh #R

        self.first_piola = self.computeFirstPiola(self.def_grad, self.pd_l) # confirmed to be approximately the same as rom4mpm, which means mu and lambda are both correct, and the polar decomposition is solid.
    
    def computeFirstPiola(self, F, R):
        RTF = torch.einsum('bji, bjk->bik', R, F)
        RTFmI = RTF - torch.eye(self.sim_dim).type_as(F)
        trace = torch.einsum('bii->b', RTFmI)
        traceR = torch.einsum('b, bij->bij',trace, R)
        return 2 * self.mu * (F-R) + self.lamb * traceR
    
    def computedPdF(self):
        A = self.computeATensor()
        self.C = self.computedRdF() # B also needs C tensor so this should be called before BTensor
        B = self.computeBTensor()
        
        result = 2*self.mu*(A-self.C) + self.lamb*B
        return result

    def computeATensor(self):
        eye = torch.eye(self.sim_dim).type_as(self.pd_l) # dim by dim
        eye = eye.view(-1,1) # dim*dim by 1
        eye_fourth = eye @ eye.transpose(0,1) # dim*dim by dim*dim
        eye_fourth = eye_fourth.view(self.sim_dim,self.sim_dim,self.sim_dim,self.sim_dim)
        eye_fourth = eye_fourth.transpose(1,2)
        eye_fourth = eye_fourth.expand(self.pd_l.size(0),-1,-1,-1,-1)
        return eye_fourth

    def computedRdF(self):
        eye = torch.eye(self.sim_dim).type_as(self.pd_r)
        IS_tensor =  torch.einsum('ac,bkp->bakcp',eye, self.pd_r).view(self.pd_r.size(0), self.sim_dim*self.sim_dim, self.sim_dim*self.sim_dim)
        SI_tensor =  torch.einsum('bac,kp->bakcp',self.pd_r, eye).view(self.pd_r.size(0), self.sim_dim*self.sim_dim, self.sim_dim*self.sim_dim)
        ISSI = IS_tensor+SI_tensor
        ISSI_inv = LA.inv(ISSI) # b,9,9

        E_l =  torch.einsum('li,bkj->bklij', eye, self.def_grad)
        E_r =  torch.einsum('jl,bki->bklij', eye, self.def_grad)
        E = E_l + E_r
        E_batch = E.view(-1, E.size(3), E.size(4)) #b*k*l,i,j

        # https://en.wikipedia.org/wiki/Vectorization_(mathematics)
        def mat2vec(mat):
            return torch.einsum('bij->bji', mat).reshape(mat.size(0), -1, 1)
        def vec2mat(vec):
            mat_dim = int(math.sqrt(vec.size(1)))
            mat_col_major = vec.view(vec.size(0), mat_dim, mat_dim)
            return torch.einsum('bji->bij', mat_col_major)

        E_vec = mat2vec(E_batch) #b*k*l,9,1
        E_vec_kl = E_vec.view(E.size(0), E.size(1), E.size(2), E_vec.size(1), E_vec.size(2)) #b,k,l,9,1
        
        D_vec_kl = torch.einsum('bmn, bklno->bklmo', ISSI_inv, E_vec_kl) #b,k,l,9,1
        D_vec = D_vec_kl.reshape(-1, D_vec_kl.size(3), D_vec_kl.size(4)) #b*k*l,9,1
        D_batch = vec2mat(D_vec) #b*k*l,3,3
        D = D_batch.view_as(E) #b,k,l,3,3
        
        dSdF = torch.einsum('bklij->bijkl', D)
        RdSdF = torch.einsum('bim,bmjkl->bijkl', self.pd_l, dSdF)
        delta_fourth = torch.einsum('ik,jl->ijkl', eye, eye)

        rhs = delta_fourth - RdSdF
        rhs_kl = torch.einsum('bijkl->bklij', rhs)

        pd_r_inv = LA.inv(self.pd_r)
        dRdF = torch.einsum('bklij,bjm->bimkl', rhs_kl, pd_r_inv)
        
        return dRdF
    
    def computeBTensor(self):
        RR = torch.einsum('bij,blm->bijlm', self.pd_l, self.pd_l)
        RF = torch.einsum('bij,bpo->bijpo', self.pd_l, self.def_grad)
        RFC = torch.einsum('bijpo, bpolm->bijlm', RF, self.C)
        RF_scalar = torch.einsum('bpo,bpo->b', self.pd_l, self.def_grad)
        RF_scalar -= 3
        third_term = torch.einsum('b,bijlm->bijlm', RF_scalar, self.C)
        B = RR + RFC + third_term
        return B
