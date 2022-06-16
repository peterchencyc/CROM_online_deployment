import os
import sys

from .ElasticityHelper.ConstiutiveLaw import *
from .ElasticityHelper.ConfigBC import *
from .ElasticityHelper.MeshH5 import *
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
from timer_cm import Timer

def computeDifference(V, T):
    D = []
    bs = T.size(0)
    X0or4s = V[T[:,0]]
    X1s = V[T[:,1]]
    X2s = V[T[:,2]]
    X3s = V[T[:,3]]
    fir = (X1s-X0or4s).view(bs, 3, 1)
    sec = (X2s-X0or4s).view(bs, 3, 1)
    thi = (X3s-X0or4s).view(bs, 3, 1)

    D = torch.cat((fir, sec, thi), 2)
    return D

# https://stackoverflow.com/questions/65565461/how-to-map-element-in-pytorch-tensor-to-id
# causes memory blowup when the number of indices is large
def NewTensor2IndicesOfBaseTensor(tensor_new, tensor_base):
    return (tensor_new.view(-1,1) == tensor_base).int().argmax(dim=1).view(-1, tensor_new.size(1))


class ElasticityFem(Experiment):
    def callBack(self, step, xhat, decoder, vhat=None, xhat_prev=None):
        self.bc.step(self.dt)
        remesh_flag = False
        if self.remesh:
            for idx, this_step in enumerate(self.remesh_step):
                if step == this_step:
                    self.readMeshH5(self.remesh_file[idx])

                    self.updateState(xhat, decoder, vhat, xhat_prev)
                    if self.dis_or_pos == 'pos':
                        self.initial_offset = self.x-self.q if self.exact_ini else torch.zeros_like(self.q)
                    elif self.dis_or_pos == 'dis':
                        self.initial_offset = -self.q if self.exact_ini else torch.zeros_like(self.q)
                    print('new dof after remsh: ', self.x.shape)
                    remesh_flag = True
        return remesh_flag

        
    def loadConfig(self, config_path):
        with h5py.File(config_path, 'r') as h5_file:
            self.lamb = float(h5_file['/lambda'][()][0][0])
            self.mu = float(h5_file['/mu'][()][0][0])
            self.dt = float(h5_file['/dt'][()][0][0])
            self.Nstep = int(h5_file['/Nstep'][()][0][0])
            self.gravity = h5_file['/gravity'][()].astype(float)
            self.gravity = torch.from_numpy(self.gravity).float().to(self.device)
            if '/beta_damping' in h5_file:
                self.beta_damping = h5_file['/beta_damping'][()][0][0]
            else:
                self.beta_damping= 0.
            if '/test_number' in h5_file:
                self.test_number = h5_file['/test_number'][()][0][0]
            else:
                self.test_number= -1
    
    def updatedt(self, dt_div):
        self.dt = self.dt/dt_div
        self.Nstep = self.Nstep * int(dt_div)
    
    def print(self):
        self.constitutive_law.print()
        print('x: ', self.x)
        print('q: ', self.q)
        print('H_density: ', self.H_density)
        print('H: ', self.H)
        print('Dm_inv: ', self.Dm_inv)
        print('volume: ', self.volume)
        print('mass: ', self.mass)
        print('dv_int', self.dv_int)
        print('force: ', self.force_per_vertex)
        print('v_trial: ', self.v_trial)
    
    def __init__(self, config_path, ini_cond, dis_or_pos, device):
        self.device = device
        self.loadConfig(config_path)
        self.dis_or_pos = dis_or_pos
        self.sim_dim = 3
        state = SimulationState(ini_cond)
        self.x = torch.from_numpy(state.x).to(self.device).view(1, -1, self.sim_dim).float()
        sample_cnt = self.x.size(1)
        self.q0_gt = torch.from_numpy(state.q).to(self.x.device).view(1, -1, self.sim_dim).float()
        if hasattr(state, 'faces'):
            self.faces = state.faces
        else:
            self.faces = None
        if hasattr(state, 'tets'):
            self.tets = state.tets
            self.tets = torch.from_numpy(self.tets).to(self.x.device)
        else:
            self.tets = None
        self.constitutive_law = LinearCorotated(self.mu, self.lamb)
        if hasattr(state, 'masses'):
            self.mass = torch.from_numpy(state.masses).float().to(self.x.device)
        else:
            exit('must provide mass')
        self.mg = self.mass * self.gravity
        self.v = torch.zeros(self.mass.size(0), self.sim_dim).to(self.x.device)
        self.bc = ConfigBC(self.test_number, self.device).bc
        self.indices_bdry = None
        self.timer = Timer("ElasticityFem")
        self.q_sample_prev = None
    
    def getPhysicalInitialCondition(self):
        return self.q0_gt
    
    def writeSample2file(self, sample_point, time, filename, xhat):
        input_x = self.x[0, sample_point.indices, :].detach().cpu().numpy()
        input_q = self.getCurrentPosSample(sample_point).detach().cpu().numpy()
        faces = None

        input_t = np.array([[time]])
        state = SimulationState(filename, False, input_x, input_q, input_t, faces)
        state.label = xhat.detach().cpu().numpy()
        state.write_to_file()
        

    def write2file(self, time, filename, xhat):
        input_x = self.x[0,:,:].detach().cpu().numpy()
        input_q = self.getCurrentPos().detach().cpu().numpy()
        
        faces = self.faces
        
        if hasattr(self, 'vis_mesh_file'):
            input_x = self.vis_x[0,:,:].detach().cpu().numpy()
            if self.dis_or_pos == 'dis':
                input_q = (self.vis_x[0, :, :] + self.vis_q[0,:,:]).detach().cpu().numpy()
            elif self.dis_or_pos == 'pos':
                input_q = (self.vis_q[0,:,:]).detach().cpu().numpy()
            faces = self.vis_faces
        
        input_t = np.array([[time]])
        state = SimulationState(filename, False, input_x, input_q, input_t, faces)
        state.label = xhat.detach().cpu().numpy()
        state.write_to_file()
    
    def updateState(self, xhat, decoder, vhat=None, xhat_prev=None):
        # update position
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), self.x.size(1), xhat.size(2))
        x = torch.cat((xhat_all, self.x), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        q = decoder.forward(x)
        self.q = q.view(1, q.size(0), q.size(1))

        # update velocity
        if vhat is not None:
            jac = self.getJacobian(xhat_prev, decoder)
            vel = jac @ vhat.view(-1,1)
            self.v = vel.view(-1, self.sim_dim)
        
        if hasattr(self, 'vis_mesh_file'):
            xhat_all = xhat.expand(xhat.size(0), self.vis_x.size(1), xhat.size(2))
            x = torch.cat((xhat_all, self.vis_x), 2)
            x = x.view(x.size(0)*x.size(1), x.size(2))
            q = decoder.forward(x)
            self.vis_q = q.view(1, q.size(0), q.size(1))

        return self.q.detach()
    
    def getCurrentPos(self):
        if self.dis_or_pos == 'pos':
            current_pos = self.q[0, :, :] + self.initial_offset[0, :, :]
        elif self.dis_or_pos == 'dis':
            current_pos = self.x[0, :, :] + self.q[0, :, :] + self.initial_offset[0, :, :]
        return current_pos
    
    def getCurrentPosSamplHelper(self, q_data, sample_indices):
        if self.dis_or_pos == 'pos':
            current_pos = q_data[0, :, :]
        elif self.dis_or_pos == 'dis':
            current_pos = self.x[0, sample_indices, :] + q_data[0, :, :]
        return current_pos

    def getCurrentPosSampleAll(self, sample_point):
        return self.getCurrentPosSamplHelper(self.q_sample_all, sample_point.indices_all)
    
    def getCurrentPosSample(self, sample_point):
        return self.getCurrentPosSamplHelper(self.q_sample, sample_point.indices)
    
    def accumulateForce(self, f0or4, f1, f2, f3):
        # accumulate nodal forces
        force_per_vertex = torch.zeros_like(self.q[0, :, :])
        for idx, tet in enumerate(self.tets):
            force_per_vertex[tet[0]] += f0or4[idx, :]
            force_per_vertex[tet[1]] += f1[idx, :]
            force_per_vertex[tet[2]] += f2[idx, :]
            force_per_vertex[tet[3]] += f3[idx, :]
        return force_per_vertex
    
    def computeAccumulatedForceFromId(self, indices_id, force_all):
        zero_dummy = torch.Tensor([0,0,0]).type_as(force_all).view(1,3)
        force_all_icl_dummy = torch.cat((force_all, zero_dummy), 0)
        forces = force_all_icl_dummy[indices_id, :]
        forces_acc = torch.sum(forces, 1)
        return forces_acc

    def accumulateForceBatch(self, f0or4, f1, f2, f3, accumulate_inidces_all):
        forces_acc0or4 = self.computeAccumulatedForceFromId(accumulate_inidces_all[0], f0or4)
        forces_acc1 = self.computeAccumulatedForceFromId(accumulate_inidces_all[1], f1)
        forces_acc2 = self.computeAccumulatedForceFromId(accumulate_inidces_all[2], f2)
        forces_acc3 = self.computeAccumulatedForceFromId(accumulate_inidces_all[3], f3)
        forces_acc = forces_acc0or4 + forces_acc1 + forces_acc2 + forces_acc3
        return forces_acc

    def computeCollisionIndicesAndVelocities(self, current_pos, step):
        indices_bdry, vel_bdry = self.bc.computeCollisionIndices(current_pos, step)
        return indices_bdry, vel_bdry

    def getResidual(self, xhat, decoder):
        pass
    
    def getJacobian(self, xhat, decoder):
        lbllength = xhat.size(2)
        xhat_all = xhat.expand(xhat.size(0), self.x.size(1), xhat.size(2))
        x = torch.cat((xhat_all, self.x), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        jacobian_part = decoder.jacobianPart(x, lbllength, 'fir')
        jacobian_part = jacobian_part.view(-1, jacobian_part.size(2))
        return jacobian_part
    
    # confirmed to get the same result as the lumped mass from bartels
    def computeMass(self, rho, vertices, tets, volumes):
        mass = torch.zeros(vertices.size(0), 1).type_as(vertices)
        rho = 1000
        for idx, tet in enumerate(tets):
            m_ver = rho * volumes[idx,0] / 4.
            mass[tet[0]] += m_ver
            mass[tet[1]] += m_ver
            mass[tet[2]] += m_ver
            mass[tet[3]] += m_ver
        return mass

    def initialSetup(self, xhat, decoder):
        self.readMeshH5(self.mesh_file)
        
        if hasattr(self, 'vis_mesh_file'):
            mesh_h5 = MeshH5()
            mesh_h5.read(self.vis_mesh_file)
            x = mesh_h5.x
            self.vis_faces = mesh_h5.faces

            self.vis_x = torch.from_numpy(x).to(self.x.device)
            self.vis_x = self.vis_x.view(1, self.vis_x.size(0), self.vis_x.size(1))
        
        self.updateState(xhat, decoder)
        if self.dis_or_pos == 'pos':
            self.initial_offset = self.x-self.q if self.exact_ini else torch.zeros_like(self.q)
        elif self.dis_or_pos == 'dis':
            self.initial_offset = -self.q if self.exact_ini else torch.zeros_like(self.q)
    
    def readMeshH5(self, mesh_file):
        mesh_h5 = MeshH5()
        mesh_h5.read(mesh_file)

        x = mesh_h5.x
        rho = 1000.
        masses = rho*mesh_h5.masses
        tets = mesh_h5.tets

        self.faces = mesh_h5.faces

        self.x = torch.from_numpy(x).to(self.device)
        self.tets = torch.from_numpy(tets).to(self.x.device)

        self.x = self.x.view(1, self.x.size(0), self.x.size(1))
        self.x = self.x[0:1, :masses.shape[0], :]
        self.v = torch.zeros(self.x.size(1), self.sim_dim).to(self.x.device)
        
        self.volume = torch.from_numpy(mesh_h5.volume).to(self.x.device)
        self.Dm_inv = torch.from_numpy(mesh_h5.Dm_inv).to(self.x.device)
        self.mass = torch.from_numpy(masses).to(self.x.device)
        self.mg = self.mass * self.gravity

        self.accumulate_inidces_all = []
        self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[0]).to(self.x.device))
        self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[1]).to(self.x.device))
        self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[2]).to(self.x.device))
        self.accumulate_inidces_all.append(torch.from_numpy(mesh_h5.accumulate_inidces_all[3]).to(self.x.device))
    
    def updateVTrialOriginalSample(self, sample_point):
        Ds = computeDifference(self.getCurrentPosSampleAll(sample_point), sample_point.tets_indices_all)
        def_grads = Ds@self.Dm_inv_sampled
        self.constitutive_law.update(def_grads)
        self.H_density = -self.constitutive_law.first_piola@torch.permute(self.Dm_inv_sampled, (0, 2, 1))
        self.H = self.volume[sample_point.indices_tets, :, :]*self.H_density
        f1 = self.H[:,:,0]
        f2 = self.H[:,:,1]
        f3 = self.H[:,:,2]
        self.f0or4 = -f1-f2-f3

        # accumulate nodal forces
        self.force_per_vertex = self.accumulateForceBatch(self.f0or4, f1, f2, f3, sample_point.accumulate_inidces_all)

        dmom_int = self.dt*self.force_per_vertex

        sample_mass = self.mass[sample_point.indices, :]

        self.dv_int = torch.div(dmom_int, sample_mass)
        mom = sample_mass * self.v_sample + dmom_int + self.dt * self.mg[sample_point.indices, :]

        damping_force = -self.beta_damping * sample_mass * self.v_sample
        dmom_damp = self.dt*damping_force
        mom += dmom_damp

        self.v_trial = torch.div(mom, sample_mass)
        self.q_sample_prev = self.q_sample.clone()
    
    def getResidualSample(self, xhat, decoder, sample_point, step):
        self.updateVTrialOriginalSample(sample_point)
        
        if self.bc:
            current_pos = self.getCurrentPosSample(sample_point)
            self.indices_bdry, self.vel_bdry = self.computeCollisionIndicesAndVelocities(current_pos, step)
            self.v_trial[self.indices_bdry, :] = self.vel_bdry

        return self.v_trial
    
    def getJacobianSample(self, xhat, decoder, sample_point):
        lbllength = xhat.size(2)
        x = self.x[:, sample_point.indices, :]
        xhat_all = xhat.expand(xhat.size(0), x.size(1), xhat.size(2))
        x = torch.cat((xhat_all, x), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))

        jacobian_part = decoder.jacobianPart(x, lbllength, 'fir')
        jac = jacobian_part.view(-1, jacobian_part.size(2))

        return jac
    
    def getJacobianSampleProj(self, xhat, decoder, sample_point):
        return self.jac_sample
    
    def updateStateSampleHelper(self, xhat, decoder, sample_indices):
        lbllength = xhat.size(2)
        x = self.x[:, sample_indices, :]
        xhat_all = xhat.expand(xhat.size(0), x.size(1), xhat.size(2))
        x = torch.cat((xhat_all, x), 2)
        x = x.view(x.size(0)*x.size(1), x.size(2))
        q = decoder.forward(x)
        q = q.view(1, q.size(0), q.size(1))
        return q

    def updateStateSample(self, xhat, decoder, sample_point):
        q = self.updateStateSampleHelper(xhat, decoder, sample_point.indices)
        self.q_sample = q
        return q
    
    def updateStateSampleAll(self, xhat, decoder, sample_point, vhat=None):
        # always use the previous xhat's jacobian to update the velocity
        with self.timer.child('Projection').child('update state sample').child('grad'):
            # update velocity
            if vhat is not None:
                vel = self.jac_sample @ vhat.view(-1,1)
                self.v_sample = vel.view(-1, self.sim_dim)
        
        with self.timer.child('Projection').child('update state sample').child('forward'):
            lbllength = xhat.size(2)
            x = self.x[:, sample_point.indices_all, :]
            xhat_all = xhat.expand(xhat.size(0), x.size(1), xhat.size(2))
            x = torch.cat((xhat_all, x), 2)
            x = x.view(x.size(0)*x.size(1), x.size(2))

            jacobian_part, q = decoder.jacobianPartAndFunc(x, lbllength, 'fir')
            jac = jacobian_part.view(-1, jacobian_part.size(2))
            
            self.q_sample_all = q
            self.q_sample = q[:, 0:sample_point.indices.size(0), :]
            self.jac_sample = jac[0:3*sample_point.indices.size(0), :]
        
        return q
    
    # this function updates the sample indices using tets information, e.g., getting the facilitating vertices
    def updateSampleIndicesWithProblemSpeficifInfo(self, sample_point):
        # obtain sample tets (indices and their actual contents)
        decision = torch.zeros(self.x.size(1))
        decision[sample_point.indices] = 1 # set useful indices to 1
        decision_all = decision[self.tets]
        decision_all_sum = torch.sum(decision_all,dim=1) # collapse the second dim
        sample_point.indices_tets = decision_all_sum.nonzero().view(-1).to(self.tets.device) # non zero terms
        tets_sampled = self.tets[sample_point.indices_tets, :]
        if hasattr(self, 'Dm_inv'):
            self.Dm_inv_sampled = self.Dm_inv[sample_point.indices_tets, :, :]
        
        # identify facilitate indices
        all_vertices = torch.unique(tets_sampled)
        combined = torch.cat((all_vertices, sample_point.indices))
        uniques, counts = combined.unique(return_counts=True)
        sample_point.indices_facilitate = uniques[counts == 1]
        sample_point.indices_all = torch.cat((sample_point.indices, sample_point.indices_facilitate))

        if sample_point.indices_facilitate.size(0) == 0: # no facilitate points; meaning all indices are included
            sample_point.tets_indices_all = tets_sampled
            if hasattr(self, 'accumulate_inidces_all'):
                sample_point.accumulate_inidces_all = self.accumulate_inidces_all
        else:
            sample_point.tets_indices_all = NewTensor2IndicesOfBaseTensor(tets_sampled, sample_point.indices_all)

            if hasattr(self, 'accumulate_inidces_all'):
                indices_tets_dummy = torch.Tensor([self.tets.size(0)]).type_as(sample_point.indices_tets)
                indices_tets_icl_dummy = torch.cat((sample_point.indices_tets, indices_tets_dummy))

                sample_point.accumulate_inidces_all = []
                for indices_id in self.accumulate_inidces_all:
                    indices_id = indices_id[sample_point.indices, :] # need only indices that participate in projection; do not need the facilitate indices
                    indices_id = NewTensor2IndicesOfBaseTensor(indices_id, indices_tets_icl_dummy)
                    sample_point.accumulate_inidces_all.append(indices_id)
        