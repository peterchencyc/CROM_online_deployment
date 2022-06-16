import torch
import torch.linalg as LA
from abc import abstractmethod

class BoundaryConditionHard(object):
    def __init__(self):
        self.collision_objects = []
    
    def append(self, collision_object):
        self.collision_objects.append(collision_object)
    
    def computeCollisionIndices(self, vertices, step):
        indices_all = torch.Tensor([]).view(-1).type_as(vertices)
        vels_all = torch.Tensor([]).view(-1, 3).type_as(vertices)
        for idx, co in enumerate(self.collision_objects):
            indices, vels = co.computeCollisionIndices(vertices, step)
            indices_all = torch.cat((indices_all, indices))
            vels_all = torch.cat((vels_all, vels))
        return indices_all.long(), vels_all
    
    def step(self, dt):
        for idx, co in enumerate(self.collision_objects):
            co.step(dt)

class CollisionObject(object):
    @abstractmethod
    def queryDistNormal(self, position):
        pass

class Plane(CollisionObject):
    def __init__(self, center, normal, radius = None, vel = None, omega = None, activate=None):
        self.center = center
        self.normal = normal
        self.radius = radius
        if vel is None: vel = [0,0,0]
        self.vel = torch.Tensor(vel).to(self.center.device)
        if omega is None: omega = [0,0,0]
        self.omega = torch.Tensor(omega).to(self.center.device).view(-1, 3)
        self.activate = activate


    def queryDistNormal(self, position):
        
        connect = position - self.center
        pen_depth = torch.dot(connect, -self.normal)
        pen_depth = pen_depth if pen_depth>0 else 0 

        if self.radius:
            normal_vec = -pen_depth*self.normal
            lateral_vec = connect - normal_vec
            x_norm = abs(lateral_vec[0])
            z_norm = abs(lateral_vec[2])
            if x_norm.item() > self.radius[0]:
                pen_depth = 0
            if z_norm.item() > self.radius[1]:
                pen_depth = 0

        return pen_depth * self.normal
    
    def computeCollisionIndices(self, vertices, step):
        if self.activate is None or self.activate(step):
            dn = self.computeDn(vertices)
            dn_norm = LA.norm(dn, axis=1)
            condition = dn_norm > 1e-12
            indices = condition.nonzero().view(-1)
            vels = self.vel.view(1, -1)
            vels = vels.expand(indices.size(0), -1)
            vels_rot = self.computeAngularVelocity(vertices[indices, :])
            vels = vels + vels_rot
        else:
            indices = torch.Tensor([]).view(-1).type_as(vertices)
            vels = torch.Tensor([]).view(-1, 3).type_as(vertices)
        return indices, vels
    
    def computeAngularVelocity(self, vertices):
        r = vertices - self.center
        vels_rot = LA.cross(self.omega, r)
        return vels_rot

    def computeDn(self, vertices):
        dn_batch = self.computeDnBatch(vertices)
        return dn_batch

    def computeDnVanila(self, vertices):
        dn = map(self.queryDistNormal, torch.unbind(vertices, 0))
        dn = list(dn)
        dn = torch.stack(dn)
        return dn
    
    def computeDnBatch(self, vertices):
        connect = vertices - self.center
        pen_depth = torch.einsum('bi,i->b', connect, -self.normal)
        pen_depth = torch.where(pen_depth > 0, pen_depth, torch.zeros_like(pen_depth))
        if self.radius:
            normal_vec = torch.einsum('b,i->bi', -pen_depth, self.normal)
            lateral_vec = connect - normal_vec
            lateral_x_norm = lateral_vec[:, 0].abs()
            lateral_z_norm = lateral_vec[:, 2].abs()
            pen_depth = torch.where(lateral_x_norm > self.radius[0], torch.zeros_like(pen_depth), pen_depth)
            pen_depth = torch.where(lateral_z_norm > self.radius[1], torch.zeros_like(pen_depth), pen_depth)
        return torch.einsum('b,i->bi', pen_depth, self.normal)
    
    def step(self, dt):
        self.center += dt*self.vel