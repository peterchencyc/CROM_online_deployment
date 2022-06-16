import os
import torch
import h5py
import torch.linalg as LA

class MeshH5(object):
    def __init__(self):
        pass
        
    def read(self, filename):
        with h5py.File(filename, 'r') as h5_file:
            self.x = h5_file['/x'][:]
            self.tets = h5_file['/tets'][:]
            self.faces = h5_file['/faces'][:]
            self.masses = h5_file['/masses'][:]
            self.volume = h5_file['/volume'][:]
            self.Dm_inv = h5_file['/Dm_inv'][:]
            self.accumulate_inidces_all = []
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces0'][:])
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces1'][:])
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces2'][:])
            self.accumulate_inidces_all.append(h5_file['/accumulate_inidces3'][:])