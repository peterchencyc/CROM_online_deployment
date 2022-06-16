from ObjLoader import *
import h5py
import os
import numpy as np

class SimulationState(object):
    def __init__(self, filename, readfile=True, input_x=None, input_q=None, input_t=None, input_faces=None):
        self.filename = filename
        if readfile:
            with h5py.File(self.filename, 'r') as h5_file:
                self.x = h5_file['/x'][:]
                self.x = self.x.T
                self.q = h5_file['/q'][:]
                self.q = self.q.T
                self.t = h5_file['/time'][0][0]
                if '/label' in h5_file:
                    self.label = h5_file['/label'][:]
                if '/faces' in h5_file:
                    self.faces = h5_file['/faces'][:]
                    self.faces = self.faces.T
                if '/masses' in h5_file:
                    self.masses = h5_file['/masses'][:]
                    self.masses = self.masses.T
                if '/tets' in h5_file:
                    self.tets = h5_file['/tets'][:]
                    self.tets = self.tets.T
                    # default to torchfem
                    # self.tets -= 1 # convert from matlab convention
                    self.tets = self.tets.astype(np.int)
                if '/f_tensor' in h5_file:
                    f_tensor_col_major = h5_file['/f_tensor'][:]
                    f_tensor_col_major = f_tensor_col_major.T
                    self.f_tensor = f_tensor_col_major.reshape(
                        -1, 3, 3).transpose(0, 2, 1)
        else:
            if input_x is None:
                print('must provide a x if not reading from file')
                exit()
            if input_q is None:
                print('must provide a q if not reading from file')
                exit()
            if input_t is None:
                print('must provide a t if not reading from file')
                exit()
            self.x = input_x
            self.q = input_q
            self.t = input_t
            if input_faces is not None:
                if input_faces.shape[1]==3: #actually contains faces instead placeholder in the shape of [1,1]
                    self.faces = input_faces
    
    def write_to_file(self, filename=None):
        if filename:
            self.filename = filename
        print('writng sim state: ', self.filename)
        dirname = os.path.dirname(self.filename)
        os.umask(0)
        os.makedirs(dirname, 0o777, exist_ok=True)
        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset("x", data=self.x.T)
            dset = h5_file.create_dataset("q", data=self.q.T)
            self.t = self.t.astype(np.float64)
            self.t = self.t.reshape(1,1)
            dset = h5_file.create_dataset("time", data=self.t)
            if hasattr(self, 'label'):
                if self.label is not None:
                    label = self.label.reshape(-1, 1)
                    label = label.astype(np.float64)
                    dset = h5_file.create_dataset("label", data=label)
            if hasattr(self, 'masses'):
                dset = h5_file.create_dataset("masses", data=self.masses.T)
            if hasattr(self, 'faces'):
                dset = h5_file.create_dataset("faces", data=self.faces.T)
            if hasattr(self, 'tets'):
                # default to torchfem
                # self.tets += 1
                dset = h5_file.create_dataset("tets", data=self.tets.T)
            if hasattr(self, 'f_tensor'):
                f_tensor_col_major = self.f_tensor.transpose(0, 2, 1).reshape(
                    -1, 9)
                dset = h5_file.create_dataset("/f_tensor", data=f_tensor_col_major.T)
                
        if hasattr(self, 'faces'):
            filename_obj = os.path.splitext(self.filename)[0]+'.obj'
            print('writng sim state obj: ', filename_obj)
            obj_loader = ObjLoader()
            obj_loader.vertices = self.q
            obj_loader.faces = self.faces
            obj_loader.export(filename_obj)


