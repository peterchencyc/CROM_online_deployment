import torch
import torch.nn as nn
import os

class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key, value in tensor_dict.items():
            setattr(self, key, value)

def writeInitialLabel(lbl, decoder_filename):
    dir_filename_model = os.path.splitext(decoder_filename)[0]
    ini_label_filename = dir_filename_model + '.ini_label'
    ini_label_filename_cpu = dir_filename_model + '_cpu.ini_label'
    assert(lbl.size(0)==1)
    assert(lbl.size(1)==1)
    lbl = lbl[0,0,:].cpu()
    tensor_dict = {'lbl': lbl}
    tensors = TensorContainer(tensor_dict)
    tensors = torch.jit.script(tensors)
    tensors.save(ini_label_filename)
    tensors.save(ini_label_filename_cpu)
    print('initial label file: ', ini_label_filename)