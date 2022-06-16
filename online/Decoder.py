from torch.autograd import grad
import torch
from torch import linalg as LA


class Decoder(object):
    def __init__(self, network, md, netfuncgrad):
        network.eval()
        for param in network.parameters():
            param.requires_grad = False
        netfuncgrad.eval()
        for param in netfuncgrad.parameters():
            param.requires_grad = False
        self.network = network
        self.md = md
        self.netfuncgrad = netfuncgrad
    
    def forward(self, x):
        with torch.inference_mode():
            return self.network(x)

    def getPartGradx(self, x, part_dim, which):
        x = x.detach()
        x_first = x[:, 0:part_dim]
        x_second = x[:, part_dim:x.size(1)]
        if which == 'fir':
            x_grad = x_first
            x_grad.requires_grad_(True)
            x = torch.cat((x_grad, x_second), 1)
        elif which == 'sec':
            x_grad = x_second
            x_grad.requires_grad_(True)
            x = torch.cat((x_first, x_grad), 1)
        else:
            exit('invalid which')
        return x_grad, x
    
    def jacobianPartAndFunc(self, x, part_dim, which):
        if self.netfuncgrad:
            with torch.inference_mode():
                grad_val, y = self.netfuncgrad(x)
                if which == 'fir':
                    grad_val = grad_val[:, :, 0:part_dim]
                elif which == 'sec':
                    grad_val = grad_val[:, :, part_dim:x.size(1)]
                jacobian = grad_val.view(-1, 1, grad_val.size(2))
            y = y.view(1, y.size(0), y.size(1))
            return jacobian, y
        else:
            exit('jacobianPartAndFunc only works with netfuncgrad')
    
    def jacobianPart(self, x, part_dim, which):
        if self.netfuncgrad:
            with torch.inference_mode():
                grad_val, y = self.netfuncgrad(x)
                if which == 'fir':
                    grad_val = grad_val[:, :, 0:part_dim]
                elif which == 'sec':
                    grad_val = grad_val[:, :, part_dim:x.size(1)]
                jacobian = grad_val.view(-1, 1, grad_val.size(2))
        else:
            x_grad, x = self.getPartGradx(x, part_dim, which)
            outputs = self.network(x)
            output_dim = outputs.size(1)
            jacobian = None
            for dim in range(output_dim):
                dy_dx = grad(outputs=outputs[:, dim], inputs=x_grad, grad_outputs=torch.ones_like(outputs[:, dim]),
                        retain_graph=True, create_graph=False)[0]
                dy_dx = dy_dx.view(dy_dx.size(0), 1, dy_dx.size(1))
                # dy_dx size: [num, 1, input_dim], i.e., gradient of a scalar is a row vector
                if jacobian is None:
                    jacobian = dy_dx
                else:
                    jacobian = torch.cat([jacobian, dy_dx], 1)
            # jacobian size: [num, output_dim, input_dim]; this is consistent with continuum mechanics convetion, e.g., F_{ij} = \frac{dx_i}{dX_j}
            jacobian = jacobian.view(-1, 1, jacobian.size(2)).detach()
            # jacobian size: [num * output_dim, 1, input_dim]; 
            # dx^1_1dX
            # dx^1_2dX
            # dx^1_3dX
            # dx^2_1dX
            # dx^2_2dX
            # dx^2_3dX
            # ...

        return jacobian
    
    def hessianPart(self, x, part_dim, which):
        x_grad, x = self.getPartGradx(x, part_dim, which)
        outputs = self.network(x)
        output_dim = outputs.size(1)
        jacobian = None
        for dim in range(output_dim):
            dy_dx = grad(outputs=outputs[:, dim], inputs=x_grad, grad_outputs=torch.ones_like(outputs[:, dim]),
                    retain_graph=True, create_graph=True)[0]
            dy_dx = dy_dx.view(dy_dx.size(0), 1, dy_dx.size(1))
            # dy_dx size: [num, 1, input_dim], i.e., gradient of a scalar is a row vector
            if jacobian is None:
                jacobian = dy_dx
            else:
                jacobian = torch.cat([jacobian, dy_dx], 1)
        # jacobian size: [num, output_dim, input_dim]; this is consistent with continuum mechanics convetion, e.g., F_{ij} = \frac{dx_i}{dX_j}

        i_dim = jacobian.size(1) # output_dim
        j_dim = jacobian.size(2) # input_dim

        hessian = None
        for i in range(i_dim):
            hessian_one = None
            for j in range(j_dim):
                dy_dx2 = grad(outputs=jacobian[:, i, j], inputs=x_grad, grad_outputs=torch.ones_like(jacobian[:, i, j]),
                    retain_graph=True, create_graph=False)[0]
                dy_dx2 = dy_dx2.view(dy_dx2.size(0), 1, 1, dy_dx2.size(1))
                # size: [num, 1, 1, input_dim]
                if hessian_one is None:
                    hessian_one = dy_dx2
                else:
                    hessian_one = torch.cat([hessian_one, dy_dx2], 2)
            # hessian_one size: [num, 1, input_dim, input_dim]
            if hessian is None:
                hessian = hessian_one
            else:
                hessian = torch.cat([hessian, hessian_one], 1)
        # hessian size: [num, output_dim, input_dim, input_dim]; this is consistent with continuum mechanics convetion, e.g., F_{ij,k} = \frac{dF_{ij}}{dX_k}
        return jacobian.detach(), hessian.detach()