import torch
from timer_cm import Timer

class NonlinearSolver():
    def __init__(self, problem, decoder, diff_threshold, step_size_threshold):
        self.problem = problem
        self.decoder = decoder
        self.diff_threshold = diff_threshold
        self.step_size_threshold = step_size_threshold
        self.timer = Timer('NonlinearSolver')
    
    def computeLoss(self, r):
        element_wise_square= torch.mul(r, r)
        r_square = torch.sum(element_wise_square, 1)
        loss = torch.sum(r_square)
        return loss
    
    def computeLossAndR(self, xhat, q_target, sample_point):
        with self.timer.child('Projection').child('nonlinear solve').child('computeLossAndR'):
            q = self.problem.updateStateSample(xhat, self.decoder, sample_point)
            r = (q - q_target).view(-1, 1)
            loss = self.computeLoss(r)
            return loss, r

    def solve(self, xhat, q_target, sample_point, step_size = 1, max_iters = 5):
        xhat_new = xhat.detach().clone()
        xhat = xhat.detach().clone()
        xhat = xhat - torch.ones_like(xhat) # dummy 1
        
        xhat = xhat.type_as(q_target)
        xhat_new = xhat_new.type_as(q_target)
        steps = 0
        diff = xhat_new - xhat
        
        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
        while torch.linalg.norm(diff) > self.diff_threshold and steps < max_iters:
            xhat = xhat_new.detach().clone()
            with self.timer.child('Projection').child('nonlinear solve').child('computeJacobian'):
                jac = self.problem.getJacobianSample(xhat, self.decoder, sample_point)
            
            loss, r = self.computeLossAndR(xhat, q_target, sample_point)
            
            mat = torch.inverse(torch.matmul(jac.transpose(1, 0), jac))
            gradf = torch.matmul(jac.transpose(1, 0), r)
            descent_direction_col = -torch.matmul(mat, gradf)
            descent_direction = descent_direction_col.view_as(xhat)

            # line search
            alpha = 0.25
            beta = 0.5

            t = step_size
            ln_search_max_iter = 10
            backtracking_count = 0

            while ln_search_max_iter > 0:
                backtracking_count += 1
                xhat_search = xhat + t * descent_direction
                loss_search, r = self.computeLossAndR(xhat_search, q_target, sample_point)
                dot_result = torch.dot(gradf.view(-1), descent_direction_col.view(-1))
                critical_value = loss_search - (loss + alpha * t * dot_result)
                if critical_value.item() > 0.0 and backtracking_count < ln_search_max_iter:
                    t *= beta
                else:
                    if critical_value.item() > 0.0:
                        t *= beta
                    break
            max_step_size = (t * descent_direction).abs().max().item()
            # print('backtracking t: ', t)
            # print('backtracking count: ', backtracking_count)
            # print('max_step_size: ', max_step_size)
            if max_step_size < self.step_size_threshold:
                break
            xhat_new = xhat + t * descent_direction
            steps += 1
            diff = xhat_new - xhat
        
        xhat = xhat_new.detach().clone()
        loss, r = self.computeLossAndR(xhat, q_target, sample_point)

        print('GN, steps={steps}, loss={loss}'.format(steps=steps, loss=loss.item()))

        return xhat