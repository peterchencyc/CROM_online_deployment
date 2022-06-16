import numpy as np

def generateSteps(proj_steps):
    accumulated_proj_steps = [0]
    accumulated_proj_steps.extend(np.cumsum(proj_steps))
    return accumulated_proj_steps

class ProjTypeScheduler(object):
    def __init__(self, proj_type, proj_steps, nonlinear_initial_guess):
        assert(len(proj_type)==len(proj_steps))
        assert(len(proj_type)==len(nonlinear_initial_guess))
        if len(proj_type) == 1:
            self.proj_type = proj_type[0]
            self.nonlinear_initial_guess = nonlinear_initial_guess[0]
            self.accumulated_proj_steps = None
        else:
            self.proj_type = proj_type
            self.nonlinear_initial_guess = nonlinear_initial_guess
            self.accumulated_proj_steps = generateSteps(proj_steps)
    
    def getProjType(self, step):
        if self.accumulated_proj_steps is None:
            return self.proj_type, self.nonlinear_initial_guess
        else:
            for idx in range(len(self.accumulated_proj_steps)-1):
                do = self.accumulated_proj_steps[idx]
                up = self.accumulated_proj_steps[idx+1]
                if do < step <= up:
                    return self.proj_type[idx], self.nonlinear_initial_guess[idx]

