import numpy as np
import math

class LearningRateSqrtTime:
    def __init__(self, eta0=1):
        self.time=1
        self.eta0=eta0
    
    def update(self, grad):
        self.time += 1
    
    def get_lr(self):
        return self.eta0/math.sqrt(self.time)
    
    def get_name(self):
        return "LR: eta0/sqrt(t)"


class LearningRateSqrtSumSquareGrads:
    def __init__(self, eta0=1, eps=np.finfo(np.float32).eps):
        self.sum_g2 = eps
        self.eta0 = eta0
    
    def update(self, grad):
        self.sum_g2 += np.sum(grad*grad)
    
    def get_lr(self):
        return self.eta0/math.sqrt(self.sum_g2)
    
    def get_name(self):
        return "LR: eta0/sqrt(sum_t ||g_t||^2_2)"


class LearningRateConstant:
    def __init__(self, eta0=1):
        self.eta0=eta0
    
    def update(self, grad):
        pass
    
    def get_lr(self):
        return self.eta0
    
    def get_name(self):
        return "LR: eta0"


class LearningRateMaxNormGrad:
    def __init__(self, eta0=1, norm=2):
        self.eta0=eta0
        self.max_norm_grad=np.finfo(np.float32).eps
        self.norm=norm
    
    def update(self, grad):
        self.max_norm_grad=max(self.max_norm_grad,np.linalg.norm(grad,ord=self.norm))
    
    def get_lr(self):
        return self.eta0/self.max_norm_grad
    
    def get_name(self):
        return "LR: eta0/max_t ||g_t||_2"
