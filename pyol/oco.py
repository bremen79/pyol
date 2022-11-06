import numpy as np
import math
from .learning_rates import LearningRateSqrtTime


class L2OptimisticFTRL:
    def __init__(self, d, *, x0=None, lr_handler=None, proj_handler=None, hint_handler=None ):
        if x0:
            self.x0 = x0
        else:
            self.x0 = np.zeros(d)
        self.theta = np.zeros(d)
        self.proj_handler = proj_handler
        self.d = d
        self.hint=0
        if hint_handler:
            self.hint_handler=hint_handler
        else:
            self.hint_handler=None
        if lr_handler:
            self.lr_handler=lr_handler
        else:
            self.lr_handler=LearningRateSqrtTime()
            
    def predict(self):
        return self.get_x()
    
    def update(self, grad):
        self.theta += grad
        self.lr_handler.update(grad)
        if self.hint_handler:
            self.hint_handler.update(grad)
        
    def get_x(self):
        if self.hint_handler:
            self.hint=self.hint_handler.get_hint()
        tmp=-(self.theta+self.hint)*self.lr_handler.get_lr()+self.x0
        if self.proj_handler:
            tmp = self.proj_handler.proj(tmp)
        return tmp

    def get_name(self):
        tmp = "L2"
        if self.hint_handler:
            tmp += " Optimistic"
        tmp += " FTRL, "
        return tmp+self.lr_handler.get_name()


class L2OptimisticOMD:
    def __init__(self, d, *, x0=None, lr_handler=None, proj_handler=None, hint_handler=None ):
        self.d = d
        if x0:
            self.x = x0
        else:
            self.x = np.zeros(d)
        self.proj_handler = proj_handler
        if self.proj_handler:
            self.x=proj_handler.proj(self.x)
        if hint_handler:
            self.hint_handler=hint_handler
        else:
            self.hint_handler=None
        if lr_handler:
            self.lr_handler=lr_handler
        else:
            self.lr_handler=LearningRateSqrtTime()
        self.hint=0
        self.previous_hint=0

    def predict(self):
        return self.x
    
    def update(self, grad):
        self.lr_handler.update(grad)
        if self.hint_handler:
            self.previous_hint=self.hint
            self.hint_handler.update(grad)
            self.hint=self.hint_handler.get_hint()
        self.x = self.x - self.lr_handler.get_lr()*(grad-self.previous_hint+self.hint)
        if self.proj_handler:
            self.x=self.proj_handler.proj(self.x)
        
    def get_x(self):
        return self.x
    
    def get_name(self):
        tmp = "L2"
        if self.hint_handler:
            tmp += " Optimistic"
        tmp += " OMD, "
        return tmp+self.lr_handler.get_name()


class KT:
    def __init__(self, d, wealth0=1):
        self.theta = np.zeros(d)
        self.d = d
        self.wealth0 = wealth0
        self.wealth = wealth0
        self.time = 1
        
    def predict(self):
        return self.get_x()
    
    def update(self, grad):
        self.theta += grad
        self.wealth += np.dot(-grad, self.get_x())
        self.time +=1
        
    def get_x(self):
        return -self.theta/self.time*self.wealth
    
    def get_name(self):
        return "KT: Wealth0="+str(self.wealth0)


        

class EntropyOptimisticFTRL:
    def __init__(self, d, *, lr_handler=None, hint_handler=None ):
        self.theta = np.zeros(d)
        if lr_handler:
            self.lr_handler = lr_handler
        else:
            self.lr_handler = LearningRateSqrtTime(math.sqrt(math.log(d)))
        self.d = d
        self.hint=0
        if hint_handler:
            self.hint_handler=hint_handler
        else:
            self.hint_handler=None

    def predict(self):
        x=self.get_x()
        return np.random.choice(self.d, p=self.get_x())

    def update(self, grad):
        self.theta = self.theta + grad
        self.lr_handler.update(grad)
        if self.hint_handler:
            self.hint_handler.update(grad)
        
    def get_x(self):
        if self.hint_handler:
            self.hint=self.hint_handler.get_hint()
        tmp=self.theta+self.hint
        weights = np.exp(- self.lr_handler.get_lr() * (tmp-np.min(tmp)))
        return weights / np.sum(weights)

    def get_name(self):
        tmp = "Entropy"
        if self.hint_handler:
            tmp += " Optimistic"
        tmp += " FTRL, "
        return tmp+self.lr_handler.get_name()


class KTExperts:
    def __init__(self, d):
        self.theta = np.zeros(d)
        self.wealth = np.full(d,1)
        self.time = 1
        self.d = d

    def predict(self):
        x=self.get_x()
        return np.random.choice(self.d, p=self.get_x())

    def update(self, grad):
        weights = self.theta*self.wealth/(self.time)
        trunc_weights=np.maximum(weights,0)
        sum_weights = np.sum(trunc_weights,0)
        x = trunc_weights / sum_weights if sum_weights>0 else np.full(self.d,1/self.d)
        r = np.dot(grad,x)-grad
        idx_neg=(weights<=0)
        r[idx_neg]=np.maximum(r[idx_neg],0)
        self.theta = self.theta + r
        self.wealth = self.wealth + weights*r
        self.time = self.time + 1
        
    def get_x(self):
        weights = np.maximum(self.theta,0)*self.wealth/self.time
        sum_weights = np.sum(weights)
        return weights / sum_weights if sum_weights>0 else np.full(self.d,1/self.d)
    
    def get_name(self):
        return "KT for experts"

class RM:
    def __init__(self, d):
        self.theta = np.zeros(d)
        self.d = d

    def predict(self):
        return np.random.choice(self.d, p=self.get_x())

    def update(self, grad):
        x = self.get_x()
        r = np.dot(grad,x) - grad
        self.theta = self.theta + r
        
    def get_x(self):
        weights = np.maximum(self.theta,0)
        sum_weights = np.sum(weights)
        return weights / sum_weights if sum_weights>0 else np.full(self.d,1/self.d)
    
    def get_name(self):
        return "RM"

class RMPlus:
    def __init__(self, d, hint_handler=None):
        self.theta = np.zeros(d)
        self.d=d
        if hint_handler:
            self.hint_handler=hint_handler
        else:
            self.hint_handler=None
        self.hint=0
        self.previous_hint=0

    def predict(self):
        return np.random.choice(self.d, p=self.get_x())

    def update(self, grad):
        x=self.get_x()
        r = grad-np.dot(grad,x)
        if self.hint_handler:
            self.previous_hint=self.hint
            self.hint_handler.update(r)
            self.hint=self.hint_handler.get_hint()
        self.theta = np.maximum(self.theta - r + self.previous_hint - self.hint,0)
        
    def get_x(self):
        sum_weights = np.sum(self.theta)
        return self.theta / sum_weights if sum_weights>0 else np.full(self.d,1/self.d)
    
    def get_name(self):
        tmp = "RM+"
        if self.hint_handler:
            tmp += " Optimistic"
        return tmp


class AdaHedge:
    def __init__(self, d, alpha=None):
        self.theta = np.zeros(d)
        self.d=d
        self.alpha2 = alpha*alpha if alpha else math.log(d)
        self.l=0

    def predict(self):
        return np.random.choice(self.d, p=self.get_x())

    def update(self, grad):
        x=self.get_x()
        self.theta += grad
        if self.l==0:
            delta=np.dot(grad,x)-min(grad)
        else:
            delta=self.l*math.log(np.sum(x*np.exp(-grad/self.l)))+np.dot(grad,x)
        self.l=self.l+delta/self.alpha2
        
    def get_x(self):
        if self.l>0:
            weights = np.exp(-(self.theta-min(self.theta))/self.l)
        else:
            weights = np.full(self.d,1)
        return weights / np.sum(weights)
    
    def get_name(self):
        return "AdaHedge: alpha^2="+str(self.alpha2)

