import numpy as np

class HintPreviousGrad:
    def __init__(self, d):
        self.hint=np.zeros(d)
        
    def update(self, grad):
        self.hint=grad
    
    def get_hint(self):
        return self.hint
    
    def get_name(self):
        return "Hints: Prev Grad"


class HintSumNegativePreviousGrad:
    def __init__(self, d):
        self.hint=np.zeros(d)
        
    def update(self, grad):
        self.hint=self.hint-grad
    
    def get_hint(self):
        return self.hint
    
    def get_name(self):
        return "Hints: Prev Grad"
