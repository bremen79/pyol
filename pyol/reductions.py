import numpy as np

class CoordinateWise:
    def __init__(self, d, base_learners):
        self.d = d
        self.base_learners = base_learners
        self.name_base_learners = ""
        for alg in base_learners:
            self.name_base_learners += alg.get_name() + "; "
        self.name_base_learners = self.name_base_learners[:-2]
        
    def update(self, grad):
        for i,bl in enumerate(self.base_learners):
            bl.update(grad[i])
    
    def get_x(self):
        x = np.zeros(self.d)
        for i,bl in enumerate(self.base_learners):
            x[i] = bl.get_x()[0]
        return x
    
    def get_name(self):
        return "Coordinate Wise: "+self.name_base_learners
