import numpy as np
import math

class ProjL2Ball:
    def __init__(self, R=1):
        self.R=R
        
    def proj(self, x):
        norm=math.sqrt(np.sum(x*x))
        if norm>0:
            return x*min(1,self.R/norm)
        else:
            return x*0


class ProjSimplex:
    def __init__(self):
        pass
        
    # https://lcondat.github.io/software.html
    def proj(self, x):
        return np.maximum(x-np.max((np.cumsum(np.sort(x)[::-1])-1)/(np.arange(1,len(x)+1))),0)
