import numpy as np

class ParabolaSequence():
    def __init__(self, data:np.array, segAmount: int):
        self.data:np.array = data
        self.segAmount = segAmount
    
    def X0(self):
        return self.data[0].item()
    def Y0(self):
        return self.data[2].item()
    def Z0(self):
        return self.data[4].item()
    def Vx(self):
        return self.data[1].item()
    def Vy(self):
        return self.data[3].item()
    def Vz(self):
        return self.data[5].item()
    def pAmount(self):
        return self.segAmount
    

    

