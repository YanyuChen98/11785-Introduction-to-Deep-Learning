import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:
            self.Z = Z
            self.N = len(self.Z)  # TODO

            self.M = self.running_M  # TODO
            self.V = self.running_V  # TODO
            self.NZ = (self.Z - self.M) / np.sqrt(self.V + self.eps)  # TODO
            self.BZ = self.BW * self.NZ + self.Bb  # TODO
            return self.BZ
            
        self.Z         = Z
        self.N         = len(self.Z) # TODO
        
        self.M         = np.mean(self.Z, axis = 0) # TODO
        self.V         = np.var(self.Z, axis = 0) # TODO
        self.NZ        = (self.Z - self.M)/np.sqrt(self.V + self.eps)# TODO
        self.BZ        = self.BW * self.NZ + self.Bb # TODO
        
        self.running_M = self.alpha*self.M + (1 - self.alpha)*self.M # TODO
        self.running_V = self.alpha*self.V + (1 - self.alpha)*self.V # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum((dLdBZ * self.NZ), axis = 0) # TODO
        self.dLdBb  = np.sum(dLdBZ, axis = 0) # TODO
        
        dLdNZ       = dLdBZ * self.BW # TODO
        dLdV        = (-0.5)*np.sum(dLdNZ*(self.Z-self.M)*np.power(self.V+self.eps, -1.5), axis = 0) # TODO
        dLdM        = -np.sum(dLdNZ*np.power(self.V+self.eps, -0.5), axis = 0)-(2/self.N)*dLdV*(np.sum(self.Z-self.M, axis = 0)) # TODO
        
        dLdZ        = dLdNZ*(np.power(self.V+self.eps, -0.5)) + dLdV * 2/self.N * (self.Z-self.M) + dLdM/self.N # TODO
        
        return  dLdZ
