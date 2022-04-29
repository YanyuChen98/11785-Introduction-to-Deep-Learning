import numpy as np

from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activation import ReLU

class MLP0:

    def __init__(self, debug = False):
    
        self.layers = [ Linear(2, 3, debug=True) ]
        self.f      = [ ReLU() ]

        self.debug = debug

    def forward(self, A0):

        Z0 = self.layers[0].forward(A0)
        A1 = self.f[0].forward(Z0)

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1
        
        return self.A1

    def backward(self, dLdA1):
    
        dA1dZ0 = self.f[0].backward()
        dLdZ0  = dLdA1 * dA1dZ0 # TODO
        dLdA0  = self.layers[0].backward(dLdZ0) # TODO

        if self.debug:

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
        
        return None
        
class MLP1:

    def __init__(self, debug = False):
    
        self.layers = [ Linear(2, 3, debug=True),
                        Linear(3, 2, debug=True) ]
        self.f      = [ ReLU(),
                        ReLU() ]

        self.debug = debug

    def forward(self, A0):
    
        Z0 = self.layers[0].forward(A0)
        A1 = self.f[0].forward(Z0) # TODO
    
        Z1 = self.layers[1].forward(A1) # TODO
        A2 = self.f[1].forward(Z1) # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2
        
        return A2

    def backward(self, dLdA2):

        dA2dZ1 = self.f[1].backward() # TODO
        dLdZ1  = dLdA2 * dA2dZ1 # TODO
        dLdA1  = self.layers[1].backward(dLdZ1) # TODO
    
        dA1dZ0 = self.f[0].backward() # TODO
        dLdZ0  = dA1dZ0 * dLdA1 # TODO
        dLdA0  = self.layers[0].backward(dLdZ0) # TODO

        if self.debug:

            self.dA2dZ1 = dA2dZ1
            self.dLdZ1  = dLdZ1
            self.dLdA1  = dLdA1

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0  = dLdZ0
            self.dLdA0  = dLdA0
        
        return None

class MLP4:
    def __init__(self, debug=False):
        
        # Hidden Layers
        self.layers = [
            Linear(2, 4, debug = True),
            Linear(4, 8, debug = True),
            Linear(8, 8, debug = True),
            Linear(8, 4, debug = True),
            Linear(4, 2, debug = True)]

        # Activations
        self.f = [
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU(),
            ReLU()]

        self.debug = debug

    def forward(self, A):

        if self.debug:

            self.Z = []
            self.A = [ A ]

        L = len(self.layers)

        for i in range(L):

            Z = self.layers[i].forward(A) # TODO
            A = self.f[i].forward(Z) # TODO

            if self.debug:

                self.Z.append(Z)
                self.A.append(A)

        return self.A[-1]

    def backward(self, dLdA):

        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [ dLdA ]

        L = len(self.layers)

        for i in reversed(range(L)):

            dAdZ = self.f[i].backward() # TODO
            dLdZ = dLdA * dAdZ # TODO
            dLdA = self.layers[i].backward(dLdZ) # TODO

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA

        return None
        