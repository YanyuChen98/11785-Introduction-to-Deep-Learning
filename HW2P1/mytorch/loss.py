import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        Ones_C = np.ones((C, 1), dtype="f")
        Ones_N = np.ones((N, 1), dtype="f")
        se     = (A-Y) * (A-Y)
        sse    = np.dot(np.dot(Ones_N.T, se), Ones_C)
        mse    = sse/(np.dot(N, C))

        return mse
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A   = A
        self.Y   = Y
        N        = A.shape[0]
        C        = A.shape[1]
        Ones_C   = np.ones((C, 1), dtype="f")
        Ones_N   = np.ones((N, 1), dtype="f")

        self.softmax     = np.exp(A) / np.dot(np.dot(np.exp(A), Ones_C), Ones_C.T)
        crossentropy     = -Y * np.log(self.softmax)
        sum_crossentropy = np.dot(np.dot(Ones_N.T, crossentropy), Ones_C)
        L = sum_crossentropy / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y
        
        return dLdA
