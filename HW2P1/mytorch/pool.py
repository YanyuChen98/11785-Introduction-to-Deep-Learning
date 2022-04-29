import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # TODO
        self.A = A

        batch_size, in_channels, input_width, input_height = self.A.shape
        output_height = (input_height - self.kernel) + 1
        output_width = (input_width - self.kernel) + 1
        full_arrays = np.zeros([batch_size, in_channels, output_width, output_height])

        for batch in range(batch_size):
            for out_channels in range(in_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        full_arrays[batch, out_channels, j, i] = \
                            max(self.A[batch, out_channels, j:j + self.kernel, i:i + self.kernel].flatten())

        Z = full_arrays  # TODO

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # TODO
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        batch_size, in_channels, input_width, input_height = self.A.shape
        dLdA = np.zeros((batch_size, out_channels, input_width, input_height))

        for batch in range(batch_size):
            for in_channel in range(out_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        temp = self.A[batch, in_channel, i:i+self.kernel, j:j+self.kernel]
                        filter = (temp == np.max(temp))
                        dLdA[batch, in_channel, i:i+self.kernel, j:j+self.kernel]  += \
                            dLdZ[batch, in_channel, i, j] * filter

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # TODO
        self.A = A
        A_copy = A.copy()
        batch_size, in_channels, input_width, input_height = self.A.shape
        output_height = (input_height - self.kernel) + 1
        output_width = (input_width - self.kernel) + 1
        full_arrays = np.zeros([batch_size, in_channels, output_width, output_height])

        for batch in range(batch_size):
            for out_channels in range(in_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        full_arrays[batch, out_channels, j, i] = \
                            np.mean(A_copy[batch, out_channels, j:j + self.kernel, i:i + self.kernel].flatten())

        Z = full_arrays  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # TODO
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        batch_size, in_channels, input_width, input_height = self.A.shape
        dLdA = np.zeros((batch_size, out_channels, input_width, input_height))

        for batch in range(batch_size):
            for in_channel in range(out_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        dLdA[batch, in_channel, i:i+self.kernel, j:j+self.kernel]  += \
                            dLdZ[batch, in_channel, i, j] / (self.kernel * self.kernel)

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # TODO

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # TODO
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdZ = self.maxpool2d_stride1.backward(dLdZ)

        return dLdZ

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # TODO
        Z = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z)

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # TODO
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdZ = self.meanpool2d_stride1.backward(dLdZ)

        return dLdZ

