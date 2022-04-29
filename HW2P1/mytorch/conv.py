# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO

        self.A = A
        A_copy = A.copy()
        batch_size, in_channels, input_size = self.A.shape
        output_size = (input_size - self.kernel_size) + 1
        full_arrays = np.zeros([batch_size, self.out_channels, output_size])

        for batch in range(batch_size):
            for out_channels in range(self.out_channels):
                for i in range(output_size):
                    full_arrays[batch, out_channels, i] = \
                        (A_copy[batch, :, i:i + self.kernel_size] * self.W[out_channels, :, :]).sum()
                full_arrays[batch, out_channels] += self.b[out_channels]

        Z = full_arrays

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # TODO
        batch_size, out_channels, output_size = dLdZ.shape
        for batch in range(batch_size):
            for in_channels in range(self.in_channels):
                for out_channels in range(self.out_channels):
                    for i in range(self.kernel_size):
                        for output in range(output_size):
                            self.dLdW[out_channels, in_channels, i] += self.A[batch, in_channels, i + output] * dLdZ[
                                batch, out_channels, output]

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        dLdA = np.zeros(self.A.shape)

        for batch in range(batch_size):
            for in_channels in range(self.in_channels):
                for out_channels in range(self.out_channels):
                    for i in range(output_size):
                        for j in range(self.kernel_size):
                            dLdA[batch, in_channels, i + j] += dLdZ[batch, out_channels, i] * self.W[
                                out_channels, in_channels, j]

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn,
                                             bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO
        # Call Conv1d_stride1
        A_copy = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(A_copy)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdW = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdW)  # TODO

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

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
        output_height = (input_height - self.kernel_size) + 1
        output_width = (input_width - self.kernel_size) + 1
        full_arrays = np.zeros([batch_size, self.out_channels, output_width, output_height])

        for batch in range(batch_size):
            for out_channels in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        full_arrays[batch, out_channels, j, i] = \
                            (A_copy[batch, :, j:j + self.kernel_size, i:i + self.kernel_size]
                             * self.W[out_channels, :, :, :]).sum()
                full_arrays[batch, out_channels] += self.b[out_channels]

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

        batch_size, out_channels, output_width, output_height = self.dLdW.shape
        batch_size, out_channels, output_width_z, output_height_z = dLdZ.shape

        padNum = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (padNum, padNum), (padNum, padNum)), 'constant',
                             constant_values=(0, 0))

        for batch in range(batch_size):
            for inC in range(self.in_channels):
                for outC in range(self.out_channels):
                    for j in range(output_width):
                        for k in range(output_height):
                            self.dLdW[outC, inC, j, k] += (self.A[batch, inC, j:j + output_width_z, k:k + output_width_z] * dLdZ[batch, outC, :, :]).sum()

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        W_flip = np.rot90(self.W, k=2, axes=(2, 3))

        batch_size, out_channels, input_width, input_height = self.A.shape
        dLdA = np.zeros(self.A.shape)

        for batch in range(batch_size):
            for inC in range(self.in_channels):
                for i in range(input_width):
                    for j in range(input_height):
                        dLdA[batch, inC, i, j] += \
                            (dLdZ_padded[batch, :, i:i + self.kernel_size, j:j + self.kernel_size]
                             * W_flip[:, inC, :, :]).sum()

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance

        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn,
                                             bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv1d_stride1
        A_copy = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(A_copy)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO
        dLdW = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdW)  # TODO

        return dLdA


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)  # TODO
        self.upsample1d = Upsample1d(self.upsampling_factor)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO
        # upsample
        self.A = A
        A_upsampled = self.upsample1d.forward(self.A)  # TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # TODO

        # Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)  # TODO

        dLdA = self.upsample1d.backward(delta_out)  # TODO

        return dLdA


class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)  # TODO
        self.upsample2d = Upsample2d(self.upsampling_factor)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)  # TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)  # TODO

        dLdA = self.upsample2d.backward(delta_out)  # TODO

        return dLdA


class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.batch_size, self.in_channels, self.in_width = A.shape

        Z = A.reshape(self.batch_size, self.in_channels * self.in_width)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, (self.batch_size, self.in_channels, self.in_width))  # TODO

        return dLdA
