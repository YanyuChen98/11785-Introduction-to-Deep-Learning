import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # TODO
        batch_size, in_channels, input_width = A.shape
        width_upsampled = input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        A_up = np.zeros((batch_size, in_channels, width_upsampled))

        for batch in range(batch_size):
            for in_channel in range(in_channels):
                for width in range(input_width):
                    A_up[batch, in_channel, width * self.upsampling_factor] = A[batch, in_channel, width]

        Z = np.array(A_up)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # TODO

        dLdZ_copy = dLdZ.copy()
        batch_size, in_channels, output_width = dLdZ.shape

        if output_width % self.upsampling_factor != 0:
            dLdZ_temp = np.zeros((batch_size, in_channels, output_width // self.upsampling_factor + 1))
        else:
            dLdZ_temp = np.zeros((batch_size, in_channels, output_width // self.upsampling_factor))

        for batch in range(batch_size):
            for in_channel in range(in_channels):
                for output in range(0, output_width, self.upsampling_factor):
                    dLdZ_temp[batch, in_channel, output//self.upsampling_factor] = dLdZ_copy[batch, in_channel, output]

        dLdA = np.array(dLdZ_temp)

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        #TODO
        self.A = A
        A_copy = A.copy()
        batch_size, in_channels, input_width = A.shape

        if input_width % self.downsampling_factor != 0:
            Z_temp = np.zeros((batch_size, in_channels, input_width // self.downsampling_factor + 1))
        else:
            Z_temp = np.zeros((batch_size, in_channels, input_width // self.downsampling_factor))


        for batch in range(batch_size):
            for in_channel in range(in_channels):
                for output in range(0, input_width, self.downsampling_factor):
                    Z_temp[batch, in_channel, output//self.downsampling_factor] = A_copy[batch, in_channel, output]

        Z = np.array(Z_temp)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # TODO
        batch_size, in_channels, output_width = dLdZ.shape
        batch_size_A, in_channels_A, input_width = self.A.shape

        dLdA_temp = np.zeros((batch_size, in_channels, input_width))
        dLdZ_copy = dLdZ.copy()

        for batch in range(batch_size):
            for in_channel in range(in_channels):
                for output in range(input_width):
                    if output * self.downsampling_factor < input_width:
                        dLdA_temp[batch, in_channel, output * self.downsampling_factor] = dLdZ_copy[
                            batch, in_channel, output]

        dLdA = np.array(dLdA_temp)
        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        # TODO
        batch_size, in_channels, input_height, input_width = A.shape
        A_copy = A.copy().tolist()

        for batch in A_copy:
            j = 0
            for input in batch:
                i = 0
                for row in input:
                    temp = [0] * (input_width * self.upsampling_factor - self.upsampling_factor + 1)
                    temp[::self.upsampling_factor] = row
                    input[i] = temp
                    i = i + 1
                input_temp = [[0] * (input_width * self.upsampling_factor - self.upsampling_factor + 1)] * (input_height * self.upsampling_factor - self.upsampling_factor + 1)
                input_temp[0::self.upsampling_factor] = input
                batch[j] = input_temp
                j = j + 1

        Z = np.array(A_copy)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # TODO
        batch_size, in_channels, input_height, input_width = dLdZ.shape

        dLdZ_copy = dLdZ.copy().tolist()

        for batch in dLdZ_copy:
            j = 0
            for input in batch:
                i = 0
                for row in input:
                    input[i] = [row[x] for x in range(0, input_width, self.upsampling_factor)]
                    i = i + 1
                batch[j] = [input[e] for e in range(0, input_height, self.upsampling_factor)]
                j = j + 1

        dLdA = np.array(dLdZ_copy)

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        # TODO
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape

        if input_width % self.downsampling_factor != 0:
            if input_height % self.downsampling_factor != 0:
                Z_temp = np.zeros((batch_size, in_channels, input_width // self.downsampling_factor + 1,
                                   input_height // self.downsampling_factor + 1))
            else:
                Z_temp = np.zeros((batch_size, in_channels, input_width // self.downsampling_factor + 1,
                                   input_height // self.downsampling_factor))
        elif input_height % self.downsampling_factor != 0:
            Z_temp = np.zeros((batch_size, in_channels, input_width // self.downsampling_factor,
                               input_height // self.downsampling_factor + 1))
        else:
            Z_temp = np.zeros((batch_size, in_channels, input_width // self.downsampling_factor,
                               input_height // self.downsampling_factor))

        for batch in range(batch_size):
            for in_channel in range(in_channels):
                for i in range(0, input_width, self.downsampling_factor):
                    for j in range(0, input_height, self.downsampling_factor):
                        Z_temp[batch, in_channel, i//self.downsampling_factor, j//self.downsampling_factor] = \
                            A[batch, in_channel, i, j]

        Z = np.array(Z_temp)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # TODO
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        batch_size_A, in_channels_A, input_width, input_height = self.A.shape

        dLdA_temp = np.zeros((batch_size, in_channels, input_width, input_height))
        dLdZ_copy = dLdZ.copy()

        for batch in range(batch_size):
            for in_channel in range(in_channels):
                for i in range(input_width):
                    for j in range(input_height):
                        if i * self.downsampling_factor < input_width:
                            if j * self.downsampling_factor < input_height:
                                dLdA_temp[batch, in_channel, i * self.downsampling_factor, j * self.downsampling_factor] = \
                                    dLdZ_copy[batch, in_channel, i, j]

        dLdA = np.array(dLdA_temp)
        return dLdA
