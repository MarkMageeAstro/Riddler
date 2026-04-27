import torch
import torch.nn as nn



class NeuralNetwork(nn.Module):

    """ 
    Class to define the neural network used for predictions.
    The neural network is constructed using a series of dense layers.
    The number of layers, the number of neurons in each layer, and the activation function are defined by the user.
    The output of the neural network is a Gaussian distribution defined by mean and sigma for each wavelength bin.
    There are separate output layers for the mean and sigma, and the sigma is constrained to be positive using a Softplus activation function.
    """

    def __init__(self, input_shape, output_shape, neurons, layers, activation):

        """ 
        Initializes the neural network.

        Parameters
        ----------
        input_shape : int
            Number of input dimensions.
        output_shape : int
            Number of output dimensions.
        NEURONS : int
            Number of neurons in each layer.
        LAYERS : int
            Number of layers.
        ACTIVATION : str
            Activation function to use.
        """

        super(NeuralNetwork, self).__init__()

        ## Define a flag for the warm-up phase, where only the mean is learned and sigma is kept fixed
        self.warm_up = False

        ## Create a sequential container for the layers
        self.layer_stack = nn.Sequential()
        self.output_mean_layer_stack = nn.Sequential()
        self.output_sigma_layer_stack = nn.Sequential()
        
        ## Define an input layer that takes the input shape
        self.layer_stack.add_module("input_layer", nn.Linear(input_shape, neurons))

        ## Add the specified number of layers
        for i in range(layers):
            self.layer_stack.add_module(f"layer_{i}", nn.Linear(neurons, neurons))
            self.layer_stack.add_module(f"activation_{i}", getattr(nn, activation)())

        ## Add output layers that return the output shape
        ## Add one for the mean and one for sigma of each wavelength bin
        self.output_mean_layer_stack.add_module("output_mean_layer", nn.Linear(neurons, output_shape))
        self.output_sigma_layer_stack.add_module("output_sigma_layer", nn.Linear(neurons, output_shape))
        ## Use a Softplus activation function to ensure that the predicted sigma is positive
        self.output_sigma_layer_stack.add_module("activation_output_sigma_layer", nn.Softplus())

        ## Count the number of parameters in the neural network
        self.n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return None

    def forward(self, x):
        """
        Forward pass through the neural network.
        """

        ## Pass the input through the layers
        x = self.layer_stack(x)
        
        ## Get the mean of the flux for each wavelength bin
        output_mean = self.output_mean_layer_stack(x)
        
        ## If in the warm-up phase, return a fixed sigma of 1 for each wavelength bin, otherwise return the predicted sigma from the output layer
        if self.warm_up:
            output_sigma = torch.ones_like(output_mean)
        else:
            output_sigma = self.output_sigma_layer_stack(x) + 1.e-4

        ## Return the mean and sigma
        return output_mean, output_sigma