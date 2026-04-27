import torch

def Loss(mean, sigma, truth):

    """  
    Define a loss function to be used by the neural network during training.
    The negative log likelihood is computed using the mean and sigma of the output Gaussian distribution and compared to the truth (i.e. trainig data).
    """

    ## Get the variance
    sigma2 = torch.square(sigma)

    ## Generalised form of MSE
    loss = torch.nn.GaussianNLLLoss(reduction='mean')
    nll = loss(mean, truth, sigma2)

    ## Return the total loss
    return nll
