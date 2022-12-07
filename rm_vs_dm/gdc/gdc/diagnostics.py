# from functools import partial
import numpy as np
import scipy.stats
import torch
import math
from typing import List
from torch import Tensor

class GradientMonitor:

    def __init__(self, model, optimizer):

        self.model = model
        self.optimizer = optimizer

    def compute(self, loss_batch):
        """ a method to calculate mean of variance of gradients estimates within a batch

        Args:
            loss_batch (Tensor): shape [bsz] Loss of batch before taking average
            optimizer (torch.optim): to call optimizer.zero_grad()
            model :  Model being trained on the loss to get the gradients
        Returns:
            dict: a dictionary containing values of mean and variance of gradients vector in this batch
        """
        output = {}
        # calculate mean of gradients
        loss = torch.mean(loss_batch)
        # square of l2 norm of the mean gradient vector
        grads = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], grad_outputs=torch.ones_like(loss), allow_unused=True, retain_graph=True)
        mu_grad = torch.cat([i.detach().flatten() for i in grads if i is not None]).cpu()

        # calculate variance
        # iterate over each sample loss in the batch
        grads_vector_l22 = []
        for l in loss_batch:

            grads = torch.autograd.grad(l, [p for p in self.model.parameters() if p.requires_grad], grad_outputs=torch.ones_like(l), allow_unused=True, retain_graph=True)
            grad_vector_l22 =  sum([(i.detach()**2).sum() for i in grads if i is not None]).cpu()
            grads_vector_l22.append(grad_vector_l22)

        output['grad_l22'] = torch.mean(torch.stack(grads_vector_l22))
        output['mu_grad'] = mu_grad

        return output