"""
Base classes for Sparse Autoencoder (SAE) variants.

This module provides:
- BaseSAE: A foundational SAE implementation using nn.Linear for encoder/decoder
  and ReLU activation. It adheres to the `dictionary_learning.dictionary.Dictionary` 
  interface.
- BaseTrainer: A trainer based on `dictionary_learning.trainers.standard.StandardTrainer` 
  that implements a basic reconstruction loss (MSE) plus an L1 sparsity penalty.

These classes are intended to be inherited by custom SAE variant implementations
in other files within the `custom_saes` directory. Users should override methods
like `encode`, `decode`, `forward` (in BaseSAE subclasses) or `loss` (in BaseTrainer
subclasses) to implement specific variant behaviors, while maintaining compatibility
with the training and evaluation pipeline.
"""

import torch as t
import torch.nn as nn
from dictionary_learning.dictionary import Dictionary
from dictionary_learning.trainers.standard import StandardTrainer

class BaseSAE(Dictionary, nn.Module):
    """
    Base Sparse Autoencoder (SAE) class.

    Implements a standard SAE architecture with a linear encoder, ReLU activation,
    and linear decoder. Includes bias terms for both encoder and decoder.
    Weights are initialized similarly to Anthropic's original implementation.
    Inherits from `dictionary_learning.dictionary.Dictionary` for compatibility.
    """
    def __init__(self, activation_dim, dict_size):
        """
        Initializes the BaseSAE.

        Args:
            activation_dim (int): The dimensionality of the input activations.
            dict_size (int): The number of features in the SAE dictionary (dimensionality
                           of the latent space).
        """
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # Initialize weights using the same method as AutoEncoderNew
        w = t.randn(activation_dim, dict_size)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x):
        """
        Encodes input activations to sparse features.

        Args:
            x (t.Tensor): Input activations tensor (batch_size, activation_dim).

        Returns:
            t.Tensor: Sparse feature activations (batch_size, dict_size).
        """
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        """
        Decodes sparse features back to activations.

        Args:
            f (t.Tensor): Sparse feature activations (batch_size, dict_size).

        Returns:
            t.Tensor: Reconstructed activations (batch_size, activation_dim).
        """
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False):
        """
        Performs a full forward pass: encode -> decode.

        Args:
            x (t.Tensor): Input activations.
            output_features (bool, optional): If True, returns both the reconstructed
                                          activations and the intermediate sparse
                                          features. Defaults to False.

        Returns:
            t.Tensor | Tuple[t.Tensor, t.Tensor]: 
                If `output_features` is False, returns the reconstructed activations.
                If `output_features` is True, returns a tuple (reconstructed_activations, features).
        """
        if not output_features:
            return self.decode(self.encode(x))
        else:
            f = self.encode(x)
            x_hat = self.decode(f)
            return x_hat, f

    @classmethod
    def from_pretrained(cls, path, device=None) -> "Dictionary":
        """Required by Dictionary ABC but not used in this implementation"""
        pass

class BaseTrainer(StandardTrainer):
    """
    Base trainer for SAEs using standard reconstruction + L1 sparsity loss.

    Inherits from `dictionary_learning.trainers.standard.StandardTrainer` and
    overrides the `loss` method to implement the typical SAE objective:
    MSE reconstruction loss + L1 penalty on feature activations.
    Includes functionality for sparsity penalty warm-up and tracking dead neurons.
    """
    # __init__ is inherited from StandardTrainer, taking config parameters.
    
    def update_dead_neurons(self, f):
        """
        Updates the count of steps since each neuron was last active.

        Assumes `self.steps_since_active` tensor exists and is initialized.
        Increments count for neurons that are zero across the entire batch (`f == 0`).
        Resets count for neurons that are active in the batch.

        Args:
            f (t.Tensor): Feature activations for the current batch (batch_size, dict_size).
        """
        deads = (f == 0).all(dim=0)
        self.steps_since_active[deads] += 1
        self.steps_since_active[~deads] = 0

    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Calculates the loss for a batch of activations.

        Loss = Reconstruction_MSE + L1_penalty * sparsity_scale * L1_norm(features)

        Also updates the dead neuron tracker.

        Args:
            x (t.Tensor): Input activations for the batch.
            step (int): Current training step number.
            logging (bool, optional): Flag for potential logging (not used here).
            **kwargs: Additional keyword arguments (not used here).

        Returns:
            t.Tensor: The calculated scalar loss value for the batch.
        """
        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        
        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        # If you redefine the loss function, make sure you still call this function
        # unless you explicitly don't want to track dead neurons
        if self.steps_since_active is not None:
            self.update_dead_neurons(f)

        return loss


# Example custom SAE and trainer, with nothing changed

class CustomSAE(BaseSAE):
    """Example of a custom SAE inheriting from BaseSAE without modifications."""
    def __init__(self, activation_dim, dict_size):
        """Initializes the example CustomSAE using the BaseSAE initializer."""
        super().__init__(activation_dim, dict_size)

class CustomTrainer(BaseTrainer):
    """Example of a custom Trainer inheriting from BaseTrainer without modifications."""
    def __init__(self, *args, **kwargs):
        """Initializes the example CustomTrainer using the BaseTrainer initializer."""
        super().__init__(*args, **kwargs)
