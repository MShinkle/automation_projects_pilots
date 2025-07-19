"""
Script to train a Sparse Autoencoder (SAE) variant.

This script handles:
- Loading configuration for the LLM, buffer, SAE, and training.
- Setting up the language model (LLM) and activation buffer.
- Dynamically importing a custom SAE variant based on a command-line argument.
- Configuring and running the training process using `trainSAE_simple`.
- Saving the trained SAE model to a pickle file.
"""

import torch as t
import pickle
from nnsight import LanguageModel
import os
import random
import argparse
import importlib

# Import our simplified training function
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from utils import trainSAE_simple

def main(sae_name):
    """
    Main function to train a specified SAE variant.

    Args:
        sae_name (str): The name of the SAE variant module located in the 
                        'custom_saes' directory (e.g., 'base', 'top_k').
                        This name is used to dynamically import the corresponding
                        `CustomSAE` and `CustomTrainer` classes.
    """
    # Dynamically import the SAE module
    sae_module = importlib.import_module(f"custom_saes.{sae_name}")
    
    # Import SAE components from the dynamic module
    SAE = sae_module.CustomSAE
    Trainer = sae_module.CustomTrainer
    
    # Check CUDA availability and set device
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configuration organized into logical groups
    # 1. LLM configuration
    llm_config = {
        "model_name": "EleutherAI/pythia-70m-deduped",
        "layer": 3,
        "device": device,
    }

    # 2. Buffer configuration
    buffer_config = {
        "context_length": 128,
        "llm_batch_size": 512,  # Fits on a 24GB GPU
        "sae_batch_size": 8192,
        "num_tokens": 10_000_000,
    }

    # 3. SAE configuration
    sae_config = {
        "expansion_factor": 8,
        "save_dir": "./trained_saes",
    }

    # 4. Training configuration
    training_config = {
        "learning_rate": 3e-4,
        "sparsity_penalty": 2.0,
        "warmup_steps": 1000,
        "random_seed": 42,
    }

    # Set random seeds
    random.seed(training_config["random_seed"])
    t.manual_seed(training_config["random_seed"])
    
    # Load the model
    model = LanguageModel(llm_config["model_name"], dispatch=True, device_map=device)
    
    # Calculate derived parameters
    num_contexts_per_batch = buffer_config["sae_batch_size"] // buffer_config["context_length"]
    n_ctxs = num_contexts_per_batch * 20  # Store 20x contexts for efficient sampling
    steps = int(buffer_config["num_tokens"] / buffer_config["sae_batch_size"])
    activation_dim = model.config.hidden_size
    
    # Setup model component to analyze
    submodule = model.gpt_neox.layers[llm_config["layer"]]
    submodule_name = f"resid_post_layer_{llm_config['layer']}"
    
    # Create dataset generator
    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")
    
    # Create activation buffer
    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=n_ctxs,
        ctx_len=buffer_config["context_length"],
        refresh_batch_size=buffer_config["llm_batch_size"],
        out_batch_size=buffer_config["sae_batch_size"],
        io="out",
        d_submodule=activation_dim,
        device=device,
    )
    
    # Create trainer config
    trainer_config = {
        "trainer": Trainer,
        "dict_class": SAE,
        "activation_dim": activation_dim,
        "dict_size": sae_config["expansion_factor"] * activation_dim,
        "lr": training_config["learning_rate"],
        "l1_penalty": training_config["sparsity_penalty"],
        "warmup_steps": training_config["warmup_steps"],
        "sparsity_warmup_steps": None,  # Matching test_end_to_end.py
        "decay_start": None,
        "steps": steps,
        "resample_steps": None,
        "seed": training_config["random_seed"],
        "layer": llm_config["layer"],
        "lm_name": llm_config["model_name"],
        "device": device,
        "submodule_name": submodule_name,
    }
    
    # Train and get both the trainer and the model
    trainer, sae_model = trainSAE_simple(
        data=activation_buffer,
        trainer_config=trainer_config,
        steps=steps,
        device=device,
    )
    
    # Save the model
    os.makedirs(sae_config["save_dir"], exist_ok=True)
    model_path = os.path.join(sae_config["save_dir"], 
                             f"{llm_config['model_name'].split('/')[-1]}_layer_{llm_config['layer']}_{sae_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(sae_model, f)
    
    print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SAE model')
    parser.add_argument('sae_name', type=str, help='Name of the SAE model to train')
    args = parser.parse_args()
    
    main(args.sae_name)
