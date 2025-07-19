"""
Utility functions and classes for SAE training and evaluation.

This module provides:
- SAEAdapter: A class to wrap dictionary_learning SAEs for compatibility with sae-bench.
- trainSAE_simple: A simplified function to train an SAE.
- run_evals: A function to execute sae-bench evaluations on trained SAEs.
"""

import os
from typing import Optional, Dict, Any, Tuple
from contextlib import nullcontext
from tqdm import tqdm
import torch as t
import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import time

RANDOM_SEED = 42

class SAEAdapter(base_sae.BaseSAE):
    """
    Adapter class that converts a dictionary_learning model to a SAEBench-compatible format.

    This class wraps an existing SAE model (trained using the dictionary_learning
    library) and exposes the necessary attributes and methods (encode, decode, forward,
    W_enc, W_dec, b_enc, b_dec) expected by the sae-bench evaluation framework.
    It handles parameter mapping and optional decoder weight normalization.
    """
    def __init__(self, d_in: int, d_sae: int, model_name: str, hook_layer: int, device: str, dtype: str):
        """
        Initializes the SAEAdapter base class.

        Args:
            d_in: Input dimension (activation size).
            d_sae: SAE dictionary size.
            model_name: Name of the base transformer model.
            hook_layer: Layer index from which activations were extracted.
            device: Computation device ('cuda' or 'cpu').
            dtype: Data type (e.g., 'float32').
        """
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype)
        self._original_model = None  # Will be set in from_dict_learning_model
    
    @classmethod
    def from_dict_learning_model(cls, 
                                original_sae,  # The original SAE model to convert
                                model_name, 
                                hook_layer, 
                                device, 
                                dtype,
                                architecture_name="sae_adapter",
                                normalize_decoder: bool = True):
        """
        Creates a SAEBench-compatible adapter from a dictionary_learning SAE model.
        
        Args:
            original_sae: The trained SAE model instance from dictionary_learning.
            model_name: Name of the base transformer model (e.g., "pythia-70m-deduped").
            hook_layer: Layer index from which activations were extracted.
            device: Computation device ('cuda' or 'cpu').
            dtype: Data type (e.g., 'float32').
            architecture_name: Name for this architecture to be used in evaluation results.
                               Defaults to "sae_adapter".
            normalize_decoder: Whether to normalize decoder weights to unit L2 norm. 
                               Defaults to True.

        Returns:
            SAEAdapter: An instance of SAEAdapter wrapping the original model.
        """
        # Create a new instance with appropriate dimensions
        d_in = original_sae.activation_dim
        d_sae = original_sae.dict_size
        adapter = cls(d_in, d_sae, model_name, hook_layer, device, dtype)
        
        # Store the original model for method delegation
        adapter._original_model = original_sae
        
        # Map parameters from dictionary_learning to SAEBench format
        with t.no_grad():
            adapter.W_enc.data = original_sae.encoder.weight.T.clone()
            adapter.W_dec.data = original_sae.decoder.weight.T.clone()
            
            # Normalize decoder weights using L2 norm along dimension 1 (matching SAEBench)
            if normalize_decoder:
                norms = t.norm(adapter.W_dec.data, dim=1, keepdim=True)
                adapter.W_dec.data /= norms
            
            if original_sae.encoder.bias is not None:
                adapter.b_enc.data = original_sae.encoder.bias.clone()
            else:
                adapter.b_enc.data = t.zeros(d_sae, device=device, dtype=dtype)
            
            if original_sae.decoder.bias is not None:
                adapter.b_dec.data = original_sae.decoder.bias.clone()
            else:
                adapter.b_dec.data = t.zeros(d_in, device=device, dtype=dtype)
        
        adapter.cfg.architecture = architecture_name
        
        return adapter
    
    def encode(self, x: t.Tensor) -> t.Tensor:
        """
        Encodes the input tensor using the original model's encode method.

        Args:
            x: Input tensor (typically activations).

        Returns:
            Tensor: Encoded features.
        """
        return self._original_model.encode(x)
    
    def decode(self, feature_acts: t.Tensor) -> t.Tensor:
        """
        Decodes the feature activations using the original model's decode method.

        Args:
            feature_acts: Tensor of feature activations.

        Returns:
            Tensor: Reconstructed input.
        """
        return self._original_model.decode(feature_acts)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Performs a forward pass using the original model's forward method.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output of the forward pass (typically reconstructed input).
        """
        return self._original_model.forward(x)

def trainSAE_simple(
    data,
    trainer_config: dict,
    steps: int,
    device: str, # Added device parameter
) -> Tuple[Any, t.nn.Module]:
    """
    Trains a single SAE using the provided configuration and data iterator.

    This function initializes a trainer based on the configuration, then iterates
    through the data, performing training updates for the specified number of steps.
    
    Args:
        data: An iterator yielding batches of activation data.
        trainer_config: A dictionary containing the configuration for the trainer, 
                        including the 'trainer' class, 'dict_class', and hyperparameters.
        steps: The total number of training steps to perform.
        device: The computation device ('cuda' or 'cpu').
    
    Returns:
        Tuple[Any, t.nn.Module]: A tuple containing the final trainer instance 
                                 and the trained SAE model (trainer.ae).
    """
    config = trainer_config.copy()
    if "wandb_name" in config:
        del config["wandb_name"]
    trainer_class = config.pop("trainer")
    
    # Create trainer with all remaining config parameters
    trainer = trainer_class(**config)
    
    for step, act in enumerate(tqdm(data, total=steps)):
        if step >= steps:
            break
        trainer.update(step, act)
    
    return trainer, trainer.ae

def run_evals(
    model_name: str,
    selected_saes: list[tuple[str, Any]],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    output_folder: str = "sae_bench_results",
    force_rerun: bool = True,
    save_activations: bool = False,
):
    """
    Runs selected sae-bench evaluations for a given model and list of SAEs.

    Initializes and runs evaluation suites (e.g., 'core', 'absorption') specified
    in `eval_types`. Results are saved to the `output_folder`.

    Args:
        model_name: Name of the base transformer model (e.g., "pythia-70m-deduped").
        selected_saes: A list of tuples, where each tuple contains a unique name 
                       for the SAE and the SAE model instance (wrapped by SAEAdapter).
        llm_batch_size: Batch size for processing data through the language model.
        llm_dtype: Data type for language model operations (e.g., 'float32').
        device: Computation device ('cuda' or 'cpu').
        eval_types: A list of strings specifying which evaluation suites to run 
                    (e.g., ['core', 'sparse_probing']).
        output_folder: Path to the directory where evaluation results will be saved.
                       Defaults to "sae_bench_results".
        force_rerun: If True, forces evaluations to run even if results already exist.
                     Defaults to True.
        save_activations: If True, saves activations during evaluations like SCR/TPP 
                          and sparse probing for potential reuse. Defaults to False.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Define evaluation runners
    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                os.path.join(output_folder, "absorption"),
                force_rerun,
            )
        ),
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=os.path.join(output_folder, "core"),
                verbose=True,
                dtype=llm_dtype,
                device=device,
                force_rerun=force_rerun,
            )
        ),
        "ravel": (
            lambda: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                os.path.join(output_folder, "ravel"),
                force_rerun,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                output_folder,
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                output_folder,
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                os.path.join(output_folder, "sparse_probing"),
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
    }

    # Store evaluation times
    eval_times = {}

    # Run evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")
        if eval_type in eval_runners:
            if eval_type not in ["scr", "tpp"]:
                os.makedirs(os.path.join(output_folder, eval_type), exist_ok=True)
            start_time = time.time()
            eval_runners[eval_type]()
            os.makedirs(os.path.join(output_folder, eval_type), exist_ok=True)
            start_time = time.time()
            eval_runners[eval_type]()
            end_time = time.time()
            eval_times[eval_type] = end_time - start_time

    # Print timing summary
    print("\nEvaluation Timing Summary:")
    print("=" * 40)
    for eval_type, duration in eval_times.items():
        print(f"{eval_type:.<30} {duration:.2f} seconds")
    print("=" * 40)
