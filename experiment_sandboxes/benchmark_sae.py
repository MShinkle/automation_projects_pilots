"""
Script to benchmark trained Sparse Autoencoder (SAE) models using sae-bench.

This script handles:
- Finding trained SAE models (.pkl files) in the specified directory.
- Loading SAE models and adapting them for sae-bench using `SAEAdapter`.
- Configuring and running selected sae-bench evaluations (e.g., core metrics) 
  using the `run_evals` function.
- Optionally benchmarking a specific model or skipping existing results.
"""

import pickle
import torch
import sae_bench.sae_bench_utils.general_utils as general_utils
from utils import SAEAdapter, run_evals
import os
import glob
import argparse

def main(force_rerun=True, specific_model=None):
    """
    Main function to load and benchmark SAE models.

    Finds SAE pickle files, loads them, adapts them using SAEAdapter,
    and runs sae-bench evaluations.

    Args:
        force_rerun (bool, optional): If True, rerun evaluations even if results 
                                    exist. Defaults to True.
        specific_model (str | None, optional): If provided, only benchmark the SAE 
                                          variant with this name (e.g., "base", "top_k"). 
                                          If None, benchmark all found SAEs. 
                                          Defaults to None.
    """
    # Model configuration
    model_name = "pythia-70m-deduped"
    batch_size = 512
    hook_layer = 3
    
    # Output folder for evaluation results
    output_folder = "./benchmark_results"

    dtype = "float32"
    torch_dtype = general_utils.str_to_dtype(dtype)
    device = general_utils.setup_environment()
    print(f"Using device: {device}")

    # Select your eval types here.
    eval_types = [
        # "absorption",
        "core",
        # "ravel",  # Commented out as it only supports Gemma 2B, not Pythia
        # "scr",
        # "tpp",
        # "sparse_probing",
    ]

    save_dir = "./trained_saes"
    
    # Get all SAE files from the trained_saes directory
    sae_files = glob.glob(os.path.join(save_dir, "*.pkl"))
    if not sae_files:
        print(f"No trained SAEs found in {save_dir}")
        return
        
    # Filter for specific model if requested
    if specific_model:
        expected_filename = f"{model_name}_layer_{hook_layer}_{specific_model}.pkl"
        sae_files = [f for f in sae_files if os.path.basename(f) == expected_filename]
        if not sae_files:
            print(f"No trained SAE found with name '{expected_filename}' in {save_dir}")
            return
        print(f"Found specific model '{specific_model}' to benchmark")
    else:
        print(f"Found {len(sae_files)} trained SAEs to benchmark")
    
    # Create list to store all SAEs and their names
    selected_saes = []
    
    # Load each SAE
    for sae_file in sae_files:
        # Extract model name, layer, and SAE name from filename
        filename = os.path.basename(sae_file)
        # Format: model_layer_X_sae_name.pkl
        parts = filename.split('_')
        model_name = parts[0]
        layer = int(parts[2])
        sae_name = '_'.join(parts[3:]).replace('.pkl', '')
        
        print(f"\nLoading {sae_name} from {model_name} layer {layer}...")

        with open(sae_file, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
            
        # Move model to the correct device
        model = model.to(device)
        print(f"Loaded model with dimensions: {model.activation_dim} × {model.dict_size}")

        # Adapt the model for SAEBench using our adapter
        sae = SAEAdapter.from_dict_learning_model(
            model,
            model_name=model_name,
            hook_layer=layer,
            device=torch.device(device),
            dtype=torch_dtype,
            architecture_name=sae_name
        )

        # Ensure the model is in the correct dtype
        sae = sae.to(dtype=torch_dtype)
        sae.cfg.dtype = dtype
        
        # Add to selected_saes list
        selected_saes.append((f"{model_name}_layer_{layer}_{sae_name}", sae))
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache()

    print(f"\nRunning evaluations on {len(selected_saes)} SAEs...")
    
    run_evals(
        model_name=model_name,
        selected_saes=selected_saes,
        llm_batch_size=batch_size,
        llm_dtype=dtype,
        device=device,
        eval_types=eval_types,
        output_folder=output_folder,
        force_rerun=force_rerun,
        save_activations=True,  # Enable saving activations for efficiency
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark all trained SAEs')
    parser.add_argument('--skip-existing', action='store_true', 
                      help='Skip evaluations that already have results (default: False)')
    parser.add_argument('model', nargs='?', type=str,
                      help='Optional: Specific model name to benchmark (e.g. "top_k"). If not provided, benchmarks all models.')
    args = parser.parse_args()
    
    main(force_rerun=not args.skip_existing, specific_model=args.model)
