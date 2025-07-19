## Table of Contents
// ... existing code ...
4.  [Training SAEs](#training-saes)
5.  [Analyzing SAE Features](#analyzing-sae-features)
    -   [Get Feature Activations for a Prompt](#get-feature-activations-for-a-prompt)
    -   [Analyze Feature Logit Contributions](#analyze-feature-logit-contributions)
6.  [Evaluating SAE Performance](#evaluating-sae-performance)
// ... existing code ...
# The TrainingSAE forward pass is implicitly handled within the SAETrainingRunner.fit() loop
# and doesn't typically require a separate documentation section unless demonstrating
# manual forward pass usage outside the trainer.

# The SAETrainer initialization and fit method are combined into the 
# 'Complete Training Cycle' section below for a more practical workflow demonstration.

---

## Training SAEs

**Common Setup:**

Before running the training subtasks, you need to initialize the core components: the configuration, the base model, and the activation store. These components will be reused across the subsequent steps.

```python
# Common imports for the training section
import torch
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer, SaveCheckpointFn
from transformer_lens import HookedTransformer
import os
import shutil
from typing import List, Optional # Added Optional

# 1. Define the Configuration (Adjust parameters as needed)
cfg = LanguageModelSAERunnerConfig(
    # Model and Data parameters
    model_name="tiny-stories-1M",
    hook_name="blocks.0.hook_resid_pre", 
    hook_layer=0,
    dataset_path="roneneldan/TinyStories",
    is_dataset_tokenized=False,
    context_size=64,
    d_in=64, # Dimension of the activation hook point
    
    # SAE Parameters
    expansion_factor=4, # SAE dictionary size = expansion_factor * d_in
    l1_coefficient=1e-4, # Sparsity penalty strength
    
    # Training Parameters
    lr=1e-4,
    lr_scheduler_name="constant", # Example: Can be "cosine_warmup", etc.
    train_batch_size_tokens=512, # Adjusted for typical use
    training_tokens=10_000_000, # Example: Set target training tokens
    
    # Activation Store Parameters
    n_batches_in_buffer=64, # Larger buffer for efficiency
    store_batch_size_prompts=32, # Larger generation batch size
    
    # Logging and Saving Parameters
    log_to_wandb=False, # Set to True to enable WandB logging
    wandb_project="saelens_training_example", # WandB project name
    wandb_log_frequency=100, # Log metrics every 100 steps
    checkpoint_path="./sae_training_checkpoints", # Base directory for checkpoints
    n_checkpoints=5, # Number of checkpoints to save during training
    
    # Runtime Parameters
    device="cpu", # Use "cuda" for GPU
    seed=42,
    dtype="float32" # Can be "bfloat16" for mixed precision
)

# 2. Initialize the Base Model (No Processing needed for training)
print("Initializing HookedTransformer model (no processing)...")
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name, device=cfg.device)
assert model is not None
print("Model initialized.")

# 3. Initialize the Activations Store
print("Initializing ActivationsStore...")
# Note: The ActivationsStore might take time to initialize and fill its buffer
store = ActivationsStore.from_config(model, cfg)
assert store is not None
print("ActivationsStore initialized.")

# Set seed for reproducibility in subsequent steps
torch.manual_seed(cfg.seed)

```

### Complete Training Cycle (Initialization, Training, Saving)

**Description:**
This section covers the complete process of setting up and running the SAE training loop. It involves initializing the `TrainingSAE` model, configuring the `SAETrainer` with an optimizer, schedulers, and a checkpoint saving mechanism, and finally executing the training using the `fit()` method.

The forward pass logic is handled internally by the `SAETrainer._train_step` method during the `fit()` execution.

**Prerequisites:**
- An initialized `LanguageModelSAERunnerConfig` object (`cfg`).
- An initialized base model (`model`).
- An initialized `ActivationsStore` (`store`).

**Expected Outcome:**
- A `TrainingSAE` model is initialized.
- An `SAETrainer` is initialized with optimizer and schedulers.
- The `fit()` method runs until `cfg.training_tokens` are processed.
- Checkpoints are saved to `cfg.checkpoint_path` at specified intervals using the provided save function.
- The method returns the trained `TrainingSAE` instance.

**Tested Code Implementation:**
_Verified via `./tests/saelens/test_sae_full_training_cycle.py`._
```python
# Assumes `cfg`, `model`, `store` are defined from the Common Setup section

# 1. Define a Checkpoint Saving Function
# This function defines *how* a checkpoint is saved when triggered by the trainer.
def refined_save_checkpoint_fn(trainer: SAETrainer, checkpoint_name: str, wandb_aliases: Optional[List[str]] = None):
    """
    Saves the SAE model state. Includes commented-out examples for saving
    optimizer and scheduler states, which are necessary if you plan to
    fully resume training from this checkpoint later.
    """
    # The trainer provides the base path and the checkpoint name (e.g., "final_10000")
    path = f"{trainer.cfg.checkpoint_path}/{checkpoint_name}" 
    print(f"Saving checkpoint: {checkpoint_name} to {path}")
    os.makedirs(path, exist_ok=True)
    
    # Calculate sparsity - Placeholder: A real implementation would need access
    # to representative data and calculate actual feature sparsity/density.
    # This might involve running a forward pass on a sample dataset.
    dtype = getattr(torch, trainer.cfg.dtype) if isinstance(trainer.cfg.dtype, str) else trainer.cfg.dtype
    device = torch.device(trainer.cfg.device)
    print("Warning: Sparsity calculation in save function is using a placeholder (zeros).")
    sparsity = torch.zeros(trainer.sae.cfg.d_sae, device=device, dtype=dtype)
    
    # Save the SAE model weights, config, and sparsity
    trainer.sae.save_model(path, sparsity=sparsity)
    
    # --- Optional: Save optimizer and scheduler state for resuming ---
    # Uncomment the following lines if you need to resume training later.
    # torch.save(trainer.optimizer.state_dict(), f"{path}/optimizer.pt")
    # torch.save(trainer.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
    # if hasattr(trainer, 'l1_scheduler') and trainer.l1_scheduler:
    #     torch.save(trainer.l1_scheduler.state_dict(), f"{path}/l1_scheduler.pt")
    # print("Optimizer and scheduler states saved (required for resuming training).")
    # --- End Optional ---
    
    print(f"Checkpoint '{checkpoint_name}' saved successfully.")
    # Note: The SAETrainingRunner handles saving additional state like activation store.

# 2. Initialize TrainingSAE
print("Initializing TrainingSAE...")
training_sae_cfg = TrainingSAEConfig.from_dict(cfg.get_training_sae_cfg_dict())
sae = TrainingSAE(training_sae_cfg)
assert sae is not None
print(f"TrainingSAE initialized: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

# 3. Initialize SAETrainer
print("Initializing SAETrainer...")
trainer = SAETrainer( 
    model=model, 
    sae=sae,
    activation_store=store, 
    cfg=cfg, 
    save_checkpoint_fn=refined_save_checkpoint_fn # Provide the refined save function
)
assert trainer is not None
assert hasattr(trainer, 'optimizer')
assert hasattr(trainer, 'lr_scheduler')
print("SAETrainer initialized successfully.")

# 4. Run Training Loop (Fit)
print(f"Running SAETrainer.fit() to train for {cfg.total_training_tokens} tokens...")
# The fit method handles batch fetching, forward/backward passes, optimization, 
# logging (if enabled), and calling the save_checkpoint_fn.
final_sae = trainer.fit()

assert final_sae is trainer.sae # fit() returns the same SAE instance
assert trainer.n_training_tokens >= cfg.total_training_tokens
print("SAETrainer.fit() completed.")
print(f"Trained for {trainer.n_training_tokens} tokens.")

# Checkpoint(s) should have been saved by refined_save_checkpoint_fn during fit.
# Verify by checking the cfg.checkpoint_path directory.

```

**Implementation Tips:**
- The core training logic (forward pass, loss calculation, backward pass, optimizer step) is encapsulated within the `SAETrainer.fit()` and `SAETrainer._train_step()` methods. You don't need to manually implement the forward pass for training.
- The `refined_save_checkpoint_fn` is essential. You must provide a function that takes the `trainer`, a `checkpoint_name` string, and optional `wandb_aliases`, and performs the actual saving. The example saves the SAE model and provides commented-out lines for saving optimizer/scheduler state, which is necessary for fully resuming training.
- The `SAETrainer` uses the parameters in `cfg` (e.g., `lr`, `adam_beta1`, `lr_scheduler_name`, `l1_warm_up_steps`) to configure the Adam optimizer and learning rate/L1 coefficient schedulers automatically.
- Monitor console output for progress (tokens trained, loss values if logged) and any potential errors.
- If `cfg.log_to_wandb=True`, ensure you are logged into WandB (`wandb login`) and have specified `cfg.wandb_project` and optionally `cfg.wandb_entity`.
- Remember to clean up the `cfg.checkpoint_path` directory after test runs or when checkpoints are no longer needed.
- **Sparsity Calculation:** Note that the `refined_save_checkpoint_fn` example uses a placeholder (zeros) for sparsity. A production implementation would need to calculate actual feature sparsity, potentially by running a forward pass on a representative dataset within the save function or periodically during training.

**Variants:**
- **Optimizer/Scheduler:** Customize by changing parameters in `cfg` (e.g., `lr`, `lr_end`, `lr_scheduler_name`, `adam_beta1`, `adam_beta2`, `l1_warm_up_steps`).
- **Checkpointing:** Adjust `cfg.n_checkpoints` to control save frequency. Implement full state saving in `refined_save_checkpoint_fn` for resumability.
- **Hardware:** Run on GPU by setting `cfg.device="cuda"` (ensure all components use the same device).
- **Precision:** Use mixed precision by setting `cfg.dtype="bfloat16"` (if hardware supports it).

**Troubleshooting:**
- **CUDA Out of Memory:** Reduce `cfg.train_batch_size_tokens` or `cfg.store_batch_size_prompts`. Consider using mixed precision (`cfg.dtype="bfloat16"`) if hardware supports it.
- **Slow Training:** Increase batch sizes if memory allows. Ensure the dataset is pre-tokenized (see `cfg.is_dataset_tokenized` and related docs). Use a GPU.
- **NaN Losses:** Lower the learning rate (`cfg.lr`). Check activation normalization (`cfg.normalize_activations`). Investigate potential numerical issues in the base model or SAE architecture.
- **Checkpointing Issues:** Verify write permissions for `cfg.checkpoint_path`. Ensure the `save_checkpoint_fn` works correctly and handles potential exceptions.
- **Directory Cleanup:** Remember to manually clean up the `cfg.checkpoint_path` directory after test runs or if checkpoints are no longer needed, as the example code doesn't automatically remove them.
- **SAE File Loading/Downloading:** Ensure the required SAE file (`.pt` and `.json`) exists at the expected path before attempting to load it with `SAE.load_from_pretrained`. Automatic download utilities (like those previously found in `sae_lens.toolkit.pretrained_saes`) may change location or become unavailable in different package versions. It's often more reliable to manually download or confirm the existence of the SAE file first.

---

## Analyzing SAE Features

Once an SAE is trained or loaded (using methods from previous sections or `SAE.from_pretrained`), you can analyze its features to understand what they represent and how they affect the model's behavior. Common analysis techniques involve looking at which inputs activate specific features and how features influence the model's output logits.

### Get Feature Activations for a Prompt

**Description:**
Identify which SAE features activate most strongly for a given input prompt, particularly for specific token positions.

**Prerequisites:**
- An initialized base model (`model`, e.g., `HookedTransformer`).
- A loaded or trained SAE (`sae`) corresponding to a hook point in the model.
- An input prompt string.

**Expected Outcome:**
- Feature activations for the prompt are computed.
- The features with the highest activation values for a specific token position are identified.

**Tested Code Implementation:**
_Verified via `./tests/saelens/test_feature_activation_analysis.py`._
```python
# Assumes `model` and `sae` are loaded (e.g., using gpt2-small and gpt2-small-res-jb)
import torch 

PROMPT = "The quick brown fox jumps over the lazy"
TOKEN_INDEX_TO_ANALYZE = -1 # Analyze the last token
TOP_K_FEATURES = 5

print(f"Analyzing feature activations for prompt: '{PROMPT}'")

# 1. Tokenize
tokens = model.to_tokens(PROMPT, prepend_bos=True)

# 2. Get original activations from the model
sae.eval() # Ensure SAE is in eval mode
with torch.no_grad():
    _, cache = model.run_with_cache(tokens)
    original_activations = cache[sae.cfg.hook_name] # Shape: [batch, seq_pos, d_in]

    # 3. Encode with SAE to get feature activations
    feature_acts = sae.encode(original_activations) # Shape: [batch, seq_pos, d_sae]

# 4. Extract activations for the target token
# Indexing: [batch_index, token_index, features]
target_token_acts = feature_acts[0, TOKEN_INDEX_TO_ANALYZE, :] # Shape: [d_sae]

# 5. Find top activating features
top_k_values, top_k_indices = torch.topk(target_token_acts, TOP_K_FEATURES)

print(f"Top {TOP_K_FEATURES} activating features for token '{model.to_string(tokens[0, TOKEN_INDEX_TO_ANALYZE])}':")
for i in range(TOP_K_FEATURES):
    print(f"  Feature {top_k_indices[i].item()}: Activation = {top_k_values[i].item():.4f}")

# Example assertion (optional)
assert len(top_k_values) == TOP_K_FEATURES
assert torch.all(top_k_values >= 0) # Activations are usually non-negative (e.g., ReLU)

```

**Implementation Tips:**
- Use `model.run_with_cache` to efficiently get the internal activations needed for the SAE.
- The `sae.encode()` method takes the model's activations and returns the SAE's feature activations.
- Analyze the `feature_acts` tensor. Its dimensions are typically `[batch_size, sequence_length, sae_dimension]`. Index into the desired batch and token position to get feature activations for that specific token.
- `torch.topk` is useful for finding the features with the highest activation values.

**Variants:**
- Change `TOKEN_INDEX_TO_ANALYZE` to look at different positions in the sequence.
- Analyze activations averaged over the sequence or specific token types.
- Run multiple prompts (increase batch size) to see activation patterns across different inputs.

**Troubleshooting:**
- **Shape Mismatches:** Ensure the `sae.cfg.hook_name` corresponds to the activation key in the `cache` and that `sae.cfg.d_in` matches the dimension of the cached activation.
- **Device Mismatches:** Make sure the `model`, `sae`, and input `tokens` are all on the same device (`cpu` or `cuda`).
- **Loading Errors:** Verify that the correct `SAE_RELEASE` and `sae_id` (hook name) are used when loading the SAE with `SAE.from_pretrained`.

### Analyze Feature Logit Contributions

**Description:**
Determine the direct impact of each SAE feature on the model's output vocabulary logits. This helps understand what tokens a feature promotes or inhibits.

**Prerequisites:**
- An initialized base model (`model`). **Crucially, the model must be loaded *with* processing enabled (e.g., using `HookedTransformer.from_pretrained(...)`, *not* `..._no_processing(...)`) to ensure the unembedding matrix (`model.W_U`) is available and correctly processed (e.g., handling layer norm folding).**
- A loaded or trained SAE (`sae`) corresponding to a hook point in the model.

**Expected Outcome:**
- A projection matrix mapping SAE features to logit changes is computed.
- The tokens most strongly influenced (positively or negatively) by selected features are identified.

**Tested Code Implementation:**
_Verified via `./tests/saelens/test_feature_logit_analysis.py`._
```python
# Assumes `model` (loaded with processing) and `sae` are available
import torch

FEATURE_INDICES_TO_ANALYZE = [3380, 300] # Example feature indices
TOP_K_TOKENS = 5

print(f"Analyzing logit contributions for features: {FEATURE_INDICES_TO_ANALYZE}")

# 1. Get SAE decoder and model unembedding matrix
W_dec = sae.W_dec # Shape: [d_sae, d_in]
W_U = model.W_U   # Shape: [d_in, d_vocab]

# 2. Calculate projection matrix (feature contributions to logits)
sae.eval()
model.eval()
with torch.no_grad():
    projection_matrix = W_dec @ W_U # Shape: [d_sae, d_vocab]

print(f"Calculated projection matrix shape: {projection_matrix.shape}")

# 3. Get logit vectors for selected features
selected_feature_logits = projection_matrix[FEATURE_INDICES_TO_ANALYZE, :]

# 4. Find top K tokens promoted by each feature
with torch.no_grad():
    top_k_values, top_k_indices = torch.topk(selected_feature_logits, TOP_K_TOKENS, dim=-1)

print(f"\nTop {TOP_K_TOKENS} positively influenced tokens:")
for i, feature_idx in enumerate(FEATURE_INDICES_TO_ANALYZE):
    print(f"--- Feature {feature_idx} ---")
    tokens = [model.to_string(tok_idx) for tok_idx in top_k_indices[i]]
    values = [f"{val:.2f}" for val in top_k_values[i]]
    print(f"  Tokens: {list(zip(tokens, values))}")

# Example: Find top K tokens *inhibited* by each feature (most negative logits)
with torch.no_grad():
    bottom_k_values, bottom_k_indices = torch.topk(-selected_feature_logits, TOP_K_TOKENS, dim=-1)

print(f"\nTop {TOP_K_TOKENS} negatively influenced tokens:")
for i, feature_idx in enumerate(FEATURE_INDICES_TO_ANALYZE):
    print(f"--- Feature {feature_idx} ---")
    tokens = [model.to_string(tok_idx) for tok_idx in bottom_k_indices[i]]
    values = [f"{-val:.2f}" for val in bottom_k_values[i]] # Negate back for display
    print(f"  Tokens: {list(zip(tokens, values))}")

# Example assertion
assert projection_matrix.shape == (sae.cfg.d_sae, model.cfg.d_vocab)

```

**Implementation Tips:**
- The core calculation is the matrix multiplication `W_dec @ W_U`.
- `W_dec` is the decoder weight matrix of the SAE (`sae.W_dec`).
- `W_U` is the unembedding matrix of the base model (`model.W_U`). Ensure the model was loaded *without* `_no_processing` for `W_U` to be available and correctly processed (e.g., handling layer norm folding if applicable).
- Each row of the resulting `projection_matrix` corresponds to an SAE feature, and the columns correspond to the vocabulary logits.
- Use `torch.topk` on a feature's row to find the tokens it most strongly promotes (highest positive values).
- Use `torch.topk` on the *negative* of a feature's row (`-projection_matrix[feature_idx, :]`) to find the tokens it most strongly inhibits (most negative values).

**Variants:**
- Analyze a larger set of `FEATURE_INDICES_TO_ANALYZE`.
- Visualize the logit distribution for a feature using a histogram.
- Calculate cosine similarity between feature projection vectors to find related features.

**Troubleshooting:**
- **`AttributeError: 'HookedTransformer' object has no attribute 'W_U'`:** Ensure the base model was loaded *with* processing (e.g., `HookedTransformer.from_pretrained(MODEL_NAME)`) instead of `HookedTransformer.from_pretrained_no_processing(...)`.
- **Dimension Mismatch in `W_dec @ W_U`:** Verify that the inner dimensions match (`sae.cfg.d_in` should equal `model.cfg.d_model` or the input dimension of `W_U`).
- **Device Mismatches:** Ensure `sae.W_dec` and `model.W_U` are on the same device.

### Integrated Train -> Load -> Analyze Demo

The script `./tests/saelens/integrated_pipeline_demo.py` demonstrates the complete workflow:
1.  **Training:** Briefly trains a new SAE on a small model (`tiny-stories-1M`) using the configurations defined in the script.
2.  **Saving:** Saves the trained SAE checkpoint to a temporary directory.
3.  **Loading:** Loads the *just-trained* SAE from the temporary checkpoint directory using `TrainingSAE.load_from_pretrained()`.
4.  **Analysis:** Performs feature activation and logit contribution analysis on the loaded SAE, similar to the individual analysis sections above.

This demo verifies that the components (training, saving, loading, analysis) interface correctly and provides a template for building end-to-end SAE workflows.

```python
# Key snippet from ./tests/saelens/integrated_pipeline_demo.py showing the load step:

# ... (Training and saving happens earlier in the script) ...

import torch
from transformer_lens import HookedTransformer
from sae_lens.training.training_sae import TrainingSAE

# Assuming `saved_checkpoint_path` contains the path to the saved checkpoint directory
# and `DEVICE` is set (e.g., "cpu" or "cuda")
saved_checkpoint_path = "/path/to/your/checkpoint/run_id/final_12345" # Example path
DEVICE = "cpu"

# Load the model with processing for analysis
model_analyze = HookedTransformer.from_pretrained("tiny-stories-1M", device=DEVICE)
model_analyze.eval()

print(f"Loading trained SAE from checkpoint directory: {saved_checkpoint_path}...")
try:
    # Use TrainingSAE.load_from_pretrained to load the saved TrainingSAE object
    sae = TrainingSAE.load_from_pretrained(saved_checkpoint_path, device=DEVICE)
    sae.eval()
    print("Trained SAE loaded successfully.")
    # Basic check
    assert sae.cfg.d_in == model_analyze.cfg.d_model # d_model includes processing adjustments
    
    # ... (Analysis code follows) ...
    
except Exception as e:
    print(f"Failed to load SAE from checkpoint: {e}")
    # Handle error


```

---

## Evaluating SAE Performance

After training and analyzing an SAE, evaluating its performance quantitatively is essential. This typically involves measuring metrics like reconstruction loss, sparsity (L0 norm), and the effect of the SAE on the base model's downstream task performance (e.g., perplexity).

---

## Integration with Other Packages

### Issue: Training with Pre-Computed Activations from External Sources (e.g., `nnsight` or `dictionary_learning` `ActivationBuffer`)

**Description:** 
The `sae_lens.training.sae_trainer.SAETrainer` class, as shown in the guide's examples, is designed to work closely with an `sae_lens.training.activations_store.ActivationsStore` instance. This ActivationStore handles fetching data and running the base model (`HookedTransformer`) to generate activations on-the-fly during training. Attempting to train a `sae-lens` SAE on activations generated externally (e.g., using `nnsight` and cached, or loaded from disk) presents significant challenges. Directly substituting the `ActivationStore` with a different data provider is not supported due to interface mismatches. Furthermore, implementing a manual training loop that accurately replicates the behavior of the internal `SAETrainer` using external activation batches is complex and error-prone. **Practical attempts to combine `nnsight` (following its guide for activation access) with `SAELens` for training confirmed this incompatibility; the standard `SAELens` trainer could not be readily adapted.**

**Integration Tips & Workaround:**
- **Compatibility:** The standard `sae-lens` training workflow is primarily intended for use with activations generated on-the-fly via `HookedTransformer` and the built-in `ActivationsStore`. Using activations generated by other methods (like `nnsight`'s trace/intervention focused approach) is not directly supported by the trainer as documented.
- **Manual Loop Complexity:** Avoid manual training loops unless deeply familiar with `sae-lens` internals, as replicating behavior is difficult.
- **Recommendation for Cross-Package Comparison:** If comparing SAEs trained on identical activation data is the goal, consider using a framework designed to accept pre-computed activation iterators (like `dictionary_learning`) for all SAEs being compared, even if it means not using the native `sae-lens` trainer.

**Debugging Steps Taken (Illustrative):**
- Attempted to initialize `SAETrainer` with custom iterators yielding pre-computed activations. Encountered `TypeError` due to mismatched `activation_store` type.
- Analysis during `single_layer_reconstruction` experiment confirmed the lack of a clear path to integrate `nnsight` activation generation (as described in its guide) with the `SAETrainer`'s requirements.

--- 
