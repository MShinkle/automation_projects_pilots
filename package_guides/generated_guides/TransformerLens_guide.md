# TransformerLens Package Examples and Usage

This document provides a comprehensive guide to using the TransformerLens package, covering key functionalities.

## Table of Contents
1. [Loading Models](#loading-models)
2. [Running Models & Caching Activations](#running-models--caching-activations)
3. [Using Hook Points](#using-hook-points)
4. [Activation Patching](#activation-patching)
5. [Model Evaluation](#model-evaluation)
6. [Component Analysis](#component-analysis)
7. [Working with Specific Architectures](#working-with-specific-architectures)
8. [Utilities](#utilities)
9. [Gradient Computation (Autograd)](#gradient-computation-autograd)

---

## Loading Models

### Load a Standard GPT-Style Model

**Description:**
Load a standard decoder-only transformer model (like GPT-2) from Hugging Face into the `HookedTransformer` format using the `from_pretrained` class method.

**Prerequisites:**
- A valid Hugging Face model identifier for a compatible GPT-style model (e.g., `"gpt2-small"`, `"EleutherAI/pythia-70m-deduped"`).
- Optional: Desired device (`"cuda"`, `"cpu"`).

**Expected Outcome:**
- A functional `transformer_lens.HookedTransformer` object is returned.
- The model is loaded onto the specified device.
- The `model.cfg` attribute contains the correct configuration, though the `model_name` might be simplified (e.g., `"gpt2"` for `"gpt2-small"`).
- The model object contains the expected architectural components (e.g., `model.blocks`).

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_load_gpt_model.py`_
```python
import torch
import transformer_lens

# --- Configuration ---
MODEL_NAME = "gpt2-small" # Or other HF model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Attempting to load model: {MODEL_NAME} onto device: {DEVICE}")

try:
    # Load the model using TransformerLens' from_pretrained method
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE
    )
    
    print(f"Successfully loaded model: {MODEL_NAME}")
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Model has {len(model.blocks)} blocks.")
    # Basic check of config
    assert model.cfg.d_model > 0
    print(f"Model config loaded (model name: {model.cfg.model_name})")

except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
```

**Implementation Tips:**
- `HookedTransformer.from_pretrained` is the primary way to load models.
- It automatically downloads model weights and configuration from Hugging Face.
- Specify the `device` argument to control placement (defaults usually to CPU if CUDA is unavailable or not specified).
- Be aware that the `model.cfg.model_name` might be a simplified version of the requested Hugging Face identifier (e.g., loading `"gpt2-small"` results in `model.cfg.model_name == "gpt2"`).
- The `from_pretrained` method takes other arguments like `revision`, `trust_remote_code`, etc., similar to Hugging Face Transformers.

**Variants:**
- **Different Models:** Replace `"gpt2-small"` with other compatible Hugging Face model IDs (e.g., `"gpt2-medium"`, `"distilgpt2"`, `"EleutherAI/pythia-160m"`).
- **CPU Loading:** Set `device="cpu"`.
- **Specific Revision:** Use `revision="main"` or a specific commit hash/branch name.
- **Trust Remote Code:** Add `trust_remote_code=True` if loading models that require custom code execution (use with caution).

**Troubleshooting:**
- **`OSError` / `RepositoryNotFoundError`:** The model identifier `MODEL_NAME` is likely incorrect or doesn't exist on the Hugging Face Hub. Check the spelling and availability.
- **Memory Errors (OOM):** Loading large models requires significant RAM/VRAM. Ensure your device has sufficient memory, or load on CPU.
- **Device Errors:** Ensure CUDA is installed correctly if using `device="cuda"`. Errors might occur if the requested device isn't available.
- **Model Name Mismatch:** As noted in Tips, `model.cfg.model_name` might differ slightly from the input `MODEL_NAME`. Check the loaded config if exact name matching is required downstream.

### Load a Model with Configuration Overrides

**Description:**
Load a model using `from_pretrained` while overriding specific configuration parameters. This allows modifying aspects like activation functions, dropout rates, etc., based on the arguments accepted by the underlying Hugging Face model's configuration.

**Prerequisites:**
- A valid Hugging Face model identifier (e.g., `"gpt2-small"`).
- A dictionary containing keys and values for the configuration parameters to override. **Important:** Keys must generally correspond to valid arguments for the underlying Hugging Face model's configuration (`AutoConfig.from_pretrained`) or model loading (`AutoModel.from_pretrained`). Overriding `HookedTransformer`-specific config keys that don't map to HF keys via this method is unreliable.

**Expected Outcome:**
- A `transformer_lens.HookedTransformer` object is returned.
- The model is loaded with the specified configuration overrides applied.
- The `model.cfg` attribute reflects the overridden values (note that `HookedTransformer` might use different attribute names, e.g., `activation_function` in HF config maps to `act_fn` in `model.cfg`).

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_load_model_override.py`_
```python
import torch
import transformer_lens

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuration Overrides ---
# Override a key that exists in the underlying Hugging Face config.
# Default for gpt2 is 'gelu_new'. We override it to 'relu'.
CFG_OVERRIDES = {
    "activation_function": "relu",
}

print(f"Attempting to load model: {MODEL_NAME} with overrides: {CFG_OVERRIDES}")

try:
    # Pass the overrides as keyword arguments using **
    model = transformer_lens.HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        **CFG_OVERRIDES 
    )
    
    print(f"Successfully loaded model: {MODEL_NAME} with overrides.")
    
    # Verify the override was applied 
    # (HF 'activation_function' maps to TL 'act_fn')
    expected_act_fn = CFG_OVERRIDES["activation_function"]
    assert model.cfg.act_fn == expected_act_fn
    print(f"Config override verified: act_fn is now {model.cfg.act_fn}")

except Exception as e:
    print(f"Error loading model {MODEL_NAME} with overrides: {e}")
```

**Implementation Tips:**
- Pass the override dictionary as keyword arguments using the `**` operator to `from_pretrained`.
- Consult the Hugging Face documentation for the specific model type to know which configuration keys can be overridden (e.g., look at the `Config` class for the model, like `GPT2Config`).
- Be aware of potential name mapping between Hugging Face config keys and `HookedTransformerConfig` attribute names (e.g., `activation_function` -> `act_fn`). Check the `HookedTransformerConfig` source code or experiment to confirm the exact attribute mapping in TransformerLens for the key you wish to override.
- Overriding core architectural parameters (like activation function, layer counts) on a pretrained model might negatively impact performance or cause errors if the weights are incompatible with the new configuration.

**Variants:**
- **Different Overrides:** Override other valid HF config keys like `n_head`, `n_layer` (if creating a model from scratch or if compatible), `attn_pdrop`, `resid_pdrop`, etc.

**Troubleshooting:**
- **`TypeError: ... got an unexpected keyword argument ...`:** The key provided in the override dictionary is not a valid argument for the underlying Hugging Face `AutoConfig.from_pretrained` or `AutoModelForCausalLM.from_pretrained` calls for that specific model. Double-check the key against HF documentation.
- **AttributeError / Incompatibility:** Overriding structural parameters might lead to errors during model initialization or runtime if the pretrained weights don't match the new structure.

### Load a BERT-Style Model (using HookedEncoder)

**Description:**
Load a BERT-style (encoder-only) model from Hugging Face. **Note:** BERT models should generally be loaded using `transformer_lens.HookedEncoder`, not `HookedTransformer`, as the latter is primarily designed for decoder-only models and may encounter issues with BERT configurations.

**Prerequisites:**
- A valid Hugging Face model identifier for a BERT model (e.g., `"bert-base-uncased"`, `"google-bert/bert-large-uncased"`).
- Optional: Desired device (`"cuda"`, `"cpu"`).

**Expected Outcome:**
- A functional `transformer_lens.HookedEncoder` object is returned.
- The model is loaded onto the specified device.
- The `model.cfg` attribute contains the correct configuration, including identifying the specific BERT architecture loaded (e.g., `BertForMaskedLM`).
- The model object contains the expected encoder components (`model.blocks`).

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_load_bert_model.py`_
```python
import torch
import transformer_lens
from transformer_lens import HookedEncoder # Important!

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Attempting to load BERT model: {MODEL_NAME} using HookedEncoder...")

try:
    # Use HookedEncoder.from_pretrained for BERT
    model = HookedEncoder.from_pretrained(
        MODEL_NAME,
        device=DEVICE
    )
    
    print(f"Successfully loaded model: {MODEL_NAME} using HookedEncoder.")
    
    # --- Verification ---
    assert isinstance(model, HookedEncoder)
    # Verify specific architecture loaded (often BertForMaskedLM)
    assert model.cfg.original_architecture == "BertForMaskedLM"
    assert hasattr(model, 'blocks') and len(model.blocks) > 0
    assert next(model.parameters()).device.type == DEVICE
    print(f"Verification successful. Loaded {model.cfg.original_architecture}")

except Exception as e:
    print(f"Error loading BERT model {MODEL_NAME} with HookedEncoder: {e}")
```

**Implementation Tips:**
- Always use `transformer_lens.HookedEncoder.from_pretrained` for loading standard BERT models to ensure correct configuration handling.
- The `HookedEncoder` class provides a similar API to `HookedTransformer` but is tailored for encoder architectures.
- Be aware of the warning regarding experimental BERT support; functionality might be less extensive or stable compared to GPT models in `HookedTransformer`.
- `HookedEncoder` loads the specific architecture from Hugging Face (e.g., `BertForMaskedLM`), which will be reflected in `model.cfg.original_architecture`.

**Variants:**
- **Different BERT Models:** Replace `"bert-base-uncased"` with other BERT variants (e.g., `"bert-large-cased"`, `"distilbert-base-uncased"`). Check if `HookedEncoder` supports them.
- **CPU Loading:** Set `device="cpu"`.

**Troubleshooting:**
- **Using `HookedTransformer`:** Attempting to load BERT with `HookedTransformer` is likely to fail due to configuration mismatches (`KeyError: 'normalization_type'`, etc.). Use `HookedEncoder` instead.
- **Architecture Mismatch:** If assertions about `model.cfg.original_architecture` fail, check the actual loaded architecture name printed in the config or error message.
- **Memory Errors:** Large BERT models still require significant memory.

---

## Running Models & Caching Activations

### Run Model with Forward Pass (Get Logits)

**Description:**
Execute a standard forward pass of a loaded `HookedTransformer` model on input text to obtain the output logits.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object (e.g., from `from_pretrained`).
- Input text (string or list of strings).
- Optional: Set model to evaluation mode using `model.eval()` if not training.

**Expected Outcome:**
- The model call returns a `torch.Tensor` containing the output logits.
- The logits tensor has the shape `[batch_size, sequence_length, d_vocab]`.
- The tensor is on the same device as the model.

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_run_model_logits.py`_
```python
import torch
import transformer_lens

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "The quick brown fox jumps over the lazy dog."

# Load the model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval() # Set to evaluation mode is good practice

# Tokenize the input text
# HookedTransformer has a built-in tokenizer accessed via model.tokenizer
# or convenience methods like to_tokens
input_ids = model.to_tokens(PROMPT) 
# input_ids shape: [1, sequence_length]
print(f"Input shape: {input_ids.shape}")

# Run the forward pass by calling the model object
with torch.no_grad(): # Important for inference
    logits = model(input_ids)

print("Forward pass completed.")

# --- Verification ---
assert logits is not None
assert isinstance(logits, torch.Tensor)
expected_shape = (input_ids.shape[0], input_ids.shape[1], model.cfg.d_vocab)
assert logits.shape == expected_shape
assert logits.device.type == model.cfg.device
print(f"Logits shape: {logits.shape}")
print(f"Logits device: {logits.device}")
```

**Implementation Tips:**
- Make the `HookedTransformer` object callable directly (`model(input_ids)`) to perform a forward pass.
- Use the model's built-in tokenizer methods like `model.to_tokens()` (for strings) or `model.to_batch()` (for lists of strings) to prepare input IDs correctly, handling padding and BOS tokens according to the model's configuration.
- Wrap the forward pass in `with torch.no_grad():` during inference to save memory and computation by disabling gradient tracking.
- Ensure the model is in evaluation mode (`model.eval()`) to disable dropout and other training-specific layers, unless you specifically want to run with dropout enabled.

**Variants:**
- **Batch Input:** Pass a list of strings to `model.to_tokens()` or use `model.to_batch()`. The `batch_size` dimension of the output logits will correspond to the number of input strings.
  ```python
  prompts = ["Sentence 1.", "Another sentence."]
  input_ids = model.to_tokens(prompts) # Handles padding
  # input_ids shape: [2, max_sequence_length]
  logits = model(input_ids)
  # logits shape: [2, max_sequence_length, d_vocab]
  ```
- **Return Different Outputs:** The forward pass can return different things based on the `return_type` argument (e.g., `"loss"`, `"embeddings"`). See `HookedTransformer.forward` documentation. The default is `"logits"`.

**Troubleshooting:**
- **Shape Mismatches:** Ensure input IDs have the correct shape (usually `[batch_size, sequence_length]`). Check the output shape matches expectations.
- **Device Mismatches:** Ensure input tensors are on the same device as the model (`model.cfg.device`). `model.to_tokens` handles this automatically.
- **Memory Errors (OOM):** Processing long sequences or large batches requires significant memory. Reduce sequence length or batch size.
- **Incorrect Tokenization:** Verify that the tokenization (especially padding and BOS/EOS tokens) matches the model's requirements. `model.to_tokens` usually handles this correctly based on the loaded config.

### Run Model and Cache Activations (`run_with_cache`)

**Description:**
Execute a forward pass while simultaneously caching specified internal activations of the model using the `run_with_cache` method. This is a core functionality for mechanistic interpretability, allowing inspection of intermediate states.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.
- Input text (string or list of strings).

**Expected Outcome:**
- The method returns a tuple: `(output, cache)`.
- `output`: The standard model output (logits by default, same as a direct forward pass).
- `cache`: A `transformer_lens.ActivationCache` object (which behaves like a dictionary) containing the cached activations as `torch.Tensor` objects. Keys are strings identifying the hook point (e.g., `"blocks.0.attn.hook_z"`).
- Cached tensors have the expected shapes (e.g., `[batch, pos, d_model]` for residual stream, `[batch, pos, n_heads, d_head]` for attention outputs) and are on the model's device.

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_run_with_cache.py`_
```python
import torch
import transformer_lens
from transformer_lens import ActivationCache

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Caching activations helps analysis."

# Load model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

# Tokenize
input_ids = model.to_tokens(PROMPT)

# Run with cache
with torch.no_grad():
    # run_with_cache returns (output, cache_dict)
    logits, cache = model.run_with_cache(input_ids)

print("run_with_cache completed.")

# --- Verification ---
# 1. Verify Logits
assert isinstance(logits, torch.Tensor)
assert logits.shape[-1] == model.cfg.d_vocab
print(f"Logits shape: {logits.shape}")

# 2. Verify Cache
assert isinstance(cache, ActivationCache)
print(f"Cache object type: {type(cache)}")

# 3. Access and verify a specific cached activation
layer = 0
activation_key = f"blocks.{layer}.hook_resid_post"
assert activation_key in cache

resid_post_act = cache[activation_key]
assert isinstance(resid_post_act, torch.Tensor)
expected_shape = (input_ids.shape[0], input_ids.shape[1], model.cfg.d_model)
assert resid_post_act.shape == expected_shape
assert resid_post_act.device.type == model.cfg.device

print(f"Successfully accessed and verified: {activation_key}")
print(f"Shape: {resid_post_act.shape}, Device: {resid_post_act.device}")
```

**Implementation Tips:**
- `run_with_cache` is the primary method for obtaining internal activations.
- By default, it caches a standard set of useful activations including residual stream states (`hook_resid_pre`, `hook_resid_mid`, `hook_resid_post`), attention outputs (`hook_attn_out`, `hook_q`, `hook_k`, `hook_v`, `hook_z`), attention patterns (`hook_pattern`), MLP layer outputs (`hook_mlp_out`, `hook_post`), and layer norm scales (`hook_scale`). You don't need to specify hooks manually for these common cases. Check `model.hook_points` for exact names.
- The returned `cache` object acts like a dictionary. Access activations using string keys corresponding to hook point names (e.g., `cache["blocks.0.mlp.hook_post"]`).
- Use `model.hook_points` to see the available hook points and their names on the specific loaded model instance.
- Like the standard forward pass, wrap `run_with_cache` in `with torch.no_grad():` during inference.

**Variants:**
- **Caching Specific Activations:** Pass a `names_filter` argument to `run_with_cache` to specify exactly which activations to cache. This can save memory if you only need a subset. The filter can be a list of strings, a single string, or a function `Callable[[str], bool]`.
  ```python
  # Example 1: Cache only layer 0 attention output (hook_z) and final layer norm scale
  names_filter = ["blocks.0.attn.hook_z", "ln_final.hook_scale"]
  logits, cache = model.run_with_cache(input_ids, names_filter=names_filter)
  print(f"Cached keys with list filter: {list(cache.keys())}")

  # Example 2: Cache all residual stream outputs using a lambda function
  names_filter_fn = lambda name: "hook_resid" in name
  logits_resid, cache_resid = model.run_with_cache(input_ids, names_filter=names_filter_fn)
  print(f"Cached keys with function filter: {list(cache_resid.keys())}")
  ```
- **Different Return Type:** Use `return_type` (e.g., `"loss"`) similarly to the standard forward method. `run_with_cache` will still return `(return_value, cache)`.

**Troubleshooting:**
- **`KeyError`:** The requested activation key (hook point name) does not exist in the cache. Double-check the spelling against `model.hook_points` or ensure the default caching includes it. If using `names_filter`, ensure the filter matches existing hook names.
- **Memory Errors (OOM):** Caching activations, especially for long sequences or large models, consumes significant memory. Use `names_filter` to cache only what's necessary.

## Activation Patching

This section covers techniques for modifying activations during a forward pass to understand their causal effects. Patching typically involves:
1. A **clean run** on a source input (e.g., "The Eiffel Tower is located in") and caching its activations.
2. A **corrupted run** on a target input (e.g., "The Colosseum is in").
3. **Patching:** During the corrupted run, replacing a specific activation (e.g., layer 5 attention output) with the corresponding activation from the clean cache.
4. Measuring the **effect** of the patch on a chosen metric (e.g., the logit difference between "Paris" and "Rome" at the final position).

This fundamental technique is the basis for more advanced methods like **Path Patching** and **Causal Tracing**, which aim to identify important computational pathways within the model by selectively patching multiple activations.

TransformerLens provides both low-level hooks for manual patching (see [Add a Hook to Modify Activations](#add-a-hook-to-modify-patch-activations)) and higher-level utility functions in `transformer_lens.patching` that automate iterating through different patch locations.

### Use Patching Utilities (e.g., `get_act_patch_attn_head_all_pos_every`)

**Description:**
Use high-level patching functions from `transformer_lens.patching` to systematically patch different components (like attention heads or MLP layers across different layers) and measure the impact on a defined metric. This example uses `get_act_patch_attn_head_all_pos_every` to patch attention head components.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.
- Clean input tokens and corrupted input tokens (often requiring the same sequence length, depending on the specific patching function).
- A `transformer_lens.ActivationCache` object generated from a run on the clean input tokens.
- A metric function `Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]]` that takes the model's output logits and returns a scalar value representing the metric of interest.

**Expected Outcome:**
- The patching function (e.g., `get_act_patch_attn_head_all_pos_every`) returns a `torch.Tensor` containing the metric value for each patch applied.
- The shape of the tensor depends on the function used. For `get_act_patch_attn_head_all_pos_every`, the shape is `[patch_type, n_layers, n_heads]`, where `patch_type` corresponds to patching different attention components (output, query, key, value, pattern).

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_patching_util.py`_
```python
import torch
import transformer_lens
from transformer_lens import patching, utils

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLEAN_PROMPT = "The Eiffel Tower is located in"
CORRUPTED_PROMPT = "The Colosseum is located in"
ANSWER = " Paris"
WRONG_ANSWER = " Rome"

# Load Model & Prepare Inputs
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()
clean_tokens = model.to_tokens(CLEAN_PROMPT)
corrupted_tokens = model.to_tokens(CORRUPTED_PROMPT)
assert clean_tokens.shape == corrupted_tokens.shape # Ensure same length

# Define Patching Metric (Logit difference)
answer_token_index = model.to_single_token(ANSWER)
wrong_answer_token_index = model.to_single_token(WRONG_ANSWER)
def logit_diff_metric(logits):
    final_token_logits = logits[:, -1, :]
    return (final_token_logits[:, answer_token_index] - 
            final_token_logits[:, wrong_answer_token_index]).mean()

# Generate Clean Cache
with torch.no_grad():
    _, clean_cache = model.run_with_cache(clean_tokens)

# Run Patching Function (Example: Patching all attention head components)
# This function iterates through layers, heads, and patch types (out, q, k, v, pattern)
print(f"Running attention head patching utility...")
with torch.no_grad():
    patching_results = patching.get_act_patch_attn_head_all_pos_every(
        model=model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=clean_cache,
        metric=logit_diff_metric
    )
print("Patching function completed.")

# --- Verification ---
assert isinstance(patching_results, torch.Tensor)
# Shape: [patch_type=5, layer, head_index]
expected_shape = (5, model.cfg.n_layers, model.cfg.n_heads)
assert patching_results.shape == expected_shape
assert patching_results.device.type == DEVICE
print(f"Patching results verified. Shape: {patching_results.shape}")
# Example: Access result for patching Value vectors (type 3) in layer 5, head 10
# value_patch_L5H10 = patching_results[3, 5, 10] 
# print(f"Example result (Patch V, L5, H10): {value_patch_L5H10.item():.4f}")
```

**Implementation Tips:**
- Choose the appropriate patching function from `transformer_lens.patching` based on the components you want to patch (e.g., `get_act_patch_attn_head_...`, `get_act_patch_mlp_...`, `get_act_patch_resid_...`). The library offers functions for patching attention head outputs, Q, K, V, patterns, MLP layers, and residual stream components.
- Carefully define your `metric` function. It should quantify the change in model behavior you care about (e.g., change in loss, probability of correct token, difference between logits).
- Ensure the `clean_cache` is generated from the appropriate clean input.
- The dimensions of the returned results tensor correspond to the components iterated over by the specific patching function.
- These functions internally use `model.run_with_hooks`, applying the necessary patching hooks for each iteration.
- For more complex or custom patching scenarios not covered by the utility functions, you can directly use `model.run_with_hooks` and define your own hook functions to modify activations. See the documentation for [Using Hook Points](#using-hook-points).

**Variants:**
- **Different Patching Functions:** Explore other functions in `patching.py` like `get_act_patch_mlp_every` (patching MLP layers), `get_act_patch_resid_pre` (patching residual stream before blocks), `get_act_patch_attn_head_by_pos_every` (patching specific positions within heads).
- **Different Metrics:** Implement custom metric functions (e.g., based on KL divergence, entropy, probability of a specific sequence).

**Troubleshooting:**
- **Shape Mismatches:** Ensure the clean and corrupted token sequences have compatible shapes if required by the specific patching function. Double-check the expected output shape of the chosen patching utility.
- **Metric Function Errors:** Debug the metric function to ensure it correctly handles the output logits tensor and returns a scalar.
- **Performance:** Running patching utilities can be computationally intensive as they involve many forward passes. Consider patching subsets of layers/heads if necessary.

## Model Evaluation

TransformerLens models provide standard outputs like logits and can calculate loss, which are necessary for most evaluation tasks. The actual evaluation logic (comparing outputs to targets, calculating metrics like accuracy, perplexity, F1 score, etc.) typically relies on standard PyTorch operations or external evaluation libraries/scripts tailored to the specific task.

### Obtain Logits and Loss for Evaluation

**Description:**
Run the model on input data to obtain the output logits and the loss (typically cross-entropy loss for language modeling). These outputs are the basis for most downstream evaluation metrics.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.
- Input data (tokenized input IDs tensor, usually `[batch_size, sequence_length]`).

**Expected Outcome:**
- **Logits:** Calling `model(input_ids, return_type="logits")` (or just `model(input_ids)`) returns the logits tensor of shape `[batch, pos, d_vocab]`.
- **Loss:** Calling `model(input_ids, return_type="loss")` returns a scalar tensor containing the mean cross-entropy loss, calculated internally by predicting the next token based on the input sequence (i.e., using the input sequence shifted left as labels).
- Outputs are on the model's device.

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_get_eval_outputs.py`_
```python
import torch
import transformer_lens

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Evaluating model outputs requires logits or loss."

# Load model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()
input_ids = model.to_tokens(PROMPT)

with torch.no_grad():
    # --- Get Logits ---
    logits = model(input_ids, return_type="logits")
    print(f"Logits obtained. Shape: {logits.shape}")
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], model.cfg.d_vocab)

    # --- Get Loss (Internal Calculation) ---
    # Calculates standard next-token prediction loss
    loss = model(input_ids, return_type="loss")
    print(f"Loss obtained: {loss.item():.4f}")
    assert loss.ndim == 0 # Loss should be a scalar

# --- Using Outputs for Evaluation (Example) ---
# 1. Get probabilities from logits
probs = torch.softmax(logits, dim=-1)
# 2. Get top predicted token index for the last position
top_pred_idx = probs[:, -1, :].argmax(dim=-1)
# 3. Decode the predicted token
top_pred_token = model.to_string(top_pred_idx)
print(f"Predicted next token (greedy): '{top_pred_token}'")

# 4. Use the loss directly (e.g., for perplexity calculation)
perplexity = torch.exp(loss)
print(f"Perplexity (based on internal loss): {perplexity.item():.4f}")

# Further evaluation would involve comparing predictions/loss against ground truth labels
# Consider using libraries like Hugging Face's `evaluate` for standard metrics.
```

**Implementation Tips:**
- Use `return_type="logits"` (or the default) in the forward pass to get logits, which are needed for calculating probabilities, finding argmax predictions, etc.
- Use `return_type="loss"` to get the model's internally calculated cross-entropy loss for causal language modeling (predicting the next token). This is often used for calculating perplexity.
- If you need loss for a different task (e.g., classification task, using specific labels) or using specific labels (not just the next token), you must obtain the `logits` and calculate the loss manually using appropriate PyTorch loss functions (like `torch.nn.CrossEntropyLoss`) and correctly formatted labels.
- Remember `model.eval()` and `with torch.no_grad():` for evaluation.
- For standardized evaluation metrics (Accuracy, F1, BLEU, ROUGE, etc.), consider using dedicated libraries like Hugging Face's `evaluate` package, feeding it the model's predictions (derived from logits) and ground truth labels.

**Variants:**
- **Batch Evaluation:** Pass a batch of inputs to `model.to_tokens()` and the forward pass. Logits and loss will be calculated for the batch.
- **Per-Token Loss:** Use `loss_per_token=True` along with `return_type="loss"` or `return_type="both"` to get the loss for each token position instead of the mean loss.
  ```python
  # loss_per_tok shape: [batch, pos-1]
  loss_per_tok = model(input_ids, return_type="loss", loss_per_token=True)
  ```
- **Getting Both:** Use `return_type="both"` to get a tuple `(logits, loss)`.

**Troubleshooting:**
- **Unexpected Loss Value:** Ensure the model is being run on the intended task. The internal `return_type="loss"` assumes causal language modeling. If evaluating a classification task, you need to calculate loss manually using the logits and your specific labels.
- **Shape Errors in Manual Loss:** When calculating loss manually, ensure logits and labels are correctly reshaped and aligned for the chosen loss function (e.g., `CrossEntropyLoss` expects logits of shape `[N, C]` and labels of shape `[N]`).

## Component Analysis

This section focuses on analyzing specific internal components or their outputs, going beyond simple patching.

### Get and Visualize Attention Patterns

**Description:**
Retrieve the attention patterns calculated during a forward pass. Attention patterns show the distribution of attention scores (softmax output) for each query position over all key positions, for each attention head.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.
- Input data (tokenized IDs).

**Expected Outcome:**
- Running the model with `run_with_cache` populates the cache with attention patterns.
- The attention pattern for a specific layer can be accessed from the cache using the key `blocks.{layer}.attn.hook_pattern`.
- The retrieved tensor has the shape `[batch, n_heads, sequence_length, sequence_length]` and contains values between 0 and 1, summing to 1 along the last dimension (key positions).

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_get_attn_pattern.py`_
```python
import torch
import transformer_lens

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Attention patterns show focus."

# Load model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()
input_ids = model.to_tokens(PROMPT)

# --- Run with Cache --- 
# Attention patterns are typically cached by default
with torch.no_grad():
    logits, cache = model.run_with_cache(input_ids)

print("Ran model with cache.")

# --- Access and Verify Pattern ---
layer_to_inspect = 1
pattern_hook_name = f"blocks.{layer_to_inspect}.attn.hook_pattern"

assert pattern_hook_name in cache, f"Hook '{pattern_hook_name}' not found."
attn_pattern = cache[pattern_hook_name]

print(f"Accessed pattern for layer {layer_to_inspect}.")
assert isinstance(attn_pattern, torch.Tensor)

# Expected shape: [batch, n_heads, query_pos, key_pos]
expected_shape = (input_ids.shape[0], model.cfg.n_heads, input_ids.shape[1], input_ids.shape[1])
assert attn_pattern.shape == expected_shape, f"Shape mismatch: {attn_pattern.shape}"

# Check probabilities sum to 1 over keys
assert torch.allclose(attn_pattern.sum(dim=-1), torch.ones(1, device=DEVICE), atol=1e-5)
print(f"Pattern verified. Shape: {attn_pattern.shape}")

# --- Basic Visualization --- 
# (Use circuitsvis.attention.attention_patterns for proper visualization)
head_index = 0
last_token_pattern = attn_pattern[0, head_index, -1, :].cpu().numpy()
str_tokens = model.to_str_tokens(input_ids[0])
print(f"\nAttention from last token ('{str_tokens[-1]}') by Head {layer_to_inspect}.{head_index}:")
for token, score in zip(str_tokens, last_token_pattern):
    print(f"  -> '{token}': {score:.3f}")
```

**Implementation Tips:**
- Use `run_with_cache` to get the `ActivationCache`.
- Access the pattern for layer `L` using `cache[f"blocks.{L}.attn.hook_pattern"]`.
- The resulting tensor shape is `[batch, n_heads, query_pos, key_pos]`.
- For rich, interactive visualization of attention patterns, use the `circuitsvis` library, specifically `circuitsvis.attention.attention_patterns`. Manually plotting large patterns can be cumbersome.
  ```python
  # Example (assuming circuitsvis is installed and cache obtained)
  import circuitsvis as cv 
  layer = 0
  attn_pattern_l0 = cache[f"blocks.{layer}.attn.hook_pattern"]
  str_tokens = model.to_str_tokens(input_ids[0]) 
  # Display in Jupyter/compatible environment
  cv.attention.attention_patterns(tokens=str_tokens, attention=attn_pattern_l0[0]).show() 
  ```

**Variants:**
- **Different Layers:** Change `layer_to_inspect` to view patterns from other layers.
- **Specific Heads/Positions:** Slice the retrieved `attn_pattern` tensor to focus on specific heads or query positions.

**Troubleshooting:**
- **`KeyError`:** Ensure `run_with_cache` was used (simple forward pass doesn't cache). Verify the hook point name is correct. The default `run_with_cache` should include `hook_pattern`.
- **Shape Errors:** Double-check the expected dimensions against the retrieved tensor's shape.

### Access Model Weights (e.g., for OV Circuits)

**Description:**
Directly access the weight matrices of the model components (embeddings, attention layers, MLPs, unembedding). TransformerLens provides convenient stacked properties (e.g., `model.W_V`) for accessing weights across all layers/heads simultaneously, as well as access to individual layer weights. This is useful for analyzing weights directly or calculating combined circuits like the OV (Output-Value) or QK (Query-Key) circuits.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.

**Expected Outcome:**
- Properties like `model.W_E`, `model.W_V`, `model.W_O`, `model.W_Q`, `model.W_K`, `model.W_in`, `model.W_out`, `model.W_pos`, `model.W_U` return the corresponding weight tensors.
- For layer-specific weights (`W_V`, `W_O`, etc.), the tensors are stacked across layers, typically having `n_layers` as the first dimension.
- The shapes of the returned tensors match the model configuration (`d_model`, `d_vocab`, `n_layers`, `n_heads`, `d_head`, `d_mlp`

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_get_weights.py`_
```python
import torch
import transformer_lens

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()

# --- Method 1: Access Stacked Weights ---
W_V_stacked = model.W_V
W_O_stacked = model.W_O

print(f"\nAccessed stacked W_V: {W_V_stacked.shape}")
print(f"Accessed stacked W_O: {W_O_stacked.shape}")

# --- Method 2: Access Layer-by-Layer Weights ---
layer_idx = 1
head_idx = 2 
layer_block = model.blocks[layer_idx]
W_V_layer = layer_block.attn.W_V # Shape: [n_heads, d_model, d_head]
W_O_layer = layer_block.attn.W_O # Shape: [n_heads, d_head, d_model]
b_V_layer = layer_block.attn.b_V # Shape: [n_heads, d_head]
W_in_layer = layer_block.mlp.W_in # Shape: [d_model, d_mlp]
b_in_layer = layer_block.mlp.b_in # Shape: [d_mlp]

print(f"\nAccessed layer {layer_idx} W_V_layer: {W_V_layer.shape}")
print(f"Accessed layer {layer_idx} W_O_layer: {W_O_layer.shape}")
print(f"Accessed layer {layer_idx} b_V_layer: {b_V_layer.shape}")
print(f"Accessed layer {layer_idx} W_in_layer: {W_in_layer.shape}")
print(f"Accessed layer {layer_idx} b_in_layer: {b_in_layer.shape}")

# Verification: Compare layer-wise access to slice of stacked property
assert torch.allclose(W_V_layer, W_V_stacked[layer_idx])
assert torch.allclose(W_O_layer, W_O_stacked[layer_idx])
# Note: Stacked bias properties (like model.b_V) might exist but are less common than weights
print(f"Verified layer-{layer_idx} weights match stacked weights.")

**Available Weight Properties (Common Stacked Properties):**
- `model.W_E`: Token embeddings `[d_vocab, d_model]`
- `model.W_pos`: Positional embeddings `[n_ctx, d_model]` (only for absolute pos embeds)
- `model.W_U`: Unembedding matrix `[d_model, d_vocab]`
- `model.b_U`: Unembedding bias `[d_vocab]`
- `model.W_Q`: Query weights `[n_layers, n_heads, d_model, d_head]`
- `model.W_K`: Key weights `[n_layers, n_heads, d_model, d_head]`
- `model.W_V`: Value weights `[n_layers, n_heads, d_model, d_head]`
- `model.W_O`: Attention output weights `[n_layers, n_heads, d_head, d_model]`
- `model.b_Q`: Query bias `[n_layers, n_heads, d_head]`
- `model.b_K`: Key bias `[n_layers, n_heads, d_head]`
- `model.b_V`: Value bias `[n_layers, n_heads, d_head]`
- `model.b_O`: Output bias `[n_layers, d_model]`
- `model.W_in`: MLP input weights `[n_layers, d_model, d_mlp]`
- `model.W_out`: MLP output weights `[n_layers, d_mlp, d_model]`
- `model.b_in`: MLP input bias `[n_layers, d_mlp]`
- `model.b_out`: MLP output bias `[n_layers, d_model]`
- `model.W_gate`: MLP gate weights `[n_layers, d_model, d_mlp]` (only for gated MLPs)
- `model.b_gate`: MLP gate bias `[n_layers, d_mlp]` (only for gated MLPs)

**Variants:**

---

## Integrated Workflow Example

The script `./tests/TransformerLens/integrated_pipeline_demo.py` demonstrates combining several functionalities such as loading, caching, and patching into a practical workflow. Refer to this script for an example of integrating these components.

---

## Gradient Computation (Autograd)

TransformerLens models are standard `torch.nn.Module` instances, allowing you to leverage PyTorch's automatic differentiation (`torch.autograd`) to compute gradients between different parts of the computation graph (inputs, activations, outputs, loss).

**Important:** Gradient computation requires running the model *without* `torch.no_grad()` and ensuring the tensors you want gradients *for* require gradients (`requires_grad=True`). Furthermore, if you are computing gradients *with respect to* an intermediate activation captured by a hook (which is typically a non-leaf tensor in the computation graph), you **must** explicitly call `.retain_grad()` on that activation tensor within the hook function. Otherwise, PyTorch will not populate its `.grad` attribute during the backward pass.

### Gradient of Output Probability w.r.t. Activation

**Description:**
Calculate the gradient of a specific output probability (e.g., the probability of the correct next token) with respect to the activation of an intermediate layer (e.g., MLP output). This shows how sensitive the output probability is to changes in that activation.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.
- Input tokens.
- The name of the target intermediate hook point.
- The index of the target output token and target sequence position.

**Expected Outcome:**
- A gradient tensor is computed with the same shape as the intermediate activation tensor.
- This tensor represents `d(target_prob) / d(activation)`.

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_gradient_output_wrt_activation.py`_
```python
import torch
import transformer_lens
from transformer_lens import utils

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "The capital of France is"
TARGET_TOKEN = " Paris"
TARGET_POS = -1 # Gradient for the prediction at the last position
GRAD_LAYER = 6
GRAD_HOOK_NAME = utils.get_act_name("mlp_out", GRAD_LAYER) # Grad w.r.t Layer 6 MLP output

# Load Model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval() # Keep eval mode for consistency, but allow gradients
input_ids = model.to_tokens(PROMPT)
target_token_id = model.to_single_token(TARGET_TOKEN)

# --- Prepare for Gradient Calculation ---
# Store the activation tensor we want the gradient OF
stored_activation_for_grad = None
def store_activation_hook_for_grad(activation, hook):
    global stored_activation_for_grad
    # CRITICAL: Retain grad for this intermediate (non-leaf) tensor
    activation.retain_grad()
    stored_activation_for_grad = activation
    return activation

# --- Run Forward Pass with Gradients Enabled and Hook ---
print(f"Running forward pass with grad enabled, hooking {GRAD_HOOK_NAME}...")

with torch.enable_grad(): # Ensure gradients are computed
    # Apply the hook to store the activation tensor we need
    logits = model.run_with_hooks(
        input_ids,
        fwd_hooks=[(GRAD_HOOK_NAME, store_activation_hook_for_grad)],
    )

    # Calculate the specific output probability we care about
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_prob = log_probs[0, TARGET_POS, target_token_id]
    target_prob = torch.exp(target_log_prob)

print(f"Target Log Prob ('{TARGET_TOKEN}' at pos {TARGET_POS}): {target_log_prob.item():.4f}")
print(f"Target Prob ('{TARGET_TOKEN}' at pos {TARGET_POS}): {target_prob.item():.4f}")

# Check if the stored activation tensor exists and retains grad
assert stored_activation_for_grad is not None, "Hook did not store activation"
# Note: requires_grad might be True automatically, but retain_grad is crucial
assert stored_activation_for_grad.requires_grad, "Stored activation does not require grad. Check enable_grad context."
assert stored_activation_for_grad.retains_grad, "Stored activation does not retain grad. Missing retain_grad() call?"

# --- Calculate Gradient ---
print(f"Calculating d(Prob) / d({GRAD_HOOK_NAME})...")
# Compute the gradient of the target probability w.r.t the stored activation
# Clear previous gradients if any (good practice)
if stored_activation_for_grad.grad is not None:
    stored_activation_for_grad.grad.zero_()
    
target_prob.backward() # Backpropagate from the target probability

# The gradient is now stored in stored_activation_for_grad.grad
grad_tensor = stored_activation_for_grad.grad

assert grad_tensor is not None, "Gradient calculation failed"
assert grad_tensor.shape == stored_activation_for_grad.shape
print(f"Gradient tensor computed. Shape: {grad_tensor.shape}")
print(f"Gradient tensor mean abs: {torch.abs(grad_tensor).mean().item():.6f}")
# Basic check: gradient should not be zero everywhere if target prob > 0
assert torch.abs(grad_tensor).mean().item() > 1e-9
```

**Implementation Tips:**
- Run the relevant part of your code inside `with torch.enable_grad():` or ensure `torch.no_grad()` is not active.
- Use `model.run_with_hooks` to capture the intermediate activation tensor needed.
- **Crucially, call `.retain_grad()` on the activation tensor *inside the hook function* if you want to compute gradients *with respect to* that activation.** Activations captured by hooks are typically non-leaf tensors, and PyTorch's autograd engine won't store their gradients by default.
- Ensure the scalar value you differentiate (`target_prob` here) depends computationally on the hooked activation.
- Calculate the scalar value (like a specific probability or loss) you want to differentiate.
- Call `.backward()` on the scalar output value. The gradient will be computed and stored in the `.grad` attribute of the intermediate activation tensor (provided `.retain_grad()` was called).
- Zero out `.grad` on the tensor before calling `backward()` if you might perform multiple backward passes involving the same tensor to avoid accumulating gradients.

**Variants:**
- Calculate gradient of loss w.r.t. activation.
- Calculate gradient w.r.t. activations at different hook points.

**Troubleshooting:**
- **`AttributeError: 'NoneType' object has no attribute 'grad'` or Gradient is `None`:** The most likely cause is **forgetting to call `.retain_grad()`** on the intermediate activation inside the hook. Also check that the intermediate tensor was actually part of the computation graph leading to the output scalar and that `torch.enable_grad()` was active.
- **Gradient is all zeros:** This might happen if the output scalar is independent of the intermediate activation, or if the output value itself was zero.
- **RuntimeError: element 0 of tensors does not require grad...**: Ensure the tensor you call `.backward()` on depends on the tensor you want the gradient *with respect to*, and that `enable_grad` was active.

### Gradient of Activation w.r.t. Input Embedding

**Description:**
Calculate the gradient of an intermediate activation (e.g., residual stream) with respect to the input embeddings (output of the embedding layer). This shows how sensitive an internal state is to changes in the initial token/positional representations.

**Prerequisites:**
- A loaded `transformer_lens.HookedTransformer` object.
- Input tokens.
- The name of the target intermediate hook point.

**Expected Outcome:**
- A gradient tensor is computed with the same shape as the input embedding tensor (`[batch, pos, d_model]`).
- This tensor represents `d(activation.sum()) / d(input_embedding)` (where the sum is used to get a scalar for backpropagation).

**Tested Code Implementation:**
_Verified via `./tests/TransformerLens/test_gradient_activation_wrt_input.py`_
```python
import torch
import transformer_lens
from transformer_lens import utils

# --- Configuration ---
MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Gradient w.r.t. inputs"
TARGET_LAYER = 4
TARGET_HOOK_NAME = utils.get_act_name("resid_post", TARGET_LAYER) # Grad OF Layer 4 resid_post
INPUT_HOOK_NAME = "hook_embed" # Grad W.R.T output of token embedding

# Load Model
model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()
input_ids = model.to_tokens(PROMPT)

# --- Prepare for Gradient Calculation ---
# Store input embedding and target activation
stored_target_activation = None
stored_input_embed = None

def store_target_hook(activation, hook):
    global stored_target_activation
    # No retain_grad needed here if we only differentiate *through* it
    stored_target_activation = activation
    return activation

def store_input_hook(activation, hook):
    global stored_input_embed
    # Ensure the input embed requires grad
    activation.requires_grad_(True)
    # CRITICAL: Retain grad as we want the gradient *stored* on this tensor
    activation.retain_grad()
    stored_input_embed = activation
    return activation

# --- Run Forward Pass with Gradients Enabled and Hooks ---
print(f"Running forward pass, hooking {TARGET_HOOK_NAME} and {INPUT_HOOK_NAME}...")
with torch.enable_grad():
    # Need to hook *both* the input and the target activation points
    logits = model.run_with_hooks(
        input_ids,
        fwd_hooks=[(INPUT_HOOK_NAME, store_input_hook),
                   (TARGET_HOOK_NAME, store_target_hook)]
    )

    assert stored_input_embed is not None, "Input hook failed"
    assert stored_target_activation is not None, "Target hook failed"
    assert stored_input_embed.requires_grad, "Input embed does not require grad"
    assert stored_input_embed.retains_grad, "Input embed does not retain grad"

    # To calculate the gradient d(activation)/d(input), we need a scalar value
    # derived from the activation to call .backward() on.
    # Summing the target activation is standard.
    target_scalar = stored_target_activation.sum()
    print(f"Target activation sum: {target_scalar.item():.4f}")

    # --- Calculate Gradient ---
    print(f"Calculating d({TARGET_HOOK_NAME}.sum()) / d({INPUT_HOOK_NAME})...")
    # Clear any previous gradients on the input embed
    if stored_input_embed.grad is not None: 
        stored_input_embed.grad.zero_()
    
    # Calculate gradients
    target_scalar.backward()
    
    # Gradient is now in stored_input_embed.grad
    grad_tensor = stored_input_embed.grad

assert grad_tensor is not None, "Gradient calculation failed"
assert grad_tensor.shape == stored_input_embed.shape
print(f"Gradient tensor computed. Shape: {grad_tensor.shape}")
print(f"Gradient tensor mean abs: {torch.abs(grad_tensor).mean().item():.6f}")
# Basic check: gradient should not be zero everywhere
assert torch.abs(grad_tensor).mean().item() > 1e-9
```

**Implementation Tips:**
- You need hooks at *both* the point you want the gradient *of* (target activation) and the point you want the gradient *with respect to* (input embedding).
- The tensor you want gradients *with respect to* (the input embeddings in this case) **must** have `requires_grad=True` AND **must** call `.retain_grad()`. Setting `requires_grad_(True)` makes it possible to compute gradients *through* it, but `.retain_grad()` ensures the computed gradient value is actually *stored* in its `.grad` attribute.
- You need to derive a scalar value from the tensor you want the gradient *of* (the intermediate activation) to call `.backward()` on. Summing the tensor is a standard way to do this if you want the overall sensitivity.
- The computed gradient will be stored in the `.grad` attribute of the tensor designated as the input (the one where `.retain_grad()` was called).

**Variants:**
- Calculate gradient w.r.t. positional embeddings (`hook_pos_embed`).
- Calculate gradient w.r.t specific dimensions or positions by appropriate indexing before summing the target activation.

**Troubleshooting:**
- **Gradient is `None`:** The most likely cause is **forgetting to call `.retain_grad()`** on the tensor you want the gradient *with respect to* (e.g., `stored_input_embed` here). Also ensure `requires_grad=True` was set on it, and that the target scalar depends computationally on this input tensor within the autograd graph under `torch.enable_grad()`.
- **`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`:** This usually means `requires_grad=True` was not set correctly on the input tensor *before* it was used in the computation leading to the target activation.