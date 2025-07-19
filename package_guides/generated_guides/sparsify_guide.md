# Sparsify Package Examples and Usage

This document provides a comprehensive guide to using the Sparsify package, covering key functionalities for training and using sparse autoencoders (SAEs) with Hugging Face language models.

## Table of Contents
1. [Loading Pre-trained SAEs](#loading-pre-trained-saes)
2. [Data Preparation](#data-preparation)
3. [Training SAEs/Transcoders](#training-saestranscoders)
4. [Finetuning SAEs](#finetuning-saes)
5. [Using Custom Hookpoints](#using-custom-hookpoints)
6. [Distributed Training](#distributed-training)
7. [Encoding Data with SAEs](#encoding-data-with-saes)
8. [Decoding Latents with SAEs](#decoding-latents-with-saes)
9. [Conceptual Full Pipeline](#conceptual-full-pipeline)

---

## Loading Pre-trained SAEs

### Load a Single Pretrained SAE

**Description:**
Load a specific pretrained SAE from a given repository on the Hugging Face Hub for a specified model layer (hookpoint).

**Prerequisites:**
- A valid Hugging Face Hub repository ID containing pretrained SAEs (e.g., `"EleutherAI/sae-Llama-3.2-1B-131k"`).
- A valid hookpoint string corresponding to a layer whose SAE is available in the repository (e.g., `"layers.10.mlp"`).

**Expected Outcome:**
- An `Sae` object is successfully loaded.
- The loaded `Sae` object contains valid weight tensors (`encoder.weight`, `encoder.bias`, `W_dec`, `b_dec`).

**Tested Code Implementation:**
_While `Sae.load_from_hub` exists, `Sae.load_many` (documented below) is the recommended and more robust approach, even for loading a single SAE, as it correctly handles various repository structures. Use the `layers` argument to specify the single hookpoint needed._

```python
# Example using load_many to get a single SAE (Verified via tests/sparsify/test_load_single_sae.py)
from sparsify import Sae
import torch # For device selection

REPO_ID = "EleutherAI/sae-Llama-3.2-1B-131k"
HООКРОINT_TO_LOAD = "layers.10.mlp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Use load_many and specify the target layer in the 'layers' list
    sae_dict = Sae.load_many(REPO_ID, layers=[HООКРОINT_TO_LOAD], device=DEVICE)

    if HООКРОINT_TO_LOAD in sae_dict:
        single_sae = sae_dict[HООКРОINT_TO_LOAD]
        print(f"Successfully loaded SAE for {HООКРОINT_TO_LOAD} onto {single_sae.device}.")

        # Access weights using correct attributes:
        # print("Encoder Weight Shape:", single_sae.encoder.weight.shape)
        # print("Decoder Weight Shape:", single_sae.W_dec.shape)
    else:
        print(f"Error: Hookpoint {HООКРОINT_TO_LOAD} not found in loaded dictionary.")

except Exception as e:
    print(f"Error loading SAE: {e}")

# Note: The `load_from_hub` method is less flexible regarding repository structure 
# and does not currently support loading from gated models via token. 
# `load_many` is preferred for consistency and broader compatibility.

**Implementation Tips:**
- Always use `Sae.load_many` with the `layers` argument specifying the single hookpoint you need. This aligns with the method for loading multiple SAEs and handles different repository structures more reliably.
- Ensure the `REPO_ID` is correct and the specified `HООКРОINT_TO_LOAD` exists as a subdirectory within that repository, containing `cfg.json` and `sae.safetensors`.

**Variants:**
- Change `REPO_ID`, `HООКРОINT_TO_LOAD`, and `DEVICE`.

**Troubleshooting:**
- **`RepositoryNotFoundError` / `FileNotFoundError`:** Double-check `REPO_ID`. Ensure the repository exists and is structured with hookpoint subdirectories.
- **`KeyError` (in `sae_dict`)**: Ensure the specified `HООКРОINT_TO_LOAD` exists as a subdirectory in the repository.
- **Attribute Errors:** Use the correct attribute paths (`.encoder.weight`, `.W_dec`, etc.).
- **Network Issues:** Check internet connection when loading from Hub.

### Load Multiple Pretrained SAEs

**Description:**
Load all or a subset of available pretrained SAEs from a given repository on the Hugging Face Hub or a local directory using `Sae.load_many`.

**Prerequisites:**
- A valid Hugging Face Hub repository ID or local directory path containing pretrained SAEs.
- **Directory Structure:** SAEs must be stored in subdirectories named after their corresponding hookpoints (e.g., `repo_root/layers.10.mlp/`). Each subdirectory must contain `sae.safetensors` and `cfg.json` files. This matches the output structure created by the `Trainer` *within* its run-specific directory (e.g., `{save_dir}/{run_name or "unnamed"}/`).
- Optional: A list of specific hookpoint strings to load via the `layers` argument.

**Expected Outcome:**
- A dictionary is returned where keys are hookpoint strings and values are the corresponding loaded, functional `Sae` objects.
- The dictionary keys (hookpoints) are naturally sorted.
- Loaded `Sae` objects contain valid weight tensors (`encoder.weight`, `encoder.bias`, `W_dec`, `b_dec`).

**Tested Code Implementation:**
_Verified via `./tests/sparsify/test_loading.py`. This example loads all MLP SAEs from the specified repository._
```python
import torch
from sparsify import Sae
from natsort import natsorted

# --- Configuration ---
REPO_ID = 'EleutherAI/sae-Llama-3.2-1B-131k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SAEs from Hub repo: {REPO_ID}...")

saes = {}
try:
    # Load all SAEs found in the repository structure by omitting 'layers'
    # Or specify a list of hookpoints via the 'layers' argument
    saes = Sae.load_many(REPO_ID, device=DEVICE)
    
    print(f"Successfully loaded {len(saes)} SAEs.")
    loaded_hookpoints = natsorted(list(saes.keys()))
    print(f"Loaded hookpoints: {loaded_hookpoints}")

    # --- Verification and Usage Example ---
    if saes:
        first_hookpoint = loaded_hookpoints[0]
        first_sae = saes[first_hookpoint]
        
        print(f"\nVerifying first loaded SAE ({first_hookpoint})...")
        # Access attributes using correct paths
        assert hasattr(first_sae, 'encoder') and hasattr(first_sae.encoder, 'weight')
        assert hasattr(first_sae, 'W_dec')
        print(f"  Encoder Weight Shape: {first_sae.encoder.weight.shape}")
        print(f"  Decoder Weight Shape: {first_sae.W_dec.shape}")
        print(f"  SAE is on device: {first_sae.device}") # Use sae.device
        print("  Verification successful.")
    else:
        print("No SAEs were loaded.")

except Exception as e:
    print(f"An error occurred during loading: {e}")

# Example: Loading locally trained SAEs from the programmatic example:
# LOCAL_SAE_PATH = "./sparsify_training_out/unnamed/" # Path to directory containing hookpoint subdirs
# saes_local = Sae.load_many(LOCAL_SAE_PATH, local=True, device=DEVICE)
# print(f"Loaded {len(saes_local)} SAEs locally from {LOCAL_SAE_PATH}")

**Implementation Tips:**
- `Sae.load_many` is the standard method for loading single or multiple SAEs from Hub or local paths.
- **Required Repo/Directory Structure:** This function expects SAEs to be in subdirectories named after their hookpoint (e.g., `repo_root/layers.10.mlp/`). The path provided (Hub ID or local path) must point to the directory *containing* these hookpoint subdirectories (e.g., `{save_dir}/{run_name or "unnamed"}/` for Trainer output).
- **Incompatible Structures:** Loading from repositories/directories with different structures will likely fail. Custom loading logic would be needed.
- **Gated Models:** This function currently does not support passing an authentication token for loading from gated Hugging Face repositories.
- Omit the `layers` argument to load all found SAEs; provide a list of hookpoint strings to load a specific subset.
- Specify `local=True` when loading from a local directory path.
- Access encoder weights/bias via `.encoder.weight` and `.encoder.bias`.
- Access decoder weights/bias via `.W_dec` and `.b_dec`.

**Variants:**
- Load from different Hub repositories or local paths.
- Load only specific layers using the `layers` argument.
- Load to CPU by setting `device="cpu"`.

**Troubleshooting:**
- **`RepositoryNotFoundError` / `FileNotFoundError`:** Double-check the path (Hub ID or local path). Ensure the repository/directory exists and strictly follows the required structure (hookpoint subdirectories, each with `cfg.json` and `sae.safetensors`).
- **Attribute Errors:** Ensure you are using the correct attribute paths for accessing weights (`.encoder.weight`, `.W_dec`, etc.).
- **Network Issues:** Check internet connection when loading from Hub.
- **Memory Issues:** Loading many large SAEs can consume significant memory. Consider loading only necessary SAEs using the `layers` argument or loading them sequentially.

---

## Data Preparation

### Chunk and Tokenize a Dataset

**Description:**
Process a raw text dataset from Hugging Face `datasets` by chunking documents into sequences of a specified length and tokenizing them using a provided Hugging Face `transformers` tokenizer.

**Prerequisites:**
- A Hugging Face `Dataset` object (e.g., loaded via `load_dataset`).
- A Hugging Face `PreTrainedTokenizer` or `PreTrainedTokenizerFast` object.
- Optional configuration: `max_seq_len` (default 2048), `num_proc` (for parallel processing), `text_key` (dataset column containing text, default "text").

**Expected Outcome:**
- A new Hugging Face `Dataset` object is returned.
- The returned dataset contains tokenized sequences under the key "input_ids".
- Each sequence in "input_ids" has a length equal to `max_seq_len`.
- The number of examples in the returned dataset will likely differ from the input dataset due to chunking.

**Tested Code Implementation:**
_Verified via `./tests/sparsify/test_chunk_tokenize.py`._
```python
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sparsify.data import chunk_and_tokenize

# --- Example Configuration ---
DATASET_ID = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"
TOKENIZER_ID = "gpt2"
MAX_SEQ_LEN = 1024 # Example sequence length
TEXT_KEY = "text" # Column containing text in wikitext

# --- Load Raw Dataset ---
print("Loading raw dataset...")
# Use streaming=True for large datasets to avoid downloading everything
# For testing, load a small slice: split="train[:1%]"
raw_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=False) 
print("Dataset loaded.")

# --- Load Tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Set padding token if needed
print("Tokenizer loaded.")

# --- Chunk and Tokenize ---
print(f"Chunking and tokenizing with max_seq_len={MAX_SEQ_LEN}...")
tokenized_dataset = chunk_and_tokenize(
    raw_dataset,
    tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    text_key=TEXT_KEY,
    # num_proc=os.cpu_count() # Use multiple processes for speed (optional)
)
print(f"Finished processing. Resulting dataset has columns: {tokenized_dataset.column_names}")

# --- Verification (Example) ---
# The resulting tokenized_dataset is ready for use.
# You can access sequences like this:
if len(tokenized_dataset) > 0:
    first_sequence = tokenized_dataset[0]["input_ids"]
    print(f"Length of first tokenized sequence: {len(first_sequence)}")
    # Expected length is MAX_SEQ_LEN
else:
    print("Resulting dataset is empty after processing.")

```

**Implementation Tips:**
- **Tokenizer Consistency is Crucial:** The `tokenizer` used here **must** be the same instance (or loaded from the same identifier) as the one used with the base language model in subsequent steps (like Training or Encoding/Decoding). Mismatched tokenizers will lead to errors or incorrect results.
- Use `streaming=True` in `load_dataset` for very large datasets to avoid downloading the entire dataset upfront.
- Uncomment `num_proc=os.cpu_count()` (after importing `os`) in `chunk_and_tokenize` to leverage multiple CPU cores and significantly speed up processing, especially for large datasets. Ensure you have sufficient RAM.
- Ensure the `text_key` argument matches the column name containing the primary text content in your specific dataset.
- Ensure the tokenizer has a padding token set (`tokenizer.pad_token`). If it's `None`, set it, often to the EOS token (`tokenizer.pad_token = tokenizer.eos_token`), as shown in the example.

**Variants:**
- Adjust `max_seq_len` to match the context window or desired input length for the target language model and SAEs.
- Process different dataset splits (e.g., "train", "validation", "test").
- Use different tokenizers (ensuring consistency with the base model).

**Troubleshooting:**
- **`KeyError`:** Verify the `text_key` argument matches a valid column name in the input dataset.
- **Tokenization Issues:** Confirm the tokenizer is loaded correctly, has a `pad_token`, and is suitable for the language/content of the dataset.
- **Memory Errors:** If using `num_proc > 1`, reduce its value or process the dataset in smaller chunks if memory issues arise during parallel processing.

---

## Training SAEs/Transcoders

### Programmatic Training Setup

**Description:**
Set up the necessary components for training SAEs (or transcoders) programmatically using the `Trainer` class. This involves loading the base language model, preparing the tokenized dataset, configuring SAE and training parameters, and initializing the `Trainer`.

**Prerequisites:**
- A Hugging Face `transformers` model identifier (e.g., `"EleutherAI/pythia-160m-deduped"`).
- A tokenized Hugging Face `Dataset` (output of `chunk_and_tokenize` or similar), containing "input_ids" and preferably formatted as PyTorch tensors (`.set_format(type='torch')`).
- **Crucially:** A tokenizer loaded from the **same identifier** as the one used to create the `tokenized_dataset`.
- The exact hookpoint name(s) within the chosen model architecture to train on (e.g., `"gpt_neox.layers.5.mlp"`). Inspect the model structure (`print(model)`) to find correct names.
- Desired SAE configuration parameters (passed to `SaeConfig`).
- Desired training configuration parameters (passed to `TrainConfig`).

**Expected Outcome:**
- A `Trainer` object is successfully initialized without errors.
- The `Trainer` object holds references to the language model, dataset, and configurations.
- The `Trainer` correctly identifies the input dimension for the specified hookpoint(s) and initializes SAEs accordingly.

**Tested Code Implementation:**
_Based on `./tests/sparsify/test_full_training_pipeline.py`_
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
DATASET_SPLIT = "train[:1%]" # Use a small slice for example
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Target Hookpoint --- 
# Important: Find the correct name by inspecting the specific model architecture
# Example for Pythia-160m MLP output in layer 5
HOOKPOINT = "gpt_neox.layers.5.mlp"

# --- Load Base Model & Tokenizer ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset,
    tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    # num_proc=os.cpu_count() # Optional: for speed
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")

# --- Define Configurations ---
print("Defining SAE and Training configurations...")
# Configure the SAEs to be trained
sae_cfg = SaeConfig(
    expansion_factor=4, # Ratio of SAE latent dim to model dim (keep small for demo)
    k=32,               # Number of non-zero latents (sparsity)
)

# Configure the training process
train_cfg = TrainConfig(
    sae_cfg,            
    batch_size=8,       # Adjust based on GPU memory
    lr=1e-4,            
    save_every=1000,    
    save_dir="./sparsify_training_out", 
    hookpoints=[HOOKPOINT], # *** Crucial: Specify the target hookpoint(s) ***
    log_to_wandb=False
)
print(f"Configurations defined for hookpoint: {train_cfg.hookpoints}")

# --- Initialize Trainer ---
print("Initializing Trainer...")
# The Trainer automatically resolves input dimensions and initializes SAEs
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized successfully.")
print(f"Training SAE(s) for hookpoints: {list(trainer.saes.keys())}")

# --- Ready for Training ---
# The trainer object is now ready.
# Next step is to call trainer.fit() to start the actual training loop.
print("Setup complete. Call trainer.fit() to begin training.")

```

**Implementation Tips:**
- **Find Correct Hookpoints:** **This is the most common failure point.** Load your specific `model` and use `print(model)` or iterate through `model.named_modules()` to find the exact string name of the layer/module output you want to capture activations from. Pass this exact string(s) to `TrainConfig(hookpoints=[...])`.
- **Tokenizer Consistency:** Ensure the `tokenizer` loaded alongside the `model` matches the one used to create the `tokenized_dataset` passed to the `Trainer`. Mismatches will cause errors.
- **Dataset Formatting:** Ensure the `tokenized_dataset` is correctly formatted, typically by calling `.set_format(type='torch')` after tokenization, so it yields PyTorch tensors as expected by the `Trainer`.
- Choose a base model appropriate for your hardware resources.
- Configure `SaeConfig` and `TrainConfig` carefully, especially `hookpoints`.
- Place the base model on the appropriate device (`.to(DEVICE)`) before initializing the `Trainer`.

**Variants:**
- Use different base models (e.g., `gpt2`, Llama models if accessible).
- Train on different dataset splits.
- Adjust parameters in `SaeConfig` and `TrainConfig`.
- Specify multiple hookpoints or use wildcard patterns (e.g., `hookpoints=["gpt_neox.layers.*.mlp.act"]`) in `TrainConfig`.

**Troubleshooting:**
- **Model Loading Errors:** Ensure model identifier is correct and dependencies (`transformers`) are installed. Check hardware requirements.
- **Dataset Issues:** Verify the dataset is correctly tokenized and accessible. Ensure it's formatted correctly (e.g., `set_format(type='torch')`). Mismatched tokenizers between data prep and trainer initialization can also cause issues here.
- **Configuration Errors:** Double-check parameter names and values in `SaeConfig` and `TrainConfig`.
- **Hookpoint Errors:** If `Trainer` initialization fails with dimension errors or the optimizer fails with `empty parameter list`, the most likely cause is an incorrect hookpoint name provided in `TrainConfig.hookpoints`. Verify the name against `model.named_modules()`.
- **CUDA/Device Errors:** Ensure PyTorch is installed with CUDA support if using GPU. Verify the model and dataset tensors are on the correct device.

---

### Execute Training (`trainer.fit()`)

**Description:**
Runs the main training loop after the `Trainer` object has been initialized. This involves iterating through the dataset, computing activations from the specified hookpoint(s), calculating SAE losses, performing backpropagation, and updating SAE weights.

**Prerequisites:**
- A successfully initialized `Trainer` object (see "Programmatic Training Setup").
- Sufficient compute resources (CPU/GPU memory and processing power).

**Expected Outcome:**
- The training loop runs for the configured duration (determined by dataset size and batch size).
- Progress is logged to the console (and optionally Weights & Biases).
- Checkpoints (SAE weights `sae.safetensors` and config `cfg.json`) are saved periodically. They are organized within the `TrainConfig.save_dir` under a subdirectory named by `TrainConfig.run_name` (or `"unnamed"` if not set), and further subdivided by hookpoint name. The structure is: `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The `fit()` method completes without runtime errors.

**Tested Code Implementation:**
_This example uses the setup from the previous step (`test_full_training_pipeline.py`) and runs `fit()`._

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import shutil
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration (Match setup from previous step) ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
# Use enough data for a few steps
DATASET_SPLIT = "train[:120]" # ~15 steps with batch_size=8
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HООКРОINT = "gpt_neox.layers.5.mlp"
SAVE_DIR = "./sparsify_training_out"
SAVE_EVERY = 10 # Save frequently for demo
TARGET_BATCH_SIZE = 8

# --- Load Model & Tokenizer (As in setup) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset (As in setup) ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset, tokenizer, max_seq_len=MAX_SEQ_LEN
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")
num_steps = len(tokenized_dataset) // TARGET_BATCH_SIZE
print(f"Expected number of training steps: {num_steps}")

# --- Define Configurations (As in setup) ---
print("Defining SAE and Training configurations...")
sae_cfg = SaeConfig(expansion_factor=4, k=32)
train_cfg = TrainConfig(
    sae_cfg, 
    batch_size=TARGET_BATCH_SIZE,
    lr=1e-4, 
    save_every=SAVE_EVERY, 
    save_dir=SAVE_DIR,
    hookpoints=[HOOKPOINT], 
    log_to_wandb=False 
)
print("Configurations defined.")

# --- Initialize Trainer (As in setup) ---
print("Initializing Trainer...")
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized.")

# --- Execute Fit ---
try:
    print("Calling trainer.fit()...")
    # Training runs until dataset is exhausted
    trainer.fit()
    print("trainer.fit() completed.")

    # --- Verification ---
    print(f"Verifying checkpoint in {SAVE_DIR}...")
    # Check for base save directory
    assert os.path.isdir(SAVE_DIR), "Save directory was not created."
    # Account for potential 'unnamed' subdirectory if run_name not set
    base_save_path = os.path.join(SAVE_DIR, train_cfg.run_name or "unnamed")
    assert os.path.isdir(base_save_path), f"Base save path '{base_save_path}' not found."
    # Check for hookpoint subdirectory
    hookpoint_save_dir = os.path.join(base_save_path, HOOKPOINT)
    assert os.path.isdir(hookpoint_save_dir), f"Hookpoint subdirectory '{HOOKPOINT}' not found."
    # Check for specific files
    cfg_path = os.path.join(hookpoint_save_dir, "cfg.json")
    sae_path = os.path.join(hookpoint_save_dir, "sae.safetensors")
    assert os.path.isfile(cfg_path), "cfg.json not found."
    assert os.path.isfile(sae_path), "sae.safetensors not found."
    print("Checkpoint verification passed.")

except Exception as e:
    print(f"Error during trainer.fit(): {e}")
finally:
    # Optional: Clean up output directory after verification/use
    if os.path.exists(SAVE_DIR):
        print(f"Cleaning up directory: {SAVE_DIR}")
        shutil.rmtree(SAVE_DIR)
```

**Implementation Tips:**
- **Correct Setup is Key:** The success of `trainer.fit()` relies heavily on the correct initialization of the `Trainer` object in the preceding setup step. Ensure the base model, tokenized dataset (with matching tokenizer and correct formatting), and configurations (especially `hookpoints`) are all correctly specified.
- Ensure the `Trainer` is correctly initialized with the model, dataset, and configurations before calling `fit()`.
- Monitor resource usage (GPU memory, CPU).
- Adjust `batch_size` and `grad_acc_steps` to manage memory.
- Training duration depends on dataset size and batch size.
- Set `log_to_wandb=True` for detailed metric tracking.

**Troubleshooting:**
- **Setup Errors:** Failures during `fit()` (especially early on or related to tensor shapes/devices) can often stem from issues in the `Trainer` initialization. Double-check hookpoint names, tokenizer consistency, dataset formatting (`set_format`), and device placement from the setup step.
- **CUDA Out of Memory:** Reduce `batch_size`, use gradient accumulation (`grad_acc_steps`), use a smaller model, or use model parallelism/distribution features (see "Distributed Training").
- **Slow Training:** Increase `batch_size` (if memory allows), use GPU, use `num_proc` during data loading/tokenization, ensure efficient data loading.
- **NaN/Inf Losses:** Check learning rate (`lr`), potentially reduce it. Ensure input data is normalized correctly if needed. Check for numerical stability issues in the base model or SAE configuration.
- **Checkpointing Issues:** Verify write permissions for the `save_dir`.

---

### Command-Line Interface (CLI) Training

**Description:**
Train SAEs using the built-in command-line interface provided by `sparsify`. This offers a convenient way to run training jobs by specifying the model, dataset, and configuration options as arguments, leveraging the same underlying `Trainer` logic.

**Prerequisites:**
- The `sparsify` package installed in the environment.
- A base Hugging Face model identifier.
- Optional: A dataset identifier (defaults to `EleutherAI/fineweb-edu-dedup-10b` if omitted).
- Optional: Access to required compute resources (GPU recommended).

**Expected Outcome:**
- The training process starts, logging progress to the console.
- Checkpoints are saved to subdirectories within the default directory (`./checkpoints`) or a specified output directory (`--save_dir`), following the structure `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The command runs to completion or until manually stopped.

**Tested Code Implementation:**
_The following command was verified to execute successfully, running for a single step using a small model and dataset. Replace parameters as needed for actual training runs._

```bash
# Activate your conda environment first
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 1 \
    --expansion_factor 1 \
    --k 16 \
    --log_to_wandb False \
    --save_dir ./test_cli_run_out \
    --max_examples 1

# Expected output: Command runs, prints progress, saves checkpoint, and exits.
# Remember to clean up ./test_cli_run_out afterwards if desired.
```

**Implementation Tips:**
- Use `python -m sparsify --help` to see all available CLI options. Most arguments directly map to fields in the programmatic `SaeConfig` and `TrainConfig` (e.g., `--batch_size` maps to `TrainConfig.batch_size`, `--expansion_factor` maps to `SaeConfig.expansion_factor`).
- Specify arguments using `--arg_name value` format (e.g., `--batch_size 16`, `--expansion_factor 32`).
- Use boolean flags like `--load_in_8bit`. To explicitly set a boolean flag to `False` when its default is `True`, use the `--no-<flag_name>` syntax if available (check `--help`), otherwise omit the flag.
- For multi-GPU training, prepend the command with `torchrun --nproc_per_node <num_gpus>` (see "Distributed Training").
- By default, logs are sent to Weights & Biases if available; use `--no-log_to_wandb` to disable.

**Variants:**
- Specify a different dataset (Hub ID or local path).
- Override default `SaeConfig` parameters (e.g., `--k 128`, `--expansion_factor 64`).
- Override default `TrainConfig` parameters (e.g., `--lr 1e-4`, `--save_dir ./my_sae_run`).
- Train on specific layers (`--layers 0 5 10`) or using a stride (`--layer_stride 4`).
- Use custom hookpoints (`--hookpoints "h.*.attn" "h.*.mlp.act"`).

**Troubleshooting:**
- **Argument Errors:** Check `python -m sparsify --help` for correct argument names and types. Ensure boolean flags use the correct syntax (`--flag_name` or `--no-flag_name`).
- **Incompatible Arguments:** Some argument combinations are disallowed (e.g., using both `--layers` and `--hookpoints`). The CLI should raise an error if incompatible arguments are provided.
- **Model/Dataset Loading Errors:** Verify model and dataset identifiers are correct and accessible.
- **Resource Errors (OOM):** Reduce `--batch_size`, use `--grad_acc_steps N`, use `--load_in_8bit`, or use distributed training options if applicable.
- **`torchrun` Issues:** Ensure `torchrun` is available and configured correctly for your environment if doing multi-GPU training.

---

## Finetuning SAEs

**Description:**
Continue training SAEs from a previously saved checkpoint. This is useful for resuming interrupted training runs or further refining SAEs with different data or learning rates.

**Prerequisites:**
- A valid checkpoint directory path. This path should point to the directory containing the hookpoint subdirectories (e.g., the `{save_dir}/{run_name or "unnamed"}/` directory from a previous run). Each relevant hookpoint subdirectory must contain:
    - `sae.safetensors` (SAE weights)
    - `cfg.json` (SAE config)
    - `optimizer_*.pt` (Optimizer state for the SAE)
    - `scheduler_*.pt` (Scheduler state, if used)
    - `scaler_*.pt` (Gradient scaler state, if used)
- The **same base model** (`MODEL_ID`) used for the initial training.
- A **tokenized dataset** prepared using the **same tokenizer and formatting** as the initial training run.

**Expected Outcome:**
- The `Trainer` initializes, loading the SAE weights and optimizer state from the specified checkpoint directory.
- Training resumes from the step count saved in the checkpoint.
- The finetuning process runs, potentially saving new checkpoints to a different directory.

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_finetune.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset
import os

from sparsify import SaeConfig, Trainer, TrainConfig

# --- Configuration (Assume initial checkpoint exists) ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./test_finetune_checkpoint_source" # Path to the existing checkpoint
FINETUNE_SAVE_DIR = "./test_finetune_run_out" # New directory for finetuning output

# --- Load Model and Dataset (As used for initial training) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")

# --- Prepare Dummy Dataset (Replace with actual dataset) ---
DUMMY_SEQ_LEN = 64
DUMMY_DATASET_SIZE = 16
DUMMY_BATCH_SIZE = 4
dummy_data = {"input_ids": [[i] * DUMMY_SEQ_LEN for i in range(DUMMY_DATASET_SIZE)]}
dummy_tokenized_dataset = Dataset.from_dict(dummy_data)
dummy_tokenized_dataset.set_format(type='torch', columns=['input_ids'])
print("Dataset prepared.")

# --- Define Finetuning Configurations ---
print("Defining Finetuning configurations...")
# SAE config should generally match the checkpoint, 
# unless specific architecture changes are intended (advanced use)
sae_cfg = SaeConfig(expansion_factor=4, k=32)

# Training config specifies the checkpoint and new parameters
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=DUMMY_BATCH_SIZE,
    lr=5e-6,  # Often use a smaller learning rate for finetuning
    save_every=100,
    save_dir=FINETUNE_SAVE_DIR, # Save finetuning results separately
    log_to_wandb=False,
    finetune=CHECKPOINT_DIR  # Key parameter: points to the checkpoint directory
)
print("Configurations defined.")

# --- Initialize Finetuning Trainer ---
print("Initializing Finetune Trainer...")
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
# The Trainer automatically handles loading state if cfg.finetune is set.
print("Finetune Trainer initialized.")
# You might observe that training starts from a non-zero step

# --- Execute Finetuning Fit ---
# Ensure the checkpoint directory exists before running
if os.path.isdir(CHECKPOINT_DIR):
    try:
        print("Calling trainer.fit() for finetuning...")
        trainer.fit()
        print("Finetuning completed.")
        # Verify new checkpoints are saved in FINETUNE_SAVE_DIR
        assert os.path.isdir(FINETUNE_SAVE_DIR)
    except Exception as e:
        print(f"Error during finetuning: {e}")
else:
    print(f"Error: Checkpoint directory not found at {CHECKPOINT_DIR}")

# Remember to manage/clean up FINETUNE_SAVE_DIR
```

**Command-Line Interface (CLI) Implementation:**
Use the `--finetune <PATH_TO_CHECKPOINT_DIR>` argument when running `python -m sparsify`.

```bash
# Example: Finetune from a previous run saved in ./initial_run_out
# Assume ./initial_run_out exists and contains checkpoints

# Activate environment
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --finetune ./initial_run_out \
    --save_dir ./finetune_attempt_1 \
    --lr 5e-6 \
    --batch_size 4 \
    --log_to_wandb False \
    --max_examples 10 # Example: finetune for a few more examples

# Clean up ./finetune_attempt_1 afterwards if desired
```

**Implementation Tips:**
- **Checkpoint Path:** Ensure the path provided to `TrainConfig.finetune` (programmatic) or `--finetune` (CLI) points to the directory *containing* the hookpoint subdirectories from the previous run, not a hookpoint subdirectory itself.
- **Checkpoint Contents:** Verify the checkpoint directory contains all necessary files (`sae.safetensors`, `cfg.json`, and `.pt` files for optimizer, scheduler state) for the hookpoints being finetuned.
- **Model/Dataset Consistency:** Use the same base model (`MODEL_ID`) and a dataset prepared identically (tokenizer, `max_seq_len`, formatting) to the initial run for predictable behavior.
- **Learning Rate:** It's common practice to use a lower learning rate (`lr`) for finetuning compared to the initial training.
- **Save Directory:** Specify a new `save_dir` to avoid overwriting the original checkpoint, unless explicitly intended.

**Troubleshooting:**
- **`FileNotFoundError`:** Double-check the path provided to `finetune`. Verify the path exists and contains the expected hookpoint subdirectories with all required files (`.safetensors`, `.json`, `.pt`).
- **State Loading Errors:** Mismatches in `SaeConfig` between the checkpoint and the finetuning run might cause issues. More commonly, errors occur if optimizer/scheduler state files (`.pt`) are missing or incompatible (e.g., mismatch in optimizer type between runs).
- **Compatibility Issues:** Using a different base model architecture or significantly different dataset structure/tokenization during finetuning than the initial training is likely to cause errors or lead to poor results.

---

## Using Custom Hookpoints

**Description:**
Train SAEs on activations from specific modules within a base model, rather than relying on the default layer numbering (which usually targets residual streams). This allows targeting attention outputs, MLP intermediate activations, layer norms, etc.

**Prerequisites:**
- A base Hugging Face `transformers` model.
- Knowledge of the module names within the model structure. These are strings representing the attribute paths to access specific modules (e.g., `'layers.5.mlp.act_fn'`). You can typically find these by loading the model in Python and inspecting its structure (e.g., using `print(model)` or iterating through `model.named_modules()`).

**Expected Outcome:**
- The `Trainer` initializes SAEs only for the hookpoints matching the provided names or patterns.
- Training proceeds using activations gathered from these specific modules.

**Programmatic Implementation:**
Pass a list of hookpoint names or patterns (strings) to the `hookpoints` argument of `TrainConfig`.

_Based on `./tests/sparsify/test_custom_hookpoints.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset

from sparsify import SaeConfig, Trainer, TrainConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cpu"

# Dummy dataset
dummy_tokenized_dataset = Dataset.from_dict({"input_ids": [[0]*32]*4})

print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")
# print(model) # Uncomment to inspect module names

# --- Define Configurations with Custom Hookpoints ---
# Example: Target the output of the activation function in all MLP layers
# Uses fnmatch patterns (* matches anything, ? matches single char)
HOOKPOINT_PATTERNS = ["layers.*.mlp.act_fn"] 
# Other examples:
# HOOKPOINT_PATTERNS = ["layers.5.attn.q_proj", "layers.10.mlp.fc1"] # Specific modules
# HOOKPOINT_PATTERNS = ["layers.*.attn.hook_result"] # Requires model with specific hooks

sae_cfg = SaeConfig(expansion_factor=4, k=64)
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=2,
    lr=1e-4,
    save_dir="./test_trainer_init_custom_hooks_out",
    # Provide the list of patterns/names here
    hookpoints=HOOKPOINT_PATTERNS 
)
print(f"Training will target hookpoints matching: {HOOKPOINT_PATTERNS}")

# --- Initialize Trainer ---
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
print("Trainer initialized.")
print(f"Resolved hookpoints being trained: {list(trainer.saes.keys())}")

# Now trainer.fit() would train SAEs on these specific activations.
```

**Command-Line Interface (CLI) Implementation:**
Use the `--hookpoints` argument, providing one or more space-separated names or patterns (strings).

```bash
# Activate environment
# conda activate mech_interp

# Example: Train on MLP activation outputs and Attention outputs
python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --hookpoints "layers.*.mlp.act_fn" "layers.*.attn.hook_result" \
    --batch_size 4 \
    --expansion_factor 8 \
    --k 32 \
    --save_dir ./custom_hook_cli_run \
    --log_to_wandb False \
    --max_examples 10 # Run for a few steps

# Remember to clean up ./custom_hook_cli_run afterwards
```

**Implementation Tips:**
- **Finding Hookpoints:** The easiest way to find valid module names is to load the model in Python and `print(model)`. Look for the dotted attribute paths (strings) to the modules whose output activations you want to capture (e.g., `'model.layers[0].mlp.act_fn'` would likely correspond to the hookpoint string `'layers.0.mlp.act_fn'`).
- **Pattern Matching:** Use standard `fnmatch` wildcards (`*`, `?`) in the hookpoint strings for flexibility. `*` matches any sequence of characters, `?` matches a single character. Remember to quote patterns in the CLI if they contain special shell characters.
- **Specificity:** Be as specific as needed. If you only want layer 5's MLP activation, use `'layers.5.mlp.act_fn'` instead of a pattern.
- **Conflicts:** You cannot specify both `--layers` / `--layer_stride` and `--hookpoints` (or the corresponding `TrainConfig` arguments) simultaneously. Custom hookpoints take precedence.

**Troubleshooting:**
- **No SAEs Initialized:** If the pattern(s) provided don't match any module names in the loaded model, the `Trainer` might initialize with an empty `saes` dictionary or raise an error. Double-check the patterns and module names for typos or subtle differences (e.g., `block` vs. `layer`). Verify against the actual model structure.
- **Incorrect Activations:** Ensure the hookpoint string correctly identifies the *output* of the desired module operation. Training on an intermediate module like a `Linear` layer directly might capture its weights/bias instead of its output activations if not careful.

---

## Distributed Training

**Description:**
Train SAEs across multiple GPUs using PyTorch's distributed capabilities, primarily through `torchrun`. `sparsify` supports two main distributed modes:

1.  **Standard Distributed Data Parallel (DDP):** (Default when using `torchrun` without `--distribute_modules`)
    - **Replication:** Each GPU (process/rank) loads the **base model AND all SAEs** being trained.
    - **Data Sharding:** Input data batches are split across GPUs.
    - **Synchronization:** Gradients are averaged across GPUs after each backward pass.
    - **Use Case:** Provides speedup when the base model and *all* SAEs fit comfortably into each GPU's memory. Best for training fewer, larger SAEs or when memory per GPU is ample.

2.  **Module Distribution (`--distribute_modules`):** (Requires `torchrun` + `--distribute_modules` flag)
    - **Replication:** Each GPU loads the **base model**.
    - **SAE Sharding:** The set of SAEs being trained (defined by hookpoints) is **divided** among the GPUs. Each GPU only initializes and trains its assigned subset of SAEs.
    - **Synchronization:** Activations are gathered/scattered across GPUs as needed during the forward pass for SAE computation. Gradients for each SAE remain local to its assigned GPU.
    - **Use Case:** Significantly reduces memory usage per GPU for the *SAEs*, allowing training of many more SAEs simultaneously than would fit with standard DDP, especially when targeting numerous hookpoints. The base model is still replicated, so it must fit on each GPU.

**Prerequisites:**
- Multiple GPUs available on the node(s).
- PyTorch installed with CUDA and NCCL support (`torch.distributed.is_available()` and `torch.distributed.is_nccl_available()` should return `True`).
- Launching the training script using `torchrun`.

**Expected Outcome:**
- Training utilizes multiple specified GPUs.
- With DDP, faster training speed is observed compared to single GPU (assuming sufficient workload).
- With `--distribute_modules`, memory usage per GPU is lower for the SAEs compared to DDP, enabling larger-scale training across many hookpoints.

**CLI Implementation:**
Distributed training is primarily managed via the CLI by launching `python -m sparsify` with `torchrun`.

**1. Standard DDP**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train default SAEs on 2 GPUs using DDP
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 8 `# Per-GPU batch size` \
    --expansion_factor 32 \
    --k 64 \
    --save_dir ./ddp_run_out \
    --log_to_wandb False \
    --max_examples 100 # Run longer for meaningful comparison

# Clean up ./ddp_run_out afterwards
```

**2. Module Distribution**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train SAEs for many MLP layers on 2 GPUs, distributing the SAEs
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --distribute_modules `# Enable SAE sharding` \
    --hookpoints "layers.*.mlp.act_fn" `# Specify hookpoints to distribute` \
    --batch_size 4 `# Per-GPU batch size` \
    --expansion_factor 16 \
    --k 32 \
    --save_dir ./distribute_modules_run_out \
    --log_to_wandb False \
    --max_examples 100

# Clean up ./distribute_modules_run_out afterwards
```

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_full_training_pipeline.py`_
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
DATASET_SPLIT = "train[:1%]" # Use a small slice for example
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Target Hookpoint --- 
# Important: Find the correct name by inspecting the specific model architecture
# Example for Pythia-160m MLP output in layer 5
HOOKPOINT = "gpt_neox.layers.5.mlp"

# --- Load Base Model & Tokenizer ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset,
    tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    # num_proc=os.cpu_count() # Optional: for speed
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")

# --- Define Configurations ---
print("Defining SAE and Training configurations...")
# Configure the SAEs to be trained
sae_cfg = SaeConfig(
    expansion_factor=4, # Ratio of SAE latent dim to model dim (keep small for demo)
    k=32,               # Number of non-zero latents (sparsity)
)

# Configure the training process
train_cfg = TrainConfig(
    sae_cfg,            
    batch_size=8,       # Adjust based on GPU memory
    lr=1e-4,            
    save_every=1000,    
    save_dir="./sparsify_training_out", 
    hookpoints=[HOOKPOINT], # *** Crucial: Specify the target hookpoint(s) ***
    log_to_wandb=False
)
print(f"Configurations defined for hookpoint: {train_cfg.hookpoints}")

# --- Initialize Trainer ---
print("Initializing Trainer...")
# The Trainer automatically resolves input dimensions and initializes SAEs
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized successfully.")
print(f"Training SAE(s) for hookpoints: {list(trainer.saes.keys())}")

# --- Ready for Training ---
# The trainer object is now ready.
# Next step is to call trainer.fit() to start the actual training loop.
print("Setup complete. Call trainer.fit() to begin training.")

```

**Implementation Tips:**
- **Find Correct Hookpoints:** **This is the most common failure point.** Load your specific `model` and use `print(model)` or iterate through `model.named_modules()` to find the exact string name of the layer/module output you want to capture activations from. Pass this exact string(s) to `TrainConfig(hookpoints=[...])`.
- **Tokenizer Consistency:** Ensure the `tokenizer` loaded alongside the `model` matches the one used to create the `tokenized_dataset` passed to the `Trainer`. Mismatches will cause errors.
- **Dataset Formatting:** Ensure the `tokenized_dataset` is correctly formatted, typically by calling `.set_format(type='torch')` after tokenization, so it yields PyTorch tensors as expected by the `Trainer`.
- Choose a base model appropriate for your hardware resources.
- Configure `SaeConfig` and `TrainConfig` carefully, especially `hookpoints`.
- Place the base model on the appropriate device (`.to(DEVICE)`) before initializing the `Trainer`.

**Variants:**
- Use different base models (e.g., `gpt2`, Llama models if accessible).
- Train on different dataset splits.
- Adjust parameters in `SaeConfig` and `TrainConfig`.
- Specify multiple hookpoints or use wildcard patterns (e.g., `hookpoints=["gpt_neox.layers.*.mlp.act"]`) in `TrainConfig`.

**Troubleshooting:**
- **Model Loading Errors:** Ensure model identifier is correct and dependencies (`transformers`) are installed. Check hardware requirements.
- **Dataset Issues:** Verify the dataset is correctly tokenized and accessible. Ensure it's formatted correctly (e.g., `set_format(type='torch')`). Mismatched tokenizers between data prep and trainer initialization can also cause issues here.
- **Configuration Errors:** Double-check parameter names and values in `SaeConfig` and `TrainConfig`.
- **Hookpoint Errors:** If `Trainer` initialization fails with dimension errors or the optimizer fails with `empty parameter list`, the most likely cause is an incorrect hookpoint name provided in `TrainConfig.hookpoints`. Verify the name against `model.named_modules()`.
- **CUDA/Device Errors:** Ensure PyTorch is installed with CUDA support if using GPU. Verify the model and dataset tensors are on the correct device.

---

### Execute Training (`trainer.fit()`)

**Description:**
Runs the main training loop after the `Trainer` object has been initialized. This involves iterating through the dataset, computing activations from the specified hookpoint(s), calculating SAE losses, performing backpropagation, and updating SAE weights.

**Prerequisites:**
- A successfully initialized `Trainer` object (see "Programmatic Training Setup").
- Sufficient compute resources (CPU/GPU memory and processing power).

**Expected Outcome:**
- The training loop runs for the configured duration (determined by dataset size and batch size).
- Progress is logged to the console (and optionally Weights & Biases).
- Checkpoints (SAE weights `sae.safetensors` and config `cfg.json`) are saved periodically. They are organized within the `TrainConfig.save_dir` under a subdirectory named by `TrainConfig.run_name` (or `"unnamed"` if not set), and further subdivided by hookpoint name. The structure is: `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The `fit()` method completes without runtime errors.

**Tested Code Implementation:**
_This example uses the setup from the previous step (`test_full_training_pipeline.py`) and runs `fit()`._

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import shutil
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration (Match setup from previous step) ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
# Use enough data for a few steps
DATASET_SPLIT = "train[:120]" # ~15 steps with batch_size=8
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HООКРОINT = "gpt_neox.layers.5.mlp"
SAVE_DIR = "./sparsify_training_out"
SAVE_EVERY = 10 # Save frequently for demo
TARGET_BATCH_SIZE = 8

# --- Load Model & Tokenizer (As in setup) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset (As in setup) ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset, tokenizer, max_seq_len=MAX_SEQ_LEN
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")
num_steps = len(tokenized_dataset) // TARGET_BATCH_SIZE
print(f"Expected number of training steps: {num_steps}")

# --- Define Configurations (As in setup) ---
print("Defining SAE and Training configurations...")
sae_cfg = SaeConfig(expansion_factor=4, k=32)
train_cfg = TrainConfig(
    sae_cfg, 
    batch_size=TARGET_BATCH_SIZE,
    lr=1e-4, 
    save_every=SAVE_EVERY, 
    save_dir=SAVE_DIR,
    hookpoints=[HOOKPOINT], 
    log_to_wandb=False 
)
print("Configurations defined.")

# --- Initialize Trainer (As in setup) ---
print("Initializing Trainer...")
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized.")

# --- Execute Fit ---
try:
    print("Calling trainer.fit()...")
    # Training runs until dataset is exhausted
    trainer.fit()
    print("trainer.fit() completed.")

    # --- Verification ---
    print(f"Verifying checkpoint in {SAVE_DIR}...")
    # Check for base save directory
    assert os.path.isdir(SAVE_DIR), "Save directory was not created."
    # Account for potential 'unnamed' subdirectory if run_name not set
    base_save_path = os.path.join(SAVE_DIR, train_cfg.run_name or "unnamed")
    assert os.path.isdir(base_save_path), f"Base save path '{base_save_path}' not found."
    # Check for hookpoint subdirectory
    hookpoint_save_dir = os.path.join(base_save_path, HOOKPOINT)
    assert os.path.isdir(hookpoint_save_dir), f"Hookpoint subdirectory '{HOOKPOINT}' not found."
    # Check for specific files
    cfg_path = os.path.join(hookpoint_save_dir, "cfg.json")
    sae_path = os.path.join(hookpoint_save_dir, "sae.safetensors")
    assert os.path.isfile(cfg_path), "cfg.json not found."
    assert os.path.isfile(sae_path), "sae.safetensors not found."
    print("Checkpoint verification passed.")

except Exception as e:
    print(f"Error during trainer.fit(): {e}")
finally:
    # Optional: Clean up output directory after verification/use
    if os.path.exists(SAVE_DIR):
        print(f"Cleaning up directory: {SAVE_DIR}")
        shutil.rmtree(SAVE_DIR)
```

**Implementation Tips:**
- **Correct Setup is Key:** The success of `trainer.fit()` relies heavily on the correct initialization of the `Trainer` object in the preceding setup step. Ensure the base model, tokenized dataset (with matching tokenizer and correct formatting), and configurations (especially `hookpoints`) are all correctly specified.
- Ensure the `Trainer` is correctly initialized with the model, dataset, and configurations before calling `fit()`.
- Monitor resource usage (GPU memory, CPU).
- Adjust `batch_size` and `grad_acc_steps` to manage memory.
- Training duration depends on dataset size and batch size.
- Set `log_to_wandb=True` for detailed metric tracking.

**Troubleshooting:**
- **Setup Errors:** Failures during `fit()` (especially early on or related to tensor shapes/devices) can often stem from issues in the `Trainer` initialization. Double-check hookpoint names, tokenizer consistency, dataset formatting (`set_format`), and device placement from the setup step.
- **CUDA Out of Memory:** Reduce `batch_size`, use gradient accumulation (`grad_acc_steps`), use a smaller model, or use model parallelism/distribution features (see "Distributed Training").
- **Slow Training:** Increase `batch_size` (if memory allows), use GPU, use `num_proc` during data loading/tokenization, ensure efficient data loading.
- **NaN/Inf Losses:** Check learning rate (`lr`), potentially reduce it. Ensure input data is normalized correctly if needed. Check for numerical stability issues in the base model or SAE configuration.
- **Checkpointing Issues:** Verify write permissions for the `save_dir`.

---

### Command-Line Interface (CLI) Training

**Description:**
Train SAEs using the built-in command-line interface provided by `sparsify`. This offers a convenient way to run training jobs by specifying the model, dataset, and configuration options as arguments, leveraging the same underlying `Trainer` logic.

**Prerequisites:**
- The `sparsify` package installed in the environment.
- A base Hugging Face model identifier.
- Optional: A dataset identifier (defaults to `EleutherAI/fineweb-edu-dedup-10b` if omitted).
- Optional: Access to required compute resources (GPU recommended).

**Expected Outcome:**
- The training process starts, logging progress to the console.
- Checkpoints are saved to subdirectories within the default directory (`./checkpoints`) or a specified output directory (`--save_dir`), following the structure `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The command runs to completion or until manually stopped.

**Tested Code Implementation:**
_The following command was verified to execute successfully, running for a single step using a small model and dataset. Replace parameters as needed for actual training runs._

```bash
# Activate your conda environment first
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 1 \
    --expansion_factor 1 \
    --k 16 \
    --log_to_wandb False \
    --save_dir ./test_cli_run_out \
    --max_examples 1

# Expected output: Command runs, prints progress, saves checkpoint, and exits.
# Remember to clean up ./test_cli_run_out afterwards if desired.
```

**Implementation Tips:**
- Use `python -m sparsify --help` to see all available CLI options. Most arguments directly map to fields in the programmatic `SaeConfig` and `TrainConfig` (e.g., `--batch_size` maps to `TrainConfig.batch_size`, `--expansion_factor` maps to `SaeConfig.expansion_factor`).
- Specify arguments using `--arg_name value` format (e.g., `--batch_size 16`, `--expansion_factor 32`).
- Use boolean flags like `--load_in_8bit`. To explicitly set a boolean flag to `False` when its default is `True`, use the `--no-<flag_name>` syntax if available (check `--help`), otherwise omit the flag.
- For multi-GPU training, prepend the command with `torchrun --nproc_per_node <num_gpus>` (see "Distributed Training").
- By default, logs are sent to Weights & Biases if available; use `--no-log_to_wandb` to disable.

**Variants:**
- Specify a different dataset (Hub ID or local path).
- Override default `SaeConfig` parameters (e.g., `--k 128`, `--expansion_factor 64`).
- Override default `TrainConfig` parameters (e.g., `--lr 1e-4`, `--save_dir ./my_sae_run`).
- Train on specific layers (`--layers 0 5 10`) or using a stride (`--layer_stride 4`).
- Use custom hookpoints (`--hookpoints "h.*.attn" "h.*.mlp.act"`).

**Troubleshooting:**
- **Argument Errors:** Check `python -m sparsify --help` for correct argument names and types. Ensure boolean flags use the correct syntax (`--flag_name` or `--no-flag_name`).
- **Incompatible Arguments:** Some argument combinations are disallowed (e.g., using both `--layers` and `--hookpoints`). The CLI should raise an error if incompatible arguments are provided.
- **Model/Dataset Loading Errors:** Verify model and dataset identifiers are correct and accessible.
- **Resource Errors (OOM):** Reduce `--batch_size`, use `--grad_acc_steps N`, use `--load_in_8bit`, or use distributed training options if applicable.
- **`torchrun` Issues:** Ensure `torchrun` is available and configured correctly for your environment if doing multi-GPU training.

---

## Finetuning SAEs

**Description:**
Continue training SAEs from a previously saved checkpoint. This is useful for resuming interrupted training runs or further refining SAEs with different data or learning rates.

**Prerequisites:**
- A valid checkpoint directory path. This path should point to the directory containing the hookpoint subdirectories (e.g., the `{save_dir}/{run_name or "unnamed"}/` directory from a previous run). Each relevant hookpoint subdirectory must contain:
    - `sae.safetensors` (SAE weights)
    - `cfg.json` (SAE config)
    - `optimizer_*.pt` (Optimizer state for the SAE)
    - `scheduler_*.pt` (Scheduler state, if used)
    - `scaler_*.pt` (Gradient scaler state, if used)
- The **same base model** (`MODEL_ID`) used for the initial training.
- A **tokenized dataset** prepared using the **same tokenizer and formatting** as the initial training run.

**Expected Outcome:**
- The `Trainer` initializes, loading the SAE weights and optimizer state from the specified checkpoint directory.
- Training resumes from the step count saved in the checkpoint.
- The finetuning process runs, potentially saving new checkpoints to a different directory.

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_finetune.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset
import os

from sparsify import SaeConfig, Trainer, TrainConfig

# --- Configuration (Assume initial checkpoint exists) ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./test_finetune_checkpoint_source" # Path to the existing checkpoint
FINETUNE_SAVE_DIR = "./test_finetune_run_out" # New directory for finetuning output

# --- Load Model and Dataset (As used for initial training) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")

# --- Prepare Dummy Dataset (Replace with actual dataset) ---
DUMMY_SEQ_LEN = 64
DUMMY_DATASET_SIZE = 16
DUMMY_BATCH_SIZE = 4
dummy_data = {"input_ids": [[i] * DUMMY_SEQ_LEN for i in range(DUMMY_DATASET_SIZE)]}
dummy_tokenized_dataset = Dataset.from_dict(dummy_data)
dummy_tokenized_dataset.set_format(type='torch', columns=['input_ids'])
print("Dataset prepared.")

# --- Define Finetuning Configurations ---
print("Defining Finetuning configurations...")
# SAE config should generally match the checkpoint, 
# unless specific architecture changes are intended (advanced use)
sae_cfg = SaeConfig(expansion_factor=4, k=32)

# Training config specifies the checkpoint and new parameters
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=DUMMY_BATCH_SIZE,
    lr=5e-6,  # Often use a smaller learning rate for finetuning
    save_every=100,
    save_dir=FINETUNE_SAVE_DIR, # Save finetuning results separately
    log_to_wandb=False,
    finetune=CHECKPOINT_DIR  # Key parameter: points to the checkpoint directory
)
print("Configurations defined.")

# --- Initialize Finetuning Trainer ---
print("Initializing Finetune Trainer...")
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
# The Trainer automatically handles loading state if cfg.finetune is set.
print("Finetune Trainer initialized.")
# You might observe that training starts from a non-zero step

# --- Execute Finetuning Fit ---
# Ensure the checkpoint directory exists before running
if os.path.isdir(CHECKPOINT_DIR):
    try:
        print("Calling trainer.fit() for finetuning...")
        trainer.fit()
        print("Finetuning completed.")
        # Verify new checkpoints are saved in FINETUNE_SAVE_DIR
        assert os.path.isdir(FINETUNE_SAVE_DIR)
    except Exception as e:
        print(f"Error during finetuning: {e}")
else:
    print(f"Error: Checkpoint directory not found at {CHECKPOINT_DIR}")

# Remember to manage/clean up FINETUNE_SAVE_DIR
```

**Command-Line Interface (CLI) Implementation:**
Use the `--finetune <PATH_TO_CHECKPOINT_DIR>` argument when running `python -m sparsify`.

```bash
# Example: Finetune from a previous run saved in ./initial_run_out
# Assume ./initial_run_out exists and contains checkpoints

# Activate environment
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --finetune ./initial_run_out \
    --save_dir ./finetune_attempt_1 \
    --lr 5e-6 \
    --batch_size 4 \
    --log_to_wandb False \
    --max_examples 10 # Example: finetune for a few more examples

# Clean up ./finetune_attempt_1 afterwards if desired
```

**Implementation Tips:**
- **Checkpoint Path:** Ensure the path provided to `TrainConfig.finetune` (programmatic) or `--finetune` (CLI) points to the directory *containing* the hookpoint subdirectories from the previous run, not a hookpoint subdirectory itself.
- **Checkpoint Contents:** Verify the checkpoint directory contains all necessary files (`sae.safetensors`, `cfg.json`, and `.pt` files for optimizer, scheduler state) for the hookpoints being finetuned.
- **Model/Dataset Consistency:** Use the same base model (`MODEL_ID`) and a dataset prepared identically (tokenizer, `max_seq_len`, formatting) to the initial run for predictable behavior.
- **Learning Rate:** It's common practice to use a lower learning rate (`lr`) for finetuning compared to the initial training.
- **Save Directory:** Specify a new `save_dir` to avoid overwriting the original checkpoint, unless explicitly intended.

**Troubleshooting:**
- **`FileNotFoundError`:** Double-check the path provided to `finetune`. Verify the path exists and contains the expected hookpoint subdirectories with all required files (`.safetensors`, `.json`, `.pt`).
- **State Loading Errors:** Mismatches in `SaeConfig` between the checkpoint and the finetuning run might cause issues. More commonly, errors occur if optimizer/scheduler state files (`.pt`) are missing or incompatible (e.g., mismatch in optimizer type between runs).
- **Compatibility Issues:** Using a different base model architecture or significantly different dataset structure/tokenization during finetuning than the initial training is likely to cause errors or lead to poor results.

---

## Using Custom Hookpoints

**Description:**
Train SAEs on activations from specific modules within a base model, rather than relying on the default layer numbering (which usually targets residual streams). This allows targeting attention outputs, MLP intermediate activations, layer norms, etc.

**Prerequisites:**
- A base Hugging Face `transformers` model.
- Knowledge of the module names within the model structure. These are strings representing the attribute paths to access specific modules (e.g., `'layers.5.mlp.act_fn'`). You can typically find these by loading the model in Python and inspecting its structure (e.g., using `print(model)` or iterating through `model.named_modules()`).

**Expected Outcome:**
- The `Trainer` initializes SAEs only for the hookpoints matching the provided names or patterns.
- Training proceeds using activations gathered from these specific modules.

**Programmatic Implementation:**
Pass a list of hookpoint names or patterns (strings) to the `hookpoints` argument of `TrainConfig`.

_Based on `./tests/sparsify/test_custom_hookpoints.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset

from sparsify import SaeConfig, Trainer, TrainConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cpu"

# Dummy dataset
dummy_tokenized_dataset = Dataset.from_dict({"input_ids": [[0]*32]*4})

print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")
# print(model) # Uncomment to inspect module names

# --- Define Configurations with Custom Hookpoints ---
# Example: Target the output of the activation function in all MLP layers
# Uses fnmatch patterns (* matches anything, ? matches single char)
HOOKPOINT_PATTERNS = ["layers.*.mlp.act_fn"] 
# Other examples:
# HOOKPOINT_PATTERNS = ["layers.5.attn.q_proj", "layers.10.mlp.fc1"] # Specific modules
# HOOKPOINT_PATTERNS = ["layers.*.attn.hook_result"] # Requires model with specific hooks

sae_cfg = SaeConfig(expansion_factor=4, k=64)
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=2,
    lr=1e-4,
    save_dir="./test_trainer_init_custom_hooks_out",
    # Provide the list of patterns/names here
    hookpoints=HOOKPOINT_PATTERNS 
)
print(f"Training will target hookpoints matching: {HOOKPOINT_PATTERNS}")

# --- Initialize Trainer ---
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
print("Trainer initialized.")
print(f"Resolved hookpoints being trained: {list(trainer.saes.keys())}")

# Now trainer.fit() would train SAEs on these specific activations.
```

**Command-Line Interface (CLI) Implementation:**
Use the `--hookpoints` argument, providing one or more space-separated names or patterns (strings).

```bash
# Activate environment
# conda activate mech_interp

# Example: Train on MLP activation outputs and Attention outputs
python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --hookpoints "layers.*.mlp.act_fn" "layers.*.attn.hook_result" \
    --batch_size 4 \
    --expansion_factor 8 \
    --k 32 \
    --save_dir ./custom_hook_cli_run \
    --log_to_wandb False \
    --max_examples 10 # Run for a few steps

# Remember to clean up ./custom_hook_cli_run afterwards
```

**Implementation Tips:**
- **Finding Hookpoints:** The easiest way to find valid module names is to load the model in Python and `print(model)`. Look for the dotted attribute paths (strings) to the modules whose output activations you want to capture (e.g., `'model.layers[0].mlp.act_fn'` would likely correspond to the hookpoint string `'layers.0.mlp.act_fn'`).
- **Pattern Matching:** Use standard `fnmatch` wildcards (`*`, `?`) in the hookpoint strings for flexibility. `*` matches any sequence of characters, `?` matches a single character. Remember to quote patterns in the CLI if they contain special shell characters.
- **Specificity:** Be as specific as needed. If you only want layer 5's MLP activation, use `'layers.5.mlp.act_fn'` instead of a pattern.
- **Conflicts:** You cannot specify both `--layers` / `--layer_stride` and `--hookpoints` (or the corresponding `TrainConfig` arguments) simultaneously. Custom hookpoints take precedence.

**Troubleshooting:**
- **No SAEs Initialized:** If the pattern(s) provided don't match any module names in the loaded model, the `Trainer` might initialize with an empty `saes` dictionary or raise an error. Double-check the patterns and module names for typos or subtle differences (e.g., `block` vs. `layer`). Verify against the actual model structure.
- **Incorrect Activations:** Ensure the hookpoint string correctly identifies the *output* of the desired module operation. Training on an intermediate module like a `Linear` layer directly might capture its weights/bias instead of its output activations if not careful.

---

## Distributed Training

**Description:**
Train SAEs across multiple GPUs using PyTorch's distributed capabilities, primarily through `torchrun`. `sparsify` supports two main distributed modes:

1.  **Standard Distributed Data Parallel (DDP):** (Default when using `torchrun` without `--distribute_modules`)
    - **Replication:** Each GPU (process/rank) loads the **base model AND all SAEs** being trained.
    - **Data Sharding:** Input data batches are split across GPUs.
    - **Synchronization:** Gradients are averaged across GPUs after each backward pass.
    - **Use Case:** Provides speedup when the base model and *all* SAEs fit comfortably into each GPU's memory. Best for training fewer, larger SAEs or when memory per GPU is ample.

2.  **Module Distribution (`--distribute_modules`):** (Requires `torchrun` + `--distribute_modules` flag)
    - **Replication:** Each GPU loads the **base model**.
    - **SAE Sharding:** The set of SAEs being trained (defined by hookpoints) is **divided** among the GPUs. Each GPU only initializes and trains its assigned subset of SAEs.
    - **Synchronization:** Activations are gathered/scattered across GPUs as needed during the forward pass for SAE computation. Gradients for each SAE remain local to its assigned GPU.
    - **Use Case:** Significantly reduces memory usage per GPU for the *SAEs*, allowing training of many more SAEs simultaneously than would fit with standard DDP, especially when targeting numerous hookpoints. The base model is still replicated, so it must fit on each GPU.

**Prerequisites:**
- Multiple GPUs available on the node(s).
- PyTorch installed with CUDA and NCCL support (`torch.distributed.is_available()` and `torch.distributed.is_nccl_available()` should return `True`).
- Launching the training script using `torchrun`.

**Expected Outcome:**
- Training utilizes multiple specified GPUs.
- With DDP, faster training speed is observed compared to single GPU (assuming sufficient workload).
- With `--distribute_modules`, memory usage per GPU is lower for the SAEs compared to DDP, enabling larger-scale training across many hookpoints.

**CLI Implementation:**
Distributed training is primarily managed via the CLI by launching `python -m sparsify` with `torchrun`.

**1. Standard DDP**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train default SAEs on 2 GPUs using DDP
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 8 `# Per-GPU batch size` \
    --expansion_factor 32 \
    --k 64 \
    --save_dir ./ddp_run_out \
    --log_to_wandb False \
    --max_examples 100 # Run longer for meaningful comparison

# Clean up ./ddp_run_out afterwards
```

**2. Module Distribution**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train SAEs for many MLP layers on 2 GPUs, distributing the SAEs
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --distribute_modules `# Enable SAE sharding` \
    --hookpoints "layers.*.mlp.act_fn" `# Specify hookpoints to distribute` \
    --batch_size 4 `# Per-GPU batch size` \
    --expansion_factor 16 \
    --k 32 \
    --save_dir ./distribute_modules_run_out \
    --log_to_wandb False \
    --max_examples 100

# Clean up ./distribute_modules_run_out afterwards
```

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_full_training_pipeline.py`_
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
DATASET_SPLIT = "train[:1%]" # Use a small slice for example
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Target Hookpoint --- 
# Important: Find the correct name by inspecting the specific model architecture
# Example for Pythia-160m MLP output in layer 5
HOOKPOINT = "gpt_neox.layers.5.mlp"

# --- Load Base Model & Tokenizer ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset,
    tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    # num_proc=os.cpu_count() # Optional: for speed
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")

# --- Define Configurations ---
print("Defining SAE and Training configurations...")
# Configure the SAEs to be trained
sae_cfg = SaeConfig(
    expansion_factor=4, # Ratio of SAE latent dim to model dim (keep small for demo)
    k=32,               # Number of non-zero latents (sparsity)
)

# Configure the training process
train_cfg = TrainConfig(
    sae_cfg,            
    batch_size=8,       # Adjust based on GPU memory
    lr=1e-4,            
    save_every=1000,    
    save_dir="./sparsify_training_out", 
    hookpoints=[HOOKPOINT], # *** Crucial: Specify the target hookpoint(s) ***
    log_to_wandb=False
)
print(f"Configurations defined for hookpoint: {train_cfg.hookpoints}")

# --- Initialize Trainer ---
print("Initializing Trainer...")
# The Trainer automatically resolves input dimensions and initializes SAEs
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized successfully.")
print(f"Training SAE(s) for hookpoints: {list(trainer.saes.keys())}")

# --- Ready for Training ---
# The trainer object is now ready.
# Next step is to call trainer.fit() to start the actual training loop.
print("Setup complete. Call trainer.fit() to begin training.")

```

**Implementation Tips:**
- **Find Correct Hookpoints:** **This is the most common failure point.** Load your specific `model` and use `print(model)` or iterate through `model.named_modules()` to find the exact string name of the layer/module output you want to capture activations from. Pass this exact string(s) to `TrainConfig(hookpoints=[...])`.
- **Tokenizer Consistency:** Ensure the `tokenizer` loaded alongside the `model` matches the one used to create the `tokenized_dataset` passed to the `Trainer`. Mismatches will cause errors.
- **Dataset Formatting:** Ensure the `tokenized_dataset` is correctly formatted, typically by calling `.set_format(type='torch')` after tokenization, so it yields PyTorch tensors as expected by the `Trainer`.
- Choose a base model appropriate for your hardware resources.
- Configure `SaeConfig` and `TrainConfig` carefully, especially `hookpoints`.
- Place the base model on the appropriate device (`.to(DEVICE)`) before initializing the `Trainer`.

**Variants:**
- Use different base models (e.g., `gpt2`, Llama models if accessible).
- Train on different dataset splits.
- Adjust parameters in `SaeConfig` and `TrainConfig`.
- Specify multiple hookpoints or use wildcard patterns (e.g., `hookpoints=["gpt_neox.layers.*.mlp.act"]`) in `TrainConfig`.

**Troubleshooting:**
- **Model Loading Errors:** Ensure model identifier is correct and dependencies (`transformers`) are installed. Check hardware requirements.
- **Dataset Issues:** Verify the dataset is correctly tokenized and accessible. Ensure it's formatted correctly (e.g., `set_format(type='torch')`). Mismatched tokenizers between data prep and trainer initialization can also cause issues here.
- **Configuration Errors:** Double-check parameter names and values in `SaeConfig` and `TrainConfig`.
- **Hookpoint Errors:** If `Trainer` initialization fails with dimension errors or the optimizer fails with `empty parameter list`, the most likely cause is an incorrect hookpoint name provided in `TrainConfig.hookpoints`. Verify the name against `model.named_modules()`.
- **CUDA/Device Errors:** Ensure PyTorch is installed with CUDA support if using GPU. Verify the model and dataset tensors are on the correct device.

---

### Execute Training (`trainer.fit()`)

**Description:**
Runs the main training loop after the `Trainer` object has been initialized. This involves iterating through the dataset, computing activations from the specified hookpoint(s), calculating SAE losses, performing backpropagation, and updating SAE weights.

**Prerequisites:**
- A successfully initialized `Trainer` object (see "Programmatic Training Setup").
- Sufficient compute resources (CPU/GPU memory and processing power).

**Expected Outcome:**
- The training loop runs for the configured duration (determined by dataset size and batch size).
- Progress is logged to the console (and optionally Weights & Biases).
- Checkpoints (SAE weights `sae.safetensors` and config `cfg.json`) are saved periodically. They are organized within the `TrainConfig.save_dir` under a subdirectory named by `TrainConfig.run_name` (or `"unnamed"` if not set), and further subdivided by hookpoint name. The structure is: `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The `fit()` method completes without runtime errors.

**Tested Code Implementation:**
_This example uses the setup from the previous step (`test_full_training_pipeline.py`) and runs `fit()`._

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import shutil
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration (Match setup from previous step) ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
# Use enough data for a few steps
DATASET_SPLIT = "train[:120]" # ~15 steps with batch_size=8
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HООКРОINT = "gpt_neox.layers.5.mlp"
SAVE_DIR = "./sparsify_training_out"
SAVE_EVERY = 10 # Save frequently for demo
TARGET_BATCH_SIZE = 8

# --- Load Model & Tokenizer (As in setup) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset (As in setup) ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset, tokenizer, max_seq_len=MAX_SEQ_LEN
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")
num_steps = len(tokenized_dataset) // TARGET_BATCH_SIZE
print(f"Expected number of training steps: {num_steps}")

# --- Define Configurations (As in setup) ---
print("Defining SAE and Training configurations...")
sae_cfg = SaeConfig(expansion_factor=4, k=32)
train_cfg = TrainConfig(
    sae_cfg, 
    batch_size=TARGET_BATCH_SIZE,
    lr=1e-4, 
    save_every=SAVE_EVERY, 
    save_dir=SAVE_DIR,
    hookpoints=[HOOKPOINT], 
    log_to_wandb=False 
)
print("Configurations defined.")

# --- Initialize Trainer (As in setup) ---
print("Initializing Trainer...")
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized.")

# --- Execute Fit ---
try:
    print("Calling trainer.fit()...")
    # Training runs until dataset is exhausted
    trainer.fit()
    print("trainer.fit() completed.")

    # --- Verification ---
    print(f"Verifying checkpoint in {SAVE_DIR}...")
    # Check for base save directory
    assert os.path.isdir(SAVE_DIR), "Save directory was not created."
    # Account for potential 'unnamed' subdirectory if run_name not set
    base_save_path = os.path.join(SAVE_DIR, train_cfg.run_name or "unnamed")
    assert os.path.isdir(base_save_path), f"Base save path '{base_save_path}' not found."
    # Check for hookpoint subdirectory
    hookpoint_save_dir = os.path.join(base_save_path, HOOKPOINT)
    assert os.path.isdir(hookpoint_save_dir), f"Hookpoint subdirectory '{HOOKPOINT}' not found."
    # Check for specific files
    cfg_path = os.path.join(hookpoint_save_dir, "cfg.json")
    sae_path = os.path.join(hookpoint_save_dir, "sae.safetensors")
    assert os.path.isfile(cfg_path), "cfg.json not found."
    assert os.path.isfile(sae_path), "sae.safetensors not found."
    print("Checkpoint verification passed.")

except Exception as e:
    print(f"Error during trainer.fit(): {e}")
finally:
    # Optional: Clean up output directory after verification/use
    if os.path.exists(SAVE_DIR):
        print(f"Cleaning up directory: {SAVE_DIR}")
        shutil.rmtree(SAVE_DIR)
```

**Implementation Tips:**
- **Correct Setup is Key:** The success of `trainer.fit()` relies heavily on the correct initialization of the `Trainer` object in the preceding setup step. Ensure the base model, tokenized dataset (with matching tokenizer and correct formatting), and configurations (especially `hookpoints`) are all correctly specified.
- Ensure the `Trainer` is correctly initialized with the model, dataset, and configurations before calling `fit()`.
- Monitor resource usage (GPU memory, CPU).
- Adjust `batch_size` and `grad_acc_steps` to manage memory.
- Training duration depends on dataset size and batch size.
- Set `log_to_wandb=True` for detailed metric tracking.

**Troubleshooting:**
- **Setup Errors:** Failures during `fit()` (especially early on or related to tensor shapes/devices) can often stem from issues in the `Trainer` initialization. Double-check hookpoint names, tokenizer consistency, dataset formatting (`set_format`), and device placement from the setup step.
- **CUDA Out of Memory:** Reduce `batch_size`, use gradient accumulation (`grad_acc_steps`), use a smaller model, or use model parallelism/distribution features (see "Distributed Training").
- **Slow Training:** Increase `batch_size` (if memory allows), use GPU, use `num_proc` during data loading/tokenization, ensure efficient data loading.
- **NaN/Inf Losses:** Check learning rate (`lr`), potentially reduce it. Ensure input data is normalized correctly if needed. Check for numerical stability issues in the base model or SAE configuration.
- **Checkpointing Issues:** Verify write permissions for the `save_dir`.

---

### Command-Line Interface (CLI) Training

**Description:**
Train SAEs using the built-in command-line interface provided by `sparsify`. This offers a convenient way to run training jobs by specifying the model, dataset, and configuration options as arguments, leveraging the same underlying `Trainer` logic.

**Prerequisites:**
- The `sparsify` package installed in the environment.
- A base Hugging Face model identifier.
- Optional: A dataset identifier (defaults to `EleutherAI/fineweb-edu-dedup-10b` if omitted).
- Optional: Access to required compute resources (GPU recommended).

**Expected Outcome:**
- The training process starts, logging progress to the console.
- Checkpoints are saved to subdirectories within the default directory (`./checkpoints`) or a specified output directory (`--save_dir`), following the structure `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The command runs to completion or until manually stopped.

**Tested Code Implementation:**
_The following command was verified to execute successfully, running for a single step using a small model and dataset. Replace parameters as needed for actual training runs._

```bash
# Activate your conda environment first
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 1 \
    --expansion_factor 1 \
    --k 16 \
    --log_to_wandb False \
    --save_dir ./test_cli_run_out \
    --max_examples 1

# Expected output: Command runs, prints progress, saves checkpoint, and exits.
# Remember to clean up ./test_cli_run_out afterwards if desired.
```

**Implementation Tips:**
- Use `python -m sparsify --help` to see all available CLI options. Most arguments directly map to fields in the programmatic `SaeConfig` and `TrainConfig` (e.g., `--batch_size` maps to `TrainConfig.batch_size`, `--expansion_factor` maps to `SaeConfig.expansion_factor`).
- Specify arguments using `--arg_name value` format (e.g., `--batch_size 16`, `--expansion_factor 32`).
- Use boolean flags like `--load_in_8bit`. To explicitly set a boolean flag to `False` when its default is `True`, use the `--no-<flag_name>` syntax if available (check `--help`), otherwise omit the flag.
- For multi-GPU training, prepend the command with `torchrun --nproc_per_node <num_gpus>` (see "Distributed Training").
- By default, logs are sent to Weights & Biases if available; use `--no-log_to_wandb` to disable.

**Variants:**
- Specify a different dataset (Hub ID or local path).
- Override default `SaeConfig` parameters (e.g., `--k 128`, `--expansion_factor 64`).
- Override default `TrainConfig` parameters (e.g., `--lr 1e-4`, `--save_dir ./my_sae_run`).
- Train on specific layers (`--layers 0 5 10`) or using a stride (`--layer_stride 4`).
- Use custom hookpoints (`--hookpoints "h.*.attn" "h.*.mlp.act"`).

**Troubleshooting:**
- **Argument Errors:** Check `python -m sparsify --help` for correct argument names and types. Ensure boolean flags use the correct syntax (`--flag_name` or `--no-flag_name`).
- **Incompatible Arguments:** Some argument combinations are disallowed (e.g., using both `--layers` and `--hookpoints`). The CLI should raise an error if incompatible arguments are provided.
- **Model/Dataset Loading Errors:** Verify model and dataset identifiers are correct and accessible.
- **Resource Errors (OOM):** Reduce `--batch_size`, use `--grad_acc_steps N`, use `--load_in_8bit`, or use distributed training options if applicable.
- **`torchrun` Issues:** Ensure `torchrun` is available and configured correctly for your environment if doing multi-GPU training.

---

## Finetuning SAEs

**Description:**
Continue training SAEs from a previously saved checkpoint. This is useful for resuming interrupted training runs or further refining SAEs with different data or learning rates.

**Prerequisites:**
- A valid checkpoint directory path. This path should point to the directory containing the hookpoint subdirectories (e.g., the `{save_dir}/{run_name or "unnamed"}/` directory from a previous run). Each relevant hookpoint subdirectory must contain:
    - `sae.safetensors` (SAE weights)
    - `cfg.json` (SAE config)
    - `optimizer_*.pt` (Optimizer state for the SAE)
    - `scheduler_*.pt` (Scheduler state, if used)
    - `scaler_*.pt` (Gradient scaler state, if used)
- The **same base model** (`MODEL_ID`) used for the initial training.
- A **tokenized dataset** prepared using the **same tokenizer and formatting** as the initial training run.

**Expected Outcome:**
- The `Trainer` initializes, loading the SAE weights and optimizer state from the specified checkpoint directory.
- Training resumes from the step count saved in the checkpoint.
- The finetuning process runs, potentially saving new checkpoints to a different directory.

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_finetune.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset
import os

from sparsify import SaeConfig, Trainer, TrainConfig

# --- Configuration (Assume initial checkpoint exists) ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./test_finetune_checkpoint_source" # Path to the existing checkpoint
FINETUNE_SAVE_DIR = "./test_finetune_run_out" # New directory for finetuning output

# --- Load Model and Dataset (As used for initial training) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")

# --- Prepare Dummy Dataset (Replace with actual dataset) ---
DUMMY_SEQ_LEN = 64
DUMMY_DATASET_SIZE = 16
DUMMY_BATCH_SIZE = 4
dummy_data = {"input_ids": [[i] * DUMMY_SEQ_LEN for i in range(DUMMY_DATASET_SIZE)]}
dummy_tokenized_dataset = Dataset.from_dict(dummy_data)
dummy_tokenized_dataset.set_format(type='torch', columns=['input_ids'])
print("Dataset prepared.")

# --- Define Finetuning Configurations ---
print("Defining Finetuning configurations...")
# SAE config should generally match the checkpoint, 
# unless specific architecture changes are intended (advanced use)
sae_cfg = SaeConfig(expansion_factor=4, k=32)

# Training config specifies the checkpoint and new parameters
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=DUMMY_BATCH_SIZE,
    lr=5e-6,  # Often use a smaller learning rate for finetuning
    save_every=100,
    save_dir=FINETUNE_SAVE_DIR, # Save finetuning results separately
    log_to_wandb=False,
    finetune=CHECKPOINT_DIR  # Key parameter: points to the checkpoint directory
)
print("Configurations defined.")

# --- Initialize Finetuning Trainer ---
print("Initializing Finetune Trainer...")
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
# The Trainer automatically handles loading state if cfg.finetune is set.
print("Finetune Trainer initialized.")
# You might observe that training starts from a non-zero step

# --- Execute Finetuning Fit ---
# Ensure the checkpoint directory exists before running
if os.path.isdir(CHECKPOINT_DIR):
    try:
        print("Calling trainer.fit() for finetuning...")
        trainer.fit()
        print("Finetuning completed.")
        # Verify new checkpoints are saved in FINETUNE_SAVE_DIR
        assert os.path.isdir(FINETUNE_SAVE_DIR)
    except Exception as e:
        print(f"Error during finetuning: {e}")
else:
    print(f"Error: Checkpoint directory not found at {CHECKPOINT_DIR}")

# Remember to manage/clean up FINETUNE_SAVE_DIR
```

**Command-Line Interface (CLI) Implementation:**
Use the `--finetune <PATH_TO_CHECKPOINT_DIR>` argument when running `python -m sparsify`.

```bash
# Example: Finetune from a previous run saved in ./initial_run_out
# Assume ./initial_run_out exists and contains checkpoints

# Activate environment
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --finetune ./initial_run_out \
    --save_dir ./finetune_attempt_1 \
    --lr 5e-6 \
    --batch_size 4 \
    --log_to_wandb False \
    --max_examples 10 # Example: finetune for a few more examples

# Clean up ./finetune_attempt_1 afterwards if desired
```

**Implementation Tips:**
- **Checkpoint Path:** Ensure the path provided to `TrainConfig.finetune` (programmatic) or `--finetune` (CLI) points to the directory *containing* the hookpoint subdirectories from the previous run, not a hookpoint subdirectory itself.
- **Checkpoint Contents:** Verify the checkpoint directory contains all necessary files (`sae.safetensors`, `cfg.json`, and `.pt` files for optimizer, scheduler state) for the hookpoints being finetuned.
- **Model/Dataset Consistency:** Use the same base model (`MODEL_ID`) and a dataset prepared identically (tokenizer, `max_seq_len`, formatting) to the initial run for predictable behavior.
- **Learning Rate:** It's common practice to use a lower learning rate (`lr`) for finetuning compared to the initial training.
- **Save Directory:** Specify a new `save_dir` to avoid overwriting the original checkpoint, unless explicitly intended.

**Troubleshooting:**
- **`FileNotFoundError`:** Double-check the path provided to `finetune`. Verify the path exists and contains the expected hookpoint subdirectories with all required files (`.safetensors`, `.json`, `.pt`).
- **State Loading Errors:** Mismatches in `SaeConfig` between the checkpoint and the finetuning run might cause issues. More commonly, errors occur if optimizer/scheduler state files (`.pt`) are missing or incompatible (e.g., mismatch in optimizer type between runs).
- **Compatibility Issues:** Using a different base model architecture or significantly different dataset structure/tokenization during finetuning than the initial training is likely to cause errors or lead to poor results.

---

## Using Custom Hookpoints

**Description:**
Train SAEs on activations from specific modules within a base model, rather than relying on the default layer numbering (which usually targets residual streams). This allows targeting attention outputs, MLP intermediate activations, layer norms, etc.

**Prerequisites:**
- A base Hugging Face `transformers` model.
- Knowledge of the module names within the model structure. These are strings representing the attribute paths to access specific modules (e.g., `'layers.5.mlp.act_fn'`). You can typically find these by loading the model in Python and inspecting its structure (e.g., using `print(model)` or iterating through `model.named_modules()`).

**Expected Outcome:**
- The `Trainer` initializes SAEs only for the hookpoints matching the provided names or patterns.
- Training proceeds using activations gathered from these specific modules.

**Programmatic Implementation:**
Pass a list of hookpoint names or patterns (strings) to the `hookpoints` argument of `TrainConfig`.

_Based on `./tests/sparsify/test_custom_hookpoints.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset

from sparsify import SaeConfig, Trainer, TrainConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cpu"

# Dummy dataset
dummy_tokenized_dataset = Dataset.from_dict({"input_ids": [[0]*32]*4})

print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")
# print(model) # Uncomment to inspect module names

# --- Define Configurations with Custom Hookpoints ---
# Example: Target the output of the activation function in all MLP layers
# Uses fnmatch patterns (* matches anything, ? matches single char)
HOOKPOINT_PATTERNS = ["layers.*.mlp.act_fn"] 
# Other examples:
# HOOKPOINT_PATTERNS = ["layers.5.attn.q_proj", "layers.10.mlp.fc1"] # Specific modules
# HOOKPOINT_PATTERNS = ["layers.*.attn.hook_result"] # Requires model with specific hooks

sae_cfg = SaeConfig(expansion_factor=4, k=64)
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=2,
    lr=1e-4,
    save_dir="./test_trainer_init_custom_hooks_out",
    # Provide the list of patterns/names here
    hookpoints=HOOKPOINT_PATTERNS 
)
print(f"Training will target hookpoints matching: {HOOKPOINT_PATTERNS}")

# --- Initialize Trainer ---
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
print("Trainer initialized.")
print(f"Resolved hookpoints being trained: {list(trainer.saes.keys())}")

# Now trainer.fit() would train SAEs on these specific activations.
```

**Command-Line Interface (CLI) Implementation:**
Use the `--hookpoints` argument, providing one or more space-separated names or patterns (strings).

```bash
# Activate environment
# conda activate mech_interp

# Example: Train on MLP activation outputs and Attention outputs
python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --hookpoints "layers.*.mlp.act_fn" "layers.*.attn.hook_result" \
    --batch_size 4 \
    --expansion_factor 8 \
    --k 32 \
    --save_dir ./custom_hook_cli_run \
    --log_to_wandb False \
    --max_examples 10 # Run for a few steps

# Remember to clean up ./custom_hook_cli_run afterwards
```

**Implementation Tips:**
- **Finding Hookpoints:** The easiest way to find valid module names is to load the model in Python and `print(model)`. Look for the dotted attribute paths (strings) to the modules whose output activations you want to capture (e.g., `'model.layers[0].mlp.act_fn'` would likely correspond to the hookpoint string `'layers.0.mlp.act_fn'`).
- **Pattern Matching:** Use standard `fnmatch` wildcards (`*`, `?`) in the hookpoint strings for flexibility. `*` matches any sequence of characters, `?` matches a single character. Remember to quote patterns in the CLI if they contain special shell characters.
- **Specificity:** Be as specific as needed. If you only want layer 5's MLP activation, use `'layers.5.mlp.act_fn'` instead of a pattern.
- **Conflicts:** You cannot specify both `--layers` / `--layer_stride` and `--hookpoints` (or the corresponding `TrainConfig` arguments) simultaneously. Custom hookpoints take precedence.

**Troubleshooting:**
- **No SAEs Initialized:** If the pattern(s) provided don't match any module names in the loaded model, the `Trainer` might initialize with an empty `saes` dictionary or raise an error. Double-check the patterns and module names for typos or subtle differences (e.g., `block` vs. `layer`). Verify against the actual model structure.
- **Incorrect Activations:** Ensure the hookpoint string correctly identifies the *output* of the desired module operation. Training on an intermediate module like a `Linear` layer directly might capture its weights/bias instead of its output activations if not careful.

---

## Distributed Training

**Description:**
Train SAEs across multiple GPUs using PyTorch's distributed capabilities, primarily through `torchrun`. `sparsify` supports two main distributed modes:

1.  **Standard Distributed Data Parallel (DDP):** (Default when using `torchrun` without `--distribute_modules`)
    - **Replication:** Each GPU (process/rank) loads the **base model AND all SAEs** being trained.
    - **Data Sharding:** Input data batches are split across GPUs.
    - **Synchronization:** Gradients are averaged across GPUs after each backward pass.
    - **Use Case:** Provides speedup when the base model and *all* SAEs fit comfortably into each GPU's memory. Best for training fewer, larger SAEs or when memory per GPU is ample.

2.  **Module Distribution (`--distribute_modules`):** (Requires `torchrun` + `--distribute_modules` flag)
    - **Replication:** Each GPU loads the **base model**.
    - **SAE Sharding:** The set of SAEs being trained (defined by hookpoints) is divided among the GPUs. Each GPU only initializes and trains its assigned subset of SAEs.
    - **Synchronization:** Activations are gathered/scattered across GPUs as needed during the forward pass for SAE computation. Gradients for each SAE remain local to its assigned GPU.
    - **Use Case:** Significantly reduces memory usage per GPU for the *SAEs*, allowing training of many more SAEs simultaneously than would fit with standard DDP, especially when targeting numerous hookpoints. The base model is still replicated, so it must fit on each GPU.

**Prerequisites:**
- Multiple GPUs available on the node(s).
- PyTorch installed with CUDA and NCCL support (`torch.distributed.is_available()` and `torch.distributed.is_nccl_available()` should return `True`).
- Launching the training script using `torchrun`.

**Expected Outcome:**
- Training utilizes multiple specified GPUs.
- With DDP, faster training speed is observed compared to single GPU (assuming sufficient workload).
- With `--distribute_modules`, memory usage per GPU is lower for the SAEs compared to DDP, enabling larger-scale training across many hookpoints.

**CLI Implementation:**
Distributed training is primarily managed via the CLI by launching `python -m sparsify` with `torchrun`.

**1. Standard DDP**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train default SAEs on 2 GPUs using DDP
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 8 `# Per-GPU batch size` \
    --expansion_factor 32 \
    --k 64 \
    --save_dir ./ddp_run_out \
    --log_to_wandb False \
    --max_examples 100 # Run longer for meaningful comparison

# Clean up ./ddp_run_out afterwards
```

**2. Module Distribution**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train SAEs for many MLP layers on 2 GPUs, distributing the SAEs
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --distribute_modules `# Enable SAE sharding` \
    --hookpoints "layers.*.mlp.act_fn" `# Specify hookpoints to distribute` \
    --batch_size 4 `# Per-GPU batch size` \
    --expansion_factor 16 \
    --k 32 \
    --save_dir ./distribute_modules_run_out \
    --log_to_wandb False \
    --max_examples 100

# Clean up ./distribute_modules_run_out afterwards
```

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_full_training_pipeline.py`_
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
DATASET_SPLIT = "train[:1%]" # Use a small slice for example
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Target Hookpoint --- 
# Important: Find the correct name by inspecting the specific model architecture
# Example for Pythia-160m MLP output in layer 5
HOOKPOINT = "gpt_neox.layers.5.mlp"

# --- Load Base Model & Tokenizer ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset,
    tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    # num_proc=os.cpu_count() # Optional: for speed
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")

# --- Define Configurations ---
print("Defining SAE and Training configurations...")
# Configure the SAEs to be trained
sae_cfg = SaeConfig(
    expansion_factor=4, # Ratio of SAE latent dim to model dim (keep small for demo)
    k=32,               # Number of non-zero latents (sparsity)
)

# Configure the training process
train_cfg = TrainConfig(
    sae_cfg,            
    batch_size=8,       # Adjust based on GPU memory
    lr=1e-4,            
    save_every=1000,    
    save_dir="./sparsify_training_out", 
    hookpoints=[HOOKPOINT], # *** Crucial: Specify the target hookpoint(s) ***
    log_to_wandb=False
)
print(f"Configurations defined for hookpoint: {train_cfg.hookpoints}")

# --- Initialize Trainer ---
print("Initializing Trainer...")
# The Trainer automatically resolves input dimensions and initializes SAEs
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized successfully.")
print(f"Training SAE(s) for hookpoints: {list(trainer.saes.keys())}")

# --- Ready for Training ---
# The trainer object is now ready.
# Next step is to call trainer.fit() to start the actual training loop.
print("Setup complete. Call trainer.fit() to begin training.")

```

**Implementation Tips:**
- **Find Correct Hookpoints:** **This is the most common failure point.** Load your specific `model` and use `print(model)` or iterate through `model.named_modules()` to find the exact string name of the layer/module output you want to capture activations from. Pass this exact string(s) to `TrainConfig(hookpoints=[...])`.
- **Tokenizer Consistency:** Ensure the `tokenizer` loaded alongside the `model` matches the one used to create the `tokenized_dataset` passed to the `Trainer`. Mismatches will cause errors.
- **Dataset Formatting:** Ensure the `tokenized_dataset` is correctly formatted, typically by calling `.set_format(type='torch')` after tokenization, so it yields PyTorch tensors as expected by the `Trainer`.
- Choose a base model appropriate for your hardware resources.
- Configure `SaeConfig` and `TrainConfig` carefully, especially `hookpoints`.
- Place the base model on the appropriate device (`.to(DEVICE)`) before initializing the `Trainer`.

**Variants:**
- Use different base models (e.g., `gpt2`, Llama models if accessible).
- Train on different dataset splits.
- Adjust parameters in `SaeConfig` and `TrainConfig`.
- Specify multiple hookpoints or use wildcard patterns (e.g., `hookpoints=["gpt_neox.layers.*.mlp.act"]`) in `TrainConfig`.

**Troubleshooting:**
- **Model Loading Errors:** Ensure model identifier is correct and dependencies (`transformers`) are installed. Check hardware requirements.
- **Dataset Issues:** Verify the dataset is correctly tokenized and accessible. Ensure it's formatted correctly (e.g., `set_format(type='torch')`). Mismatched tokenizers between data prep and trainer initialization can also cause issues here.
- **Configuration Errors:** Double-check parameter names and values in `SaeConfig` and `TrainConfig`.
- **Hookpoint Errors:** If `Trainer` initialization fails with dimension errors or the optimizer fails with `empty parameter list`, the most likely cause is an incorrect hookpoint name provided in `TrainConfig.hookpoints`. Verify the name against `model.named_modules()`.
- **CUDA/Device Errors:** Ensure PyTorch is installed with CUDA support if using GPU. Verify the model and dataset tensors are on the correct device.

---

### Execute Training (`trainer.fit()`)

**Description:**
Runs the main training loop after the `Trainer` object has been initialized. This involves iterating through the dataset, computing activations from the specified hookpoint(s), calculating SAE losses, performing backpropagation, and updating SAE weights.

**Prerequisites:**
- A successfully initialized `Trainer` object (see "Programmatic Training Setup").
- Sufficient compute resources (CPU/GPU memory and processing power).

**Expected Outcome:**
- The training loop runs for the configured duration (determined by dataset size and batch size).
- Progress is logged to the console (and optionally Weights & Biases).
- Checkpoints (SAE weights `sae.safetensors` and config `cfg.json`) are saved periodically. They are organized within the `TrainConfig.save_dir` under a subdirectory named by `TrainConfig.run_name` (or `"unnamed"` if not set), and further subdivided by hookpoint name. The structure is: `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The `fit()` method completes without runtime errors.

**Tested Code Implementation:**
_This example uses the setup from the previous step (`test_full_training_pipeline.py`) and runs `fit()`._

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import shutil
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration (Match setup from previous step) ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
# Use enough data for a few steps
DATASET_SPLIT = "train[:120]" # ~15 steps with batch_size=8
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HООКРОINT = "gpt_neox.layers.5.mlp"
SAVE_DIR = "./sparsify_training_out"
SAVE_EVERY = 10 # Save frequently for demo
TARGET_BATCH_SIZE = 8

# --- Load Model & Tokenizer (As in setup) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset (As in setup) ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset, tokenizer, max_seq_len=MAX_SEQ_LEN
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")
num_steps = len(tokenized_dataset) // TARGET_BATCH_SIZE
print(f"Expected number of training steps: {num_steps}")

# --- Define Configurations (As in setup) ---
print("Defining SAE and Training configurations...")
sae_cfg = SaeConfig(expansion_factor=4, k=32)
train_cfg = TrainConfig(
    sae_cfg, 
    batch_size=TARGET_BATCH_SIZE,
    lr=1e-4, 
    save_every=SAVE_EVERY, 
    save_dir=SAVE_DIR,
    hookpoints=[HOOKPOINT], 
    log_to_wandb=False 
)
print("Configurations defined.")

# --- Initialize Trainer (As in setup) ---
print("Initializing Trainer...")
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized.")

# --- Execute Fit ---
try:
    print("Calling trainer.fit()...")
    # Training runs until dataset is exhausted
    trainer.fit()
    print("trainer.fit() completed.")

    # --- Verification ---
    print(f"Verifying checkpoint in {SAVE_DIR}...")
    # Check for base save directory
    assert os.path.isdir(SAVE_DIR), "Save directory was not created."
    # Account for potential 'unnamed' subdirectory if run_name not set
    base_save_path = os.path.join(SAVE_DIR, train_cfg.run_name or "unnamed")
    assert os.path.isdir(base_save_path), f"Base save path '{base_save_path}' not found."
    # Check for hookpoint subdirectory
    hookpoint_save_dir = os.path.join(base_save_path, HOOKPOINT)
    assert os.path.isdir(hookpoint_save_dir), f"Hookpoint subdirectory '{HOOKPOINT}' not found."
    # Check for specific files
    cfg_path = os.path.join(hookpoint_save_dir, "cfg.json")
    sae_path = os.path.join(hookpoint_save_dir, "sae.safetensors")
    assert os.path.isfile(cfg_path), "cfg.json not found."
    assert os.path.isfile(sae_path), "sae.safetensors not found."
    print("Checkpoint verification passed.")

except Exception as e:
    print(f"Error during trainer.fit(): {e}")
finally:
    # Optional: Clean up output directory after verification/use
    if os.path.exists(SAVE_DIR):
        print(f"Cleaning up directory: {SAVE_DIR}")
        shutil.rmtree(SAVE_DIR)
```

**Implementation Tips:**
- **Correct Setup is Key:** The success of `trainer.fit()` relies heavily on the correct initialization of the `Trainer` object in the preceding setup step. Ensure the base model, tokenized dataset (with matching tokenizer and correct formatting), and configurations (especially `hookpoints`) are all correctly specified.
- Ensure the `Trainer` is correctly initialized with the model, dataset, and configurations before calling `fit()`.
- Monitor resource usage (GPU memory, CPU).
- Adjust `batch_size` and `grad_acc_steps` to manage memory.
- Training duration depends on dataset size and batch size.
- Set `log_to_wandb=True` for detailed metric tracking.

**Troubleshooting:**
- **Setup Errors:** Failures during `fit()` (especially early on or related to tensor shapes/devices) can often stem from issues in the `Trainer` initialization. Double-check hookpoint names, tokenizer consistency, dataset formatting (`set_format`), and device placement from the setup step.
- **CUDA Out of Memory:** Reduce `batch_size`, use gradient accumulation (`grad_acc_steps`), use a smaller model, or use model parallelism/distribution features (see "Distributed Training").
- **Slow Training:** Increase `batch_size` (if memory allows), use GPU, use `num_proc` during data loading/tokenization, ensure efficient data loading.
- **NaN/Inf Losses:** Check learning rate (`lr`), potentially reduce it. Ensure input data is normalized correctly if needed. Check for numerical stability issues in the base model or SAE configuration.
- **Checkpointing Issues:** Verify write permissions for the `save_dir`.

---

### Command-Line Interface (CLI) Training

**Description:**
Train SAEs using the built-in command-line interface provided by `sparsify`. This offers a convenient way to run training jobs by specifying the model, dataset, and configuration options as arguments, leveraging the same underlying `Trainer` logic.

**Prerequisites:**
- The `sparsify` package installed in the environment.
- A base Hugging Face model identifier.
- Optional: A dataset identifier (defaults to `EleutherAI/fineweb-edu-dedup-10b` if omitted).
- Optional: Access to required compute resources (GPU recommended).

**Expected Outcome:**
- The training process starts, logging progress to the console.
- Checkpoints are saved to subdirectories within the default directory (`./checkpoints`) or a specified output directory (`--save_dir`), following the structure `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The command runs to completion or until manually stopped.

**Tested Code Implementation:**
_The following command was verified to execute successfully, running for a single step using a small model and dataset. Replace parameters as needed for actual training runs._

```bash
# Activate your conda environment first
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 1 \
    --expansion_factor 1 \
    --k 16 \
    --log_to_wandb False \
    --save_dir ./test_cli_run_out \
    --max_examples 1

# Expected output: Command runs, prints progress, saves checkpoint, and exits.
# Remember to clean up ./test_cli_run_out afterwards if desired.
```

**Implementation Tips:**
- Use `python -m sparsify --help` to see all available CLI options. Most arguments directly map to fields in the programmatic `SaeConfig` and `TrainConfig` (e.g., `--batch_size` maps to `TrainConfig.batch_size`, `--expansion_factor` maps to `SaeConfig.expansion_factor`).
- Specify arguments using `--arg_name value` format (e.g., `--batch_size 16`, `--expansion_factor 32`).
- Use boolean flags like `--load_in_8bit`. To explicitly set a boolean flag to `False` when its default is `True`, use the `--no-<flag_name>` syntax if available (check `--help`), otherwise omit the flag.
- For multi-GPU training, prepend the command with `torchrun --nproc_per_node <num_gpus>` (see "Distributed Training").
- By default, logs are sent to Weights & Biases if available; use `--no-log_to_wandb` to disable.

**Variants:**
- Specify a different dataset (Hub ID or local path).
- Override default `SaeConfig` parameters (e.g., `--k 128`, `--expansion_factor 64`).
- Override default `TrainConfig` parameters (e.g., `--lr 1e-4`, `--save_dir ./my_sae_run`).
- Train on specific layers (`--layers 0 5 10`) or using a stride (`--layer_stride 4`).
- Use custom hookpoints (`--hookpoints "h.*.attn" "h.*.mlp.act"`).

**Troubleshooting:**
- **Argument Errors:** Check `python -m sparsify --help` for correct argument names and types. Ensure boolean flags use the correct syntax (`--flag_name` or `--no-flag_name`).
- **Incompatible Arguments:** Some argument combinations are disallowed (e.g., using both `--layers` and `--hookpoints`). The CLI should raise an error if incompatible arguments are provided.
- **Model/Dataset Loading Errors:** Verify model and dataset identifiers are correct and accessible.
- **Resource Errors (OOM):** Reduce `--batch_size`, use `--grad_acc_steps N`, use `--load_in_8bit`, or use distributed training options if applicable.
- **`torchrun` Issues:** Ensure `torchrun` is available and configured correctly for your environment if doing multi-GPU training.

---

## Finetuning SAEs

**Description:**
Continue training SAEs from a previously saved checkpoint. This is useful for resuming interrupted training runs or further refining SAEs with different data or learning rates.

**Prerequisites:**
- A valid checkpoint directory path. This path should point to the directory containing the hookpoint subdirectories (e.g., the `{save_dir}/{run_name or "unnamed"}/` directory from a previous run). Each relevant hookpoint subdirectory must contain:
    - `sae.safetensors` (SAE weights)
    - `cfg.json` (SAE config)
    - `optimizer_*.pt` (Optimizer state for the SAE)
    - `scheduler_*.pt` (Scheduler state, if used)
    - `scaler_*.pt` (Gradient scaler state, if used)
- The **same base model** (`MODEL_ID`) used for the initial training.
- A **tokenized dataset** prepared using the **same tokenizer and formatting** as the initial training run.

**Expected Outcome:**
- The `Trainer` initializes, loading the SAE weights and optimizer state from the specified checkpoint directory.
- Training resumes from the step count saved in the checkpoint.
- The finetuning process runs, potentially saving new checkpoints to a different directory.

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_finetune.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset
import os

from sparsify import SaeConfig, Trainer, TrainConfig

# --- Configuration (Assume initial checkpoint exists) ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./test_finetune_checkpoint_source" # Path to the existing checkpoint
FINETUNE_SAVE_DIR = "./test_finetune_run_out" # New directory for finetuning output

# --- Load Model and Dataset (As used for initial training) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")

# --- Prepare Dummy Dataset (Replace with actual dataset) ---
DUMMY_SEQ_LEN = 64
DUMMY_DATASET_SIZE = 16
DUMMY_BATCH_SIZE = 4
dummy_data = {"input_ids": [[i] * DUMMY_SEQ_LEN for i in range(DUMMY_DATASET_SIZE)]}
dummy_tokenized_dataset = Dataset.from_dict(dummy_data)
dummy_tokenized_dataset.set_format(type='torch', columns=['input_ids'])
print("Dataset prepared.")

# --- Define Finetuning Configurations ---
print("Defining Finetuning configurations...")
# SAE config should generally match the checkpoint, 
# unless specific architecture changes are intended (advanced use)
sae_cfg = SaeConfig(expansion_factor=4, k=32)

# Training config specifies the checkpoint and new parameters
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=DUMMY_BATCH_SIZE,
    lr=5e-6,  # Often use a smaller learning rate for finetuning
    save_every=100,
    save_dir=FINETUNE_SAVE_DIR, # Save finetuning results separately
    log_to_wandb=False,
    finetune=CHECKPOINT_DIR  # Key parameter: points to the checkpoint directory
)
print("Configurations defined.")

# --- Initialize Finetuning Trainer ---
print("Initializing Finetune Trainer...")
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
# The Trainer automatically handles loading state if cfg.finetune is set.
print("Finetune Trainer initialized.")
# You might observe that training starts from a non-zero step

# --- Execute Finetuning Fit ---
# Ensure the checkpoint directory exists before running
if os.path.isdir(CHECKPOINT_DIR):
    try:
        print("Calling trainer.fit() for finetuning...")
        trainer.fit()
        print("Finetuning completed.")
        # Verify new checkpoints are saved in FINETUNE_SAVE_DIR
        assert os.path.isdir(FINETUNE_SAVE_DIR)
    except Exception as e:
        print(f"Error during finetuning: {e}")
else:
    print(f"Error: Checkpoint directory not found at {CHECKPOINT_DIR}")

# Remember to manage/clean up FINETUNE_SAVE_DIR
```

**Command-Line Interface (CLI) Implementation:**
Use the `--finetune <PATH_TO_CHECKPOINT_DIR>` argument when running `python -m sparsify`.

```bash
# Example: Finetune from a previous run saved in ./initial_run_out
# Assume ./initial_run_out exists and contains checkpoints

# Activate environment
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --finetune ./initial_run_out \
    --save_dir ./finetune_attempt_1 \
    --lr 5e-6 \
    --batch_size 4 \
    --log_to_wandb False \
    --max_examples 10 # Example: finetune for a few more examples

# Clean up ./finetune_attempt_1 afterwards if desired
```

**Implementation Tips:**
- **Checkpoint Path:** Ensure the path provided to `TrainConfig.finetune` (programmatic) or `--finetune` (CLI) points to the directory *containing* the hookpoint subdirectories from the previous run, not a hookpoint subdirectory itself.
- **Checkpoint Contents:** Verify the checkpoint directory contains all necessary files (`sae.safetensors`, `cfg.json`, and `.pt` files for optimizer, scheduler state) for the hookpoints being finetuned.
- **Model/Dataset Consistency:** Use the same base model (`MODEL_ID`) and a dataset prepared identically (tokenizer, `max_seq_len`, formatting) to the initial run for predictable behavior.
- **Learning Rate:** It's common practice to use a lower learning rate (`lr`) for finetuning compared to the initial training.
- **Save Directory:** Specify a new `save_dir` to avoid overwriting the original checkpoint, unless explicitly intended.

**Troubleshooting:**
- **`FileNotFoundError`:** Double-check the path provided to `finetune`. Verify the path exists and contains the expected hookpoint subdirectories with all required files (`.safetensors`, `.json`, `.pt`).
- **State Loading Errors:** Mismatches in `SaeConfig` between the checkpoint and the finetuning run might cause issues. More commonly, errors occur if optimizer/scheduler state files (`.pt`) are missing or incompatible (e.g., mismatch in optimizer type between runs).
- **Compatibility Issues:** Using a different base model architecture or significantly different dataset structure/tokenization during finetuning than the initial training is likely to cause errors or lead to poor results.

---

## Using Custom Hookpoints

**Description:**
Train SAEs on activations from specific modules within a base model, rather than relying on the default layer numbering (which usually targets residual streams). This allows targeting attention outputs, MLP intermediate activations, layer norms, etc.

**Prerequisites:**
- A base Hugging Face `transformers` model.
- Knowledge of the module names within the model structure. These are strings representing the attribute paths to access specific modules (e.g., `'layers.5.mlp.act_fn'`). You can typically find these by loading the model in Python and inspecting its structure (e.g., using `print(model)` or iterating through `model.named_modules()`).

**Expected Outcome:**
- The `Trainer` initializes SAEs only for the hookpoints matching the provided names or patterns.
- Training proceeds using activations gathered from these specific modules.

**Programmatic Implementation:**
Pass a list of hookpoint names or patterns (strings) to the `hookpoints` argument of `TrainConfig`.

_Based on `./tests/sparsify/test_custom_hookpoints.py`_
```python
import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset

from sparsify import SaeConfig, Trainer, TrainConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "cpu"

# Dummy dataset
dummy_tokenized_dataset = Dataset.from_dict({"input_ids": [[0]*32]*4})

print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
print("Model loaded.")
# print(model) # Uncomment to inspect module names

# --- Define Configurations with Custom Hookpoints ---
# Example: Target the output of the activation function in all MLP layers
# Uses fnmatch patterns (* matches anything, ? matches single char)
HOOKPOINT_PATTERNS = ["layers.*.mlp.act_fn"] 
# Other examples:
# HOOKPOINT_PATTERNS = ["layers.5.attn.q_proj", "layers.10.mlp.fc1"] # Specific modules
# HOOKPOINT_PATTERNS = ["layers.*.attn.hook_result"] # Requires model with specific hooks

sae_cfg = SaeConfig(expansion_factor=4, k=64)
train_cfg = TrainConfig(
    sae_cfg,
    batch_size=2,
    lr=1e-4,
    save_dir="./test_trainer_init_custom_hooks_out",
    # Provide the list of patterns/names here
    hookpoints=HOOKPOINT_PATTERNS 
)
print(f"Training will target hookpoints matching: {HOOKPOINT_PATTERNS}")

# --- Initialize Trainer ---
trainer = Trainer(
    cfg=train_cfg,
    dataset=dummy_tokenized_dataset,
    model=model
)
print("Trainer initialized.")
print(f"Resolved hookpoints being trained: {list(trainer.saes.keys())}")

# Now trainer.fit() would train SAEs on these specific activations.
```

**Command-Line Interface (CLI) Implementation:**
Use the `--hookpoints` argument, providing one or more space-separated names or patterns (strings).

```bash
# Activate environment
# conda activate mech_interp

# Example: Train on MLP activation outputs and Attention outputs
python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --hookpoints "layers.*.mlp.act_fn" "layers.*.attn.hook_result" \
    --batch_size 4 \
    --expansion_factor 8 \
    --k 32 \
    --save_dir ./custom_hook_cli_run \
    --log_to_wandb False \
    --max_examples 10 # Run for a few steps

# Remember to clean up ./custom_hook_cli_run afterwards
```

**Implementation Tips:**
- **Finding Hookpoints:** The easiest way to find valid module names is to load the model in Python and `print(model)`. Look for the dotted attribute paths (strings) to the modules whose output activations you want to capture (e.g., `'model.layers[0].mlp.act_fn'` would likely correspond to the hookpoint string `'layers.0.mlp.act_fn'`).
- **Pattern Matching:** Use standard `fnmatch` wildcards (`*`, `?`) in the hookpoint strings for flexibility. `*` matches any sequence of characters, `?` matches a single character. Remember to quote patterns in the CLI if they contain special shell characters.
- **Specificity:** Be as specific as needed. If you only want layer 5's MLP activation, use `'layers.5.mlp.act_fn'` instead of a pattern.
- **Conflicts:** You cannot specify both `--layers` / `--layer_stride` and `--hookpoints` (or the corresponding `TrainConfig` arguments) simultaneously. Custom hookpoints take precedence.

**Troubleshooting:**
- **No SAEs Initialized:** If the pattern(s) provided don't match any module names in the loaded model, the `Trainer` might initialize with an empty `saes` dictionary or raise an error. Double-check the patterns and module names for typos or subtle differences (e.g., `block` vs. `layer`). Verify against the actual model structure.
- **Incorrect Activations:** Ensure the hookpoint string correctly identifies the *output* of the desired module operation. Training on an intermediate module like a `Linear` layer directly might capture its weights/bias instead of its output activations if not careful.

---

## Distributed Training

**Description:**
Train SAEs across multiple GPUs using PyTorch's distributed capabilities, primarily through `torchrun`. `sparsify` supports two main distributed modes:

1.  **Standard Distributed Data Parallel (DDP):** (Default when using `torchrun` without `--distribute_modules`)
    - **Replication:** Each GPU loads the base model AND all SAEs being trained.
    - **Data Sharding:** Input data batches are split across GPUs.
    - **Synchronization:** Gradients are averaged across GPUs after each backward pass.
    - **Use Case:** Provides speedup when the base model and *all* SAEs fit comfortably into each GPU's memory. Best for training fewer, larger SAEs or when memory per GPU is ample.

2.  **Module Distribution (`--distribute_modules`):** (Requires `torchrun` + `--distribute_modules` flag)
    - **Replication:** Each GPU loads the base model.
    - **SAE Sharding:** The set of SAEs being trained (defined by hookpoints) is divided among the GPUs. Each GPU only initializes and trains its assigned subset of SAEs.
    - **Synchronization:** Activations are gathered/scattered across GPUs as needed during the forward pass for SAE computation. Gradients for each SAE remain local to its assigned GPU.
    - **Use Case:** Significantly reduces memory usage per GPU for the *SAEs*, allowing training of many more SAEs simultaneously than would fit with standard DDP, especially when targeting numerous hookpoints. The base model is still replicated, so it must fit on each GPU.

**Prerequisites:**
- Multiple GPUs available on the node(s).
- PyTorch installed with CUDA and NCCL support (`torch.distributed.is_available()` and `torch.distributed.is_nccl_available()` should return `True`).
- Launching the training script using `torchrun`.

**Expected Outcome:**
- Training utilizes multiple specified GPUs.
- With DDP, faster training speed is observed compared to single GPU (assuming sufficient workload).
- With `--distribute_modules`, memory usage per GPU is lower for the SAEs compared to DDP, enabling larger-scale training across many hookpoints.

**CLI Implementation:**
Distributed training is primarily managed via the CLI by launching `python -m sparsify` with `torchrun`.

**1. Standard DDP**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train default SAEs on 2 GPUs using DDP
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 8 `# Per-GPU batch size` \
    --expansion_factor 32 \
    --k 64 \
    --save_dir ./ddp_run_out \
    --log_to_wandb False \
    --max_examples 100 # Run longer for meaningful comparison

# Clean up ./ddp_run_out afterwards
```

**2. Module Distribution**

```bash
# Activate environment (ensure PyTorch with distributed support is installed)
# conda activate mech_interp

# Example: Train SAEs for many MLP layers on 2 GPUs, distributing the SAEs
NUM_GPUS=2
torchrun --standalone --nproc_per_node=$NUM_GPUS -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --distribute_modules `# Enable SAE sharding` \
    --hookpoints "layers.*.mlp.act_fn" `# Specify hookpoints to distribute` \
    --batch_size 4 `# Per-GPU batch size` \
    --expansion_factor 16 \
    --k 32 \
    --save_dir ./distribute_modules_run_out \
    --log_to_wandb False \
    --max_examples 100

# Clean up ./distribute_modules_run_out afterwards
```

**Programmatic Implementation:**
_Based on `./tests/sparsify/test_full_training_pipeline.py`_
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
DATASET_SPLIT = "train[:1%]" # Use a small slice for example
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Target Hookpoint --- 
# Important: Find the correct name by inspecting the specific model architecture
# Example for Pythia-160m MLP output in layer 5
HOOKPOINT = "gpt_neox.layers.5.mlp"

# --- Load Base Model & Tokenizer ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset,
    tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    # num_proc=os.cpu_count() # Optional: for speed
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")

# --- Define Configurations ---
print("Defining SAE and Training configurations...")
# Configure the SAEs to be trained
sae_cfg = SaeConfig(
    expansion_factor=4, # Ratio of SAE latent dim to model dim (keep small for demo)
    k=32,               # Number of non-zero latents (sparsity)
)

# Configure the training process
train_cfg = TrainConfig(
    sae_cfg,            
    batch_size=8,       # Adjust based on GPU memory
    lr=1e-4,            
    save_every=1000,    
    save_dir="./sparsify_training_out", 
    hookpoints=[HOOKPOINT], # *** Crucial: Specify the target hookpoint(s) ***
    log_to_wandb=False
)
print(f"Configurations defined for hookpoint: {train_cfg.hookpoints}")

# --- Initialize Trainer ---
print("Initializing Trainer...")
# The Trainer automatically resolves input dimensions and initializes SAEs
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized successfully.")
print(f"Training SAE(s) for hookpoints: {list(trainer.saes.keys())}")

# --- Ready for Training ---
# The trainer object is now ready.
# Next step is to call trainer.fit() to start the actual training loop.
print("Setup complete. Call trainer.fit() to begin training.")

```

**Implementation Tips:**
- **Find Correct Hookpoints:** **This is the most common failure point.** Load your specific `model` and use `print(model)` or iterate through `model.named_modules()` to find the exact string name of the layer/module output you want to capture activations from. Pass this exact string(s) to `TrainConfig(hookpoints=[...])`.
- **Tokenizer Consistency:** Ensure the `tokenizer` loaded alongside the `model` matches the one used to create the `tokenized_dataset` passed to the `Trainer`. Mismatches will cause errors.
- **Dataset Formatting:** Ensure the `tokenized_dataset` is correctly formatted, typically by calling `.set_format(type='torch')` after tokenization, so it yields PyTorch tensors as expected by the `Trainer`.
- Choose a base model appropriate for your hardware resources.
- Configure `SaeConfig` and `TrainConfig` carefully, especially `hookpoints`.
- Place the base model on the appropriate device (`.to(DEVICE)`) before initializing the `Trainer`.

**Variants:**
- Use different base models (e.g., `gpt2`, Llama models if accessible).
- Train on different dataset splits.
- Adjust parameters in `SaeConfig` and `TrainConfig`.
- Specify multiple hookpoints or use wildcard patterns (e.g., `hookpoints=["gpt_neox.layers.*.mlp.act"]`) in `TrainConfig`.

**Troubleshooting:**
- **Model Loading Errors:** Ensure model identifier is correct and dependencies (`transformers`) are installed. Check hardware requirements.
- **Dataset Issues:** Verify the dataset is correctly tokenized and accessible. Ensure it's formatted correctly (e.g., `set_format(type='torch')`). Mismatched tokenizers between data prep and trainer initialization can also cause issues here.
- **Configuration Errors:** Double-check parameter names and values in `SaeConfig` and `TrainConfig`.
- **Hookpoint Errors:** If `Trainer` initialization fails with dimension errors or the optimizer fails with `empty parameter list`, the most likely cause is an incorrect hookpoint name provided in `TrainConfig.hookpoints`. Verify the name against `model.named_modules()`.
- **CUDA/Device Errors:** Ensure PyTorch is installed with CUDA support if using GPU. Verify the model and dataset tensors are on the correct device.

---

### Execute Training (`trainer.fit()`)

**Description:**
Runs the main training loop after the `Trainer` object has been initialized. This involves iterating through the dataset, computing activations from the specified hookpoint(s), calculating SAE losses, performing backpropagation, and updating SAE weights.

**Prerequisites:**
- A successfully initialized `Trainer` object (see "Programmatic Training Setup").
- Sufficient compute resources (CPU/GPU memory and processing power).

**Expected Outcome:**
- The training loop runs for the configured duration (determined by dataset size and batch size).
- Progress is logged to the console (and optionally Weights & Biases).
- Checkpoints (SAE weights `sae.safetensors` and config `cfg.json`) are saved periodically. They are organized within the `TrainConfig.save_dir` under a subdirectory named by `TrainConfig.run_name` (or `"unnamed"` if not set), and further subdivided by hookpoint name. The structure is: `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The `fit()` method completes without runtime errors.

**Tested Code Implementation:**
_This example uses the setup from the previous step (`test_full_training_pipeline.py`) and runs `fit()`._

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset 
import shutil
import os

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# --- Configuration (Match setup from previous step) ---
MODEL_ID = "EleutherAI/pythia-160m-deduped" 
DATASET_ID = "roneneldan/TinyStories"
# Use enough data for a few steps
DATASET_SPLIT = "train[:120]" # ~15 steps with batch_size=8
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HООКРОINT = "gpt_neox.layers.5.mlp"
SAVE_DIR = "./sparsify_training_out"
SAVE_EVERY = 10 # Save frequently for demo
TARGET_BATCH_SIZE = 8

# --- Load Model & Tokenizer (As in setup) ---
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded to {DEVICE}.")

# --- Load & Prepare Tokenized Dataset (As in setup) ---
print(f"Loading and preparing dataset: {DATASET_ID}...")
raw_dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
tokenized_dataset = chunk_and_tokenize(
    raw_dataset, tokenizer, max_seq_len=MAX_SEQ_LEN
)
tokenized_dataset.set_format(type='torch')
print(f"Prepared dataset with {len(tokenized_dataset)} samples.")
num_steps = len(tokenized_dataset) // TARGET_BATCH_SIZE
print(f"Expected number of training steps: {num_steps}")

# --- Define Configurations (As in setup) ---
print("Defining SAE and Training configurations...")
sae_cfg = SaeConfig(expansion_factor=4, k=32)
train_cfg = TrainConfig(
    sae_cfg, 
    batch_size=TARGET_BATCH_SIZE,
    lr=1e-4, 
    save_every=SAVE_EVERY, 
    save_dir=SAVE_DIR,
    hookpoints=[HOOKPOINT], 
    log_to_wandb=False 
)
print("Configurations defined.")

# --- Initialize Trainer (As in setup) ---
print("Initializing Trainer...")
trainer = Trainer(
    cfg=train_cfg, 
    dataset=tokenized_dataset, 
    model=model
)
print("Trainer initialized.")

# --- Execute Fit ---
try:
    print("Calling trainer.fit()...")
    # Training runs until dataset is exhausted
    trainer.fit()
    print("trainer.fit() completed.")

    # --- Verification ---
    print(f"Verifying checkpoint in {SAVE_DIR}...")
    # Check for base save directory
    assert os.path.isdir(SAVE_DIR), "Save directory was not created."
    # Account for potential 'unnamed' subdirectory if run_name not set
    base_save_path = os.path.join(SAVE_DIR, train_cfg.run_name or "unnamed")
    assert os.path.isdir(base_save_path), f"Base save path '{base_save_path}' not found."
    # Check for hookpoint subdirectory
    hookpoint_save_dir = os.path.join(base_save_path, HOOKPOINT)
    assert os.path.isdir(hookpoint_save_dir), f"Hookpoint subdirectory '{HOOKPOINT}' not found."
    # Check for specific files
    cfg_path = os.path.join(hookpoint_save_dir, "cfg.json")
    sae_path = os.path.join(hookpoint_save_dir, "sae.safetensors")
    assert os.path.isfile(cfg_path), "cfg.json not found."
    assert os.path.isfile(sae_path), "sae.safetensors not found."
    print("Checkpoint verification passed.")

except Exception as e:
    print(f"Error during trainer.fit(): {e}")
finally:
    # Optional: Clean up output directory after verification/use
    if os.path.exists(SAVE_DIR):
        print(f"Cleaning up directory: {SAVE_DIR}")
        shutil.rmtree(SAVE_DIR)
```

**Implementation Tips:**
- **Correct Setup is Key:** The success of `trainer.fit()` relies heavily on the correct initialization of the `Trainer` object in the preceding setup step. Ensure the base model, tokenized dataset (with matching tokenizer and correct formatting), and configurations (especially `hookpoints`) are all correctly specified.
- Ensure the `Trainer` is correctly initialized with the model, dataset, and configurations before calling `fit()`.
- Monitor resource usage (GPU memory, CPU).
- Adjust `batch_size` and `grad_acc_steps` to manage memory.
- Training duration depends on dataset size and batch size.
- Set `log_to_wandb=True` for detailed metric tracking.

**Troubleshooting:**
- **Setup Errors:** Failures during `fit()` (especially early on or related to tensor shapes/devices) can often stem from issues in the `Trainer` initialization. Double-check hookpoint names, tokenizer consistency, dataset formatting (`set_format`), and device placement from the setup step.
- **CUDA Out of Memory:** Reduce `batch_size`, use gradient accumulation (`grad_acc_steps`), use a smaller model, or use model parallelism/distribution features (see "Distributed Training").
- **Slow Training:** Increase `batch_size` (if memory allows), use GPU, use `num_proc` during data loading/tokenization, ensure efficient data loading.
- **NaN/Inf Losses:** Check learning rate (`lr`), potentially reduce it. Ensure input data is normalized correctly if needed. Check for numerical stability issues in the base model or SAE configuration.
- **Checkpointing Issues:** Verify write permissions for the `save_dir`.

---

### Command-Line Interface (CLI) Training

**Description:**
Train SAEs using the built-in command-line interface provided by `sparsify`. This offers a convenient way to run training jobs by specifying the model, dataset, and configuration options as arguments, leveraging the same underlying `Trainer` logic.

**Prerequisites:**
- The `sparsify` package installed in the environment.
- A base Hugging Face model identifier.
- Optional: A dataset identifier (defaults to `EleutherAI/fineweb-edu-dedup-10b` if omitted).
- Optional: Access to required compute resources (GPU recommended).

**Expected Outcome:**
- The training process starts, logging progress to the console.
- Checkpoints are saved to subdirectories within the default directory (`./checkpoints`) or a specified output directory (`--save_dir`), following the structure `{save_dir}/{run_name or "unnamed"}/{hookpoint_name}/`.
- The command runs to completion or until manually stopped.

**Tested Code Implementation:**
_The following command was verified to execute successfully, running for a single step using a small model and dataset. Replace parameters as needed for actual training runs._

```bash
# Activate your conda environment first
# conda activate mech_interp

python -m sparsify \
    HuggingFaceTB/SmolLM2-135M \
    roneneldan/TinyStories \
    --batch_size 1 \
    --expansion_factor 1 \
    --k 16 \
    --log_to_wandb False \
    --save_dir ./test_cli_run_out \
    --max_examples 1

# Expected output: Command runs, prints progress, saves checkpoint, and exits.
# Remember to clean up ./test_cli_run_out afterwards if desired.
```

**Implementation Tips:**
- Use `python -m sparsify --help` to see all available CLI options. Most arguments directly map to fields in the programmatic `SaeConfig` and `TrainConfig` (e.g., `--batch_size` maps to `TrainConfig.batch_size`, `--expansion_factor` maps to `SaeConfig.expansion_factor`).
- Specify arguments using `--arg_name value` format (e.g., `--batch_size 16`, `--expansion_factor 32`).
- Use boolean flags like `--load_in_8bit`. To explicitly set a boolean flag to `False` when its default is `True`, use the `--no-<flag_name>` syntax if available (check `--help`), otherwise omit the flag.
- For multi-GPU training, prepend the command with `torchrun --nproc_per_node <num_gpus>` (see "Distributed Training").
- By default, logs are sent to Weights & Biases if available; use `--no-log_to_wandb` to disable.

**Variants:**
- Specify a different dataset (Hub ID or local path).
- Override default `SaeConfig` parameters (e.g., `--k 128`, `--expansion_factor 64`).
- Override default `TrainConfig` parameters (e.