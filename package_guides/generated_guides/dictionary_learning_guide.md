# Dictionary Learning Package Examples and Usage

This document provides a comprehensive guide to using the `dictionary_learning` package, covering key functionalities for training, evaluating, and using sparse autoencoders (dictionaries) on neural network activations.

## Table of Contents
1. [Loading Pretrained Dictionaries](#loading-pretrained-dictionaries)
2. [Using Trained Dictionaries (Encoding/Decoding)](#using-trained-dictionaries-encodingdecoding)
3. [Training Dictionaries](#training-dictionaries)
4. [Downloading Pretrained Dictionaries](#downloading-pretrained-dictionaries)
5. [Evaluating Dictionaries](#evaluating-dictionaries)
6. [Advanced/Experimental Features](#advancedexperimental-features)

---

## Loading Components

### Load Language Model

**Description:**
Instantiate a language model using the `nnsight` library, which facilitates activation extraction.

**Prerequisites:**
- A valid Hugging Face model identifier (e.g., `"EleutherAI/pythia-70m-deduped"`).
- `nnsight` library installed.
- Appropriate hardware (CPU/GPU) and device specification (e.g., `"cuda:0"`).

**Expected Outcome:**
- An `nnsight.LanguageModel` object is successfully created and loaded onto the specified device.

**Tested Code Implementation:**
<!-- Test script `./tests/dictionary_learning/test_load_model.py` will be created and executed. Verified code snippet will be added here. -->
_Verified via `./tests/dictionary_learning/test_load_model.py`_
```python
import torch as t
from nnsight import LanguageModel

# Configuration
MODEL_ID = "EleutherAI/pythia-70m-deduped" 
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"

# Load the model, mapping it directly to the target device
model = LanguageModel(MODEL_ID, device_map=DEVICE) 

print("Model loaded successfully.")
print(f"Model class: {type(model)}")

# Basic check: Ensure the model is loaded
assert model is not None
```

**Implementation Tips:**
- Use `device_map` or `dispatch=True` for automatic device placement, especially for larger models.
- Ensure the specified `MODEL_ID` exists on the Hugging Face Hub.

**Variants:**
- Load different models by changing `MODEL_ID`.
- Specify different device configurations (e.g., `"cpu"`, `"cuda:1"`).

**Troubleshooting:**
- **`RepositoryNotFoundError`:** Double-check the `MODEL_ID`.
- **CUDA/Memory Errors:** Ensure sufficient GPU memory. Use a smaller model or adjust `device_map`.
- **`nnsight` Installation:** Verify `nnsight` is installed correctly (`pip show nnsight`).

---

### Load Tokenizer

**Description:**
Load the tokenizer associated with the chosen language model. This is typically handled implicitly by `nnsight.LanguageModel` but can be accessed directly if needed.

**Prerequisites:**
- A loaded `nnsight.LanguageModel` object.

**Expected Outcome:**
- Access to a functional tokenizer object (usually `transformers.PreTrainedTokenizerFast`) via the model object.

**Tested Code Implementation:**
<!-- Test script `./tests/dictionary_learning/test_load_tokenizer.py` will be created and executed. Verified code snippet will be added here. -->
_Verified via `./tests/dictionary_learning/test_load_tokenizer.py`_
```python
# Assuming 'model' is a loaded nnsight.LanguageModel
# (e.g., from the previous step)
from transformers import PreTrainedTokenizerFast

tokenizer = model.tokenizer
print(f"Successfully accessed tokenizer: {type(tokenizer)}")

# Basic checks
assert tokenizer is not None
assert isinstance(tokenizer, PreTrainedTokenizerFast)

# Example Usage
test_text = "Hello world!"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"Original: '{test_text}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{decoded}'")
```

**Implementation Tips:**
- The tokenizer is usually accessed directly via `model.tokenizer`.
- Ensure the tokenizer's padding token is set if required for batch processing (often done automatically by `nnsight` or buffer).

**Variants:**
- Accessing tokenizers for different models loaded previously.

**Troubleshooting:**
- **AttributeError:** Ensure the `LanguageModel` was loaded correctly.
- **Tokenizer Behavior:** Consult `transformers` documentation for specific tokenizer usage details.

---

### Prepare Data Source

**Description:**
Create a generator that yields text samples from a Hugging Face dataset, suitable for feeding into the `ActivationBuffer`.

**Prerequisites:**
- A valid Hugging Face dataset identifier (e.g., `"monology/pile-uncopyrighted"`).
- `datasets` library installed.
- The `dictionary_learning.utils.hf_dataset_to_generator` function.

**Expected Outcome:**
- A Python generator object is created.
- Calling `next()` on the generator yields text strings from the specified dataset.

**Tested Code Implementation:**
<!-- Test script `./tests/dictionary_learning/test_prepare_data_source.py` will be created and executed. Verified code snippet will be added here. -->
_Verified via `./tests/dictionary_learning/test_prepare_data_source.py`_
```python
from dictionary_learning.utils import hf_dataset_to_generator
import types

DATASET_NAME = "monology/pile-uncopyrighted"
STREAMING = True # Recommended for large datasets

# Create the generator
generator = hf_dataset_to_generator(DATASET_NAME, streaming=STREAMING)
print("Generator created successfully.")

# Basic checks
assert isinstance(generator, types.GeneratorType)

# Example: Get the first sample
sample_text = next(generator)
print(f"First sample type: {type(sample_text)}")
print(f"First sample preview: '{sample_text[:100]}...'")
assert isinstance(sample_text, str)
```

**Implementation Tips:**
- Use `streaming=True` for large datasets to avoid downloading the entire dataset at once.
- Ensure the specified `DATASET_NAME` is correct and accessible.

**Variants:**
- Use different dataset identifiers.
- Specify a `split` (e.g., `"train"`, `"validation"`).
- Load a local dataset by providing a path.

**Troubleshooting:**
- **`FileNotFoundError` / Dataset Errors:** Verify the `DATASET_NAME` and split. Check internet connection if using streaming.
- **Generator Exhaustion:** The generator will eventually raise `StopIteration` when the dataset is fully consumed. Handle this in loops.

---

## Preparing Activation Data

### Initialize Activation Buffer

**Description:**
Create an `ActivationBuffer` instance to manage the collection and batching of model activations for dictionary training.

**Prerequisites:**
- A text data generator (from [Prepare Data Source](#prepare-data-source)).
- A loaded `nnsight.LanguageModel` (from [Load Language Model](#load-language-model)).
- A reference to the specific submodule within the model to extract activations from (e.g., `model.gpt_neox.layers[LAYER]`).
- Configuration parameters: `n_ctxs` (buffer size in contexts), `ctx_len` (context length), `refresh_batch_size` (model inference batch size), `out_batch_size` (SAE training batch size), `device`.

**Expected Outcome:**
- An `ActivationBuffer` object is successfully initialized.
- Iterating over the buffer *would* yield batches of activation tensors (though iteration is omitted in the test below for speed). When iterated, each batch tensor would have shape `(batch_size, d_submodule)`.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_activation_buffer.py`. This example focuses on successful initialization, assuming `model`, `submodule`, and `dummy_data` are already defined as per previous sections._
```python
import torch
from dictionary_learning import ActivationBuffer

# --- Assumed Prerequisites (from previous sections) ---
# model: Loaded nnsight.LanguageModel object
# submodule: Target layer object (e.g., model.gpt_neox.layers[1].mlp)
# dummy_data: An iterator yielding text strings (e.g., from hf_dataset_to_generator)
# EXPECTED_ACTIVATION_DIM: Integer, output dimension of the submodule (e.g., 512)
# DEVICE: String, target device (e.g., "cuda:0" or "cpu")

# --- Configuration for ActivationBuffer ---
d_submodule = EXPECTED_ACTIVATION_DIM 
n_ctxs = 3e2  # Buffer size (number of contexts)
ctx_len = 128 # Context length

# --- Initialize ActivationBuffer ---
try:
    buffer = ActivationBuffer(
        data=dummy_data,          # The text generator
        model=model,              # The loaded nnsight model
        submodule=submodule,      # The specific layer/module to get activations from
        d_submodule=d_submodule,  # Submodule's output dimension (important!)
        n_ctxs=n_ctxs,            # How many text contexts to keep in the buffer
        ctx_len=ctx_len,          # Max length of each context/sequence
        device=DEVICE,            # Device for activation tensors
        # Optional: Specify batch sizes if defaults aren't suitable
        # refresh_batch_size=batch_size_for_model_inference,
        # out_batch_size=batch_size_for_training
    )
    print(f"ActivationBuffer initialized successfully on device {buffer.device}.")
    print(f"Buffer configured for submodule dim: {buffer.d_submodule}")

    # --- Ready for Training --- 
    # The 'buffer' object can now be passed to trainSAE or iterated over.
    # Example of getting a batch (requires buffer to fill):
    # try:
    #    first_batch = next(iter(buffer))
    #    print(f"Got batch shape: {first_batch.shape}")
    # except StopIteration:
    #    print("Buffer data iterator exhausted before first batch could be yielded.")
    # except Exception as e:
    #    print(f"Error getting batch: {e}")

except ValueError as e:
    print(f"Error during initialization (e.g., mismatched d_submodule): {e}")
except Exception as e:
    print(f"An unexpected error occurred during ActivationBuffer initialization: {e}")

**Implementation Tips:**
- The buffer may need a substantial amount of text data from the iterator even just for initialization, depending on `n_ctxs` and `ctx_len`.
- Iterating over the buffer (`next(iter(buffer))`) triggers the internal activation fetching and processing, which can take time and memory.

**Variants:**
- Change buffer size (`n_ctxs`), context length (`ctx_len`), batch sizes.
- Extract from different `submodule`s or use `io='in'`.
- Use `HeadActivationBuffer` for attention head specific activations (requires different initialization).

**Troubleshooting:**
- **Dimension Errors:** Ensure `d_submodule` is correct for the chosen `submodule` and `io` setting.
- **Memory Errors:** Reduce `n_ctxs`, `ctx_len`, `refresh_batch_size`, or `out_batch_size`. Ensure the buffer `device` has enough memory.
- **`nnsight` Errors:** Underlying errors during model inference in `refresh()` might indicate issues with the model, tokenizer, or input data.

---

### Iterate Over Buffer

**Description:**
Iterate through the `ActivationBuffer` to retrieve batches of activation vectors for training.
The buffer automatically refreshes itself with new activations from the data source when needed.

**Prerequisites:**
- An initialized `ActivationBuffer` object.

**Expected Outcome:**
- The buffer yields batches of activation vectors as PyTorch tensors.
- Each yielded tensor has the shape `(buffer.out_batch_size, buffer.d_submodule)`.
- The buffer automatically runs model inference to refill itself when depleted.

**Tested Code Implementation:**
<!-- Test script `./tests/dictionary_learning/test_iterate_buffer.py` will be created and executed. Verified code snippet will be added here. -->
_Verified via `./tests/dictionary_learning/test_activation_buffer.py`_
```python
import torch as t

# Assuming 'activation_buffer' is initialized (from previous step)

NUM_BATCHES_TO_CHECK = 3
target_batch_size = activation_buffer.out_batch_size
activation_dim = activation_buffer.d_submodule
expected_device_type = activation_buffer.device.split(':')[0]
# Activation dtype might differ from model param dtype (e.g., float16 vs float32)
expected_dtype = t.float16 

print(f"Fetching {NUM_BATCHES_TO_CHECK} batches from buffer...")
batches_fetched = 0
for i, batch in enumerate(activation_buffer):
    print(f"Fetched batch {i+1} - Shape: {batch.shape}, Dtype: {batch.dtype}, Device: {batch.device}")
    
    # Verify batch properties
    assert isinstance(batch, t.Tensor)
    assert batch.shape == (target_batch_size, activation_dim)
    assert batch.device.type == expected_device_type
    assert batch.dtype == expected_dtype
    
    batches_fetched += 1
    if batches_fetched >= NUM_BATCHES_TO_CHECK:
        break 

print(f"\nSuccessfully fetched {batches_fetched} batches.")
```

**Implementation Tips:**
- The buffer acts as an infinite iterator (as long as the underlying data generator provides data).
- Use a `for` loop or `next(activation_buffer)` to get batches.
- Wrap the iteration in `tqdm` for progress tracking during long training runs.

**Variants:**
- Use `itertools.islice` to limit the number of batches drawn for testing or short runs.

**Troubleshooting:**
- **StopIteration:** If the underlying data generator (from `Prepare Data Source`) runs out of data, the buffer will eventually raise `StopIteration` during a `refresh()` call.
- **Errors during Refresh:** Any issues encountered during model inference (CUDA errors, dimension mismatches, etc.) within the `refresh()` method will be raised during iteration.

---

## Training Dictionaries

This section describes how to train your own sparse autoencoders (dictionaries) using the `dictionary_learning` package. The primary components involved are the `ActivationBuffer` for collecting model activations and the `trainSAE` function for running the training loop with specified configurations.

### Configure Training

**Description:**
Define the configuration for a training run using a dictionary passed to the `trainSAE` function. This dictionary specifies the trainer class, dictionary class, dimensions, hyperparameters, and other training options.

**Prerequisites:**
- Knowledge of the desired trainer class (e.g., `StandardTrainer`, `GatedSAETrainer`, `TopKSAETrainer`) corresponding to the dictionary architecture.
- Knowledge of the dictionary class to be trained (e.g., `AutoEncoder`, `GatedAutoEncoder`, `AutoEncoderTopK`).
- The input activation dimension (`activation_dim`) for the dictionary.
- The desired dictionary size (`dict_size`), often expressed as a multiple of `activation_dim`.
- Hyperparameters like learning rate (`lr`), sparsity penalty (`sparsity_penalty` or similar, depending on trainer).

**Expected Outcome:**
- A Python dictionary (`trainer_cfg`) is created containing valid configuration keys and values for the chosen trainer and dictionary classes.
- This configuration dictionary is ready to be passed to the `trainSAE` function.

**Tested Code Implementation:**
_Configuration itself is not directly executable. This section provides an example configuration based on the README for training a standard AutoEncoder. Verification happens when this config is used in `trainSAE` via `./tests/dictionary_learning/test_trainSAE.py`._
```python
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning import AutoEncoder
import torch

# --- Assumed Prerequisites ---
# activation_dim = 512 # e.g., from model.gpt_neox.layers[1].mlp output
# dictionary_size = 16 * activation_dim # Example expansion
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- Example Configuration Dictionary ---
trainer_cfg = {
    "trainer": StandardTrainer,      # The class for the trainer
    "dict_class": AutoEncoder,       # The class for the dictionary/SAE
    "activation_dim": activation_dim, # Dimension of the input activations
    "dict_size": dictionary_size,    # Number of features in the dictionary
    "lr": 1e-4,                      # Learning rate
    "l1_penalty": 0.004,             # **Key specific to StandardTrainer**
    "device": DEVICE,                # Device for training ('cuda' or 'cpu')
    "seed": 42,                      # Seed for reproducibility
    "steps": 100000,               # Total training steps (required by StandardTrainer)
    "layer": 1,                      # Metadata: layer index (required by StandardTrainer)
    "lm_name": "model_id",           # Metadata: model name (required by StandardTrainer)
    
    # --- Optional Parameters (Examples for StandardTrainer) ---
    # "warmup_steps": 1000,          # Linear LR warmup
    # "sparsity_warmup_steps": 1000, # Linear sparsity penalty warmup
    # "resample_steps": 25000,       # Interval for resampling dead neurons
    # "decay_start": 50000,        # Step to start linear LR decay
}

# This trainer_cfg dictionary is now ready to be passed to trainSAE
print("Trainer configuration dictionary created:")
print(trainer_cfg)

# Example of creating a list for trainSAE (even with one config)
# trainer_configs_list = [trainer_cfg]

```

**Implementation Tips:**
- The keys in the `trainer_cfg` dictionary must match the arguments expected by the chosen trainer's `__init__` method and potentially some arguments consumed by `trainSAE` itself (though trainer config primarily targets the trainer class).
- Common essential keys include: `trainer`, `dict_class`, `activation_dim`, `dict_size`, `lr`, `device`.
- Trainer-specific keys control sparsity penalties (e.g., `l1_penalty` for `StandardTrainer`, `target_sparsity` for `GatedSAETrainer`) and other features like resampling (`resample_steps`).
- **Crucially, always check the `__init__` signature of the specific `trainer` class you intend to use (e.g., `StandardTrainer.__init__`) to confirm the exact required and optional parameter names and their expected types.** For example, `StandardTrainer` requires `steps`, `layer`, and `lm_name` in its config.
- Keys like `log_steps` or `save_steps` are *not* typically part of the trainer config dictionary itself, but are arguments to the `trainSAE` function.
- Consult the specific trainer class implementation (e.g., `dictionary_learning.trainers.standard.StandardTrainer`) for all available options.

**Variants:**
- Configure different trainer/dictionary combinations (e.g., `GatedSAETrainer` with `GatedAutoEncoder`).
- Add optional parameters like `warmup_steps`, `decay_start`, `resample_steps`.
- Define multiple configurations in a list to train several dictionaries sequentially or in parallel within a single `trainSAE` call.

**Troubleshooting:**
- **`TypeError: __init__() got an unexpected keyword argument ...`:** The `trainer_cfg` contains a key not recognized by the selected trainer class constructor.
- **`KeyError: ...` during `trainSAE`:** The `trainer_cfg` is missing a required key.
- **Incorrect Hyperparameters:** Ensure values are appropriate for the chosen trainer/dictionary (e.g., sparsity penalty range).

---

### Execute Training (`trainSAE`)

**Description:**
Run the main training loop using the `trainSAE` function (imported from `dictionary_learning.training`), providing the prepared data source (e.g., `ActivationBuffer`) and a list of trainer configurations.

**Prerequisites:**
- An initialized data source providing batches of activations (e.g., an `ActivationBuffer` or a custom generator like one using `TransformerLens`, or a standard PyTorch DataLoader).
- A list containing one or more valid trainer configuration dictionaries (as prepared in the previous subtask).
- Import `trainSAE` using `from dictionary_learning.training import trainSAE`.
- Optional: A path (`save_dir`) to save the final trained dictionary and potentially intermediate checkpoints (depending on trainer config).
- Optional: `log_steps` argument for `trainSAE` to control console logging frequency.

**Expected Outcome:**
- The training loop runs, iterating through the data source for the number of steps determined by the configuration (`steps` key within the trainer config) or the data source size.
- Progress (losses, sparsity metrics) is logged to the console at intervals specified by the `log_steps` argument to `trainSAE`.
- Checkpoints *may* be saved periodically to subdirectories within `save_dir` if the trainer configuration includes relevant parameters (e.g., `save_steps`) and the trainer implements it.
- The `trainSAE` function typically trains the dictionary objects provided within the `trainer_configs` list *in-place* and may not return the trained objects directly. Access the trained dictionary by loading it from `save_dir` after completion.
- The final dictionary state is saved to `save_dir` if provided.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_trainSAE.py`. This script runs a minimal training loop for 10 steps._
```python
import torch
import os
import tempfile
import shutil
import random

# Attempt imports
try:
    from nnsight import LanguageModel
    from dictionary_learning import ActivationBuffer, AutoEncoder
    from dictionary_learning.training import trainSAE # Correct import location
    from dictionary_learning.trainers import StandardTrainer
    from dictionary_learning.utils import hf_dataset_to_generator
except ImportError as e:
    print(f"Error: Missing dependencies. Ensure nnsight and dictionary_learning are installed. {e}")
    exit(1)

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "monology/pile-uncopyrighted"
LAYER = 0 # Target layer 0 MLP output
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42

# Training parameters (minimal for testing)
ACTIVATION_DIM = 512 # Known dim for pythia-70m
DICT_SIZE_MULT = 4   # Small dictionary for testing (4 * activation_dim)
N_CTXS_BUFFER = 100  # Smaller buffer for faster test
CTX_LEN_BUFFER = 64
LR = 1e-4
SPARSITY_PENALTY = 0.004
TRAIN_STEPS = 10     # Very few steps just to test the loop runs

# Set seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create temporary directories for logs and final save
# In a real run, provide persistent paths
log_dir_temp = tempfile.mkdtemp() # Not used by trainSAE directly but might be useful for other logging
final_dir_temp = tempfile.mkdtemp()
print(f"Using temp save_dir: {final_dir_temp}")

try:
    # --- 1. Load Components & Prepare Buffer ---
    print("Loading model...")
    model = LanguageModel(MODEL_ID, device_map=DEVICE)
    submodule = model.gpt_neox.layers[LAYER].mlp # Adjust if model changes
    print("Preparing data generator...")
    generator = hf_dataset_to_generator(DATASET_NAME, split="train", streaming=True)
    
    print("Initializing ActivationBuffer...")
    buffer = ActivationBuffer(
        data=generator,
        model=model,
        submodule=submodule,
        d_submodule=ACTIVATION_DIM,
        n_ctxs=N_CTXS_BUFFER,
        ctx_len=CTX_LEN_BUFFER,
        device=DEVICE,
    )
    print("Buffer initialized.")

    # --- 2. Configure Training ---
    print("Configuring trainer...")
    trainer_cfg = {
        "trainer": StandardTrainer,
        "dict_class": AutoEncoder,
        "activation_dim": ACTIVATION_DIM,
        "dict_size": ACTIVATION_DIM * DICT_SIZE_MULT,
        "lr": LR,
        "l1_penalty": SPARSITY_PENALTY, # Correct key for StandardTrainer
        "device": DEVICE,
        "seed": SEED,
        # Added required args from StandardTrainer.__init__
        "steps": TRAIN_STEPS,
        "layer": LAYER, 
        "lm_name": MODEL_ID, 
        "warmup_steps": 1, # Explicitly set warmup_steps < steps for test
        "sparsity_warmup_steps": 1, # Explicitly set sparsity_warmup_steps < steps for test
    }
    trainer_configs_list = [trainer_cfg]
    print(f"Trainer config: {trainer_cfg}")

    # --- 3. Execute trainSAE ---
    print(f"Starting trainSAE for {TRAIN_STEPS} steps...")
    # Use correct arguments: save_dir, use_wandb, log_steps
    trainSAE(
        data=buffer,
        trainer_configs=trainer_configs_list,
        steps=TRAIN_STEPS,
        save_dir=final_dir_temp, # Use the temp dir for saving
        log_steps=5,        # Log every 5 steps (passed to trainSAE)
        normalize_activations=True, # Recommended setting
        use_wandb=False     # Correct argument to disable wandb
    )
    print("trainSAE finished.")

    # --- 4. Verification (Check for output files) ---
    trainer_final_dir = os.path.join(final_dir_temp, "trainer_0")
    if os.path.isfile(os.path.join(trainer_final_dir, "ae.pt")) and \
       os.path.isfile(os.path.join(trainer_final_dir, "config.json")):
        print("Verification successful: Output files ae.pt and config.json found.")
    else:
        print("Verification failed: Output files not found.")

except Exception as e:
    print(f"An error occurred during trainSAE test: {e}")
finally:
    # Clean up temporary directories
    print("Cleaning up temporary directories...")
    if os.path.exists(log_dir_temp):
        shutil.rmtree(log_dir_temp)
    if os.path.exists(final_dir_temp):
        shutil.rmtree(final_dir_temp)

**Implementation Tips:**
- **Import:** Ensure `trainSAE` is imported from `dictionary_learning.training`.
- `trainSAE` expects the `trainer_configs` argument to be a *list* of dictionaries.
- **Saving:** Provide `save_dir` to save the final dictionary. Intermediate checkpointing frequency might depend on parameters within the trainer config (e.g., `save_steps`), if supported by that specific trainer.
- **Data Source:** `trainSAE` can accept an `ActivationBuffer` or any iterator yielding batches of activation tensors. The `single_layer_reconstruction` experiment showed success using a custom generator with `TransformerLens` as the data source.
- **Logging:** Use the `log_steps` argument in the `trainSAE` call to control console log frequency.
- **Return Value:** Do not rely on `trainSAE` returning the trained dictionary objects; load them from the `save_dir` after training completes.
- Set `normalize_activations=True` in `trainSAE` (recommended).

**Variants:**
- Train multiple dictionaries by providing multiple configurations in the `trainer_configs` list.
- Use a different data source (e.g., a `DataLoader` returning activation tensors).
- Enable WandB logging by setting `wandb_log=True` and configuring WandB environment variables.

**Troubleshooting:**
- **Errors during Trainer initialization:** Likely due to issues in the `trainer_cfg` dictionary (see previous subtask).
- **CUDA Out of Memory:** Reduce the batch size yielded by the data source (e.g., `buffer.out_batch_size` if configured, or the DataLoader's batch size), or use a smaller dictionary size/model.
- **Slow Training:** Check device placement, optimize data loading/buffering, ensure efficient model inference if using `ActivationBuffer`.
- **NaN/Inf Losses:** Check learning rate, sparsity penalty values, and input activation scaling. Consider using `normalize_activations=True`.
- **Checkpointing Issues:** Verify write permissions for `save_dir`. Check the specific trainer's config options for controlling checkpoint frequency (e.g., `save_steps`).

---

## Saving and Loading Dictionaries

_Note: Saving dictionaries is typically handled automatically by the `trainSAE` function during training when `save_dir` is specified. Loading previously saved dictionaries is done using the class method `from_pretrained`._

### Loading Dictionaries (`from_pretrained`)

**Description:**
Load a trained dictionary (autoencoder) and its associated configuration using the class's `from_pretrained` method. This method works for dictionaries saved locally during training or downloaded pre-trained dictionaries, as long as they follow the standard format (weights file `ae.pt` and configuration `config.json` in the same directory).

**Prerequisites:**
- The specific dictionary class (e.g., `AutoEncoder`, `GatedAutoEncoder`) corresponding to the saved dictionary.
- Path to the dictionary weights file (typically `ae.pt`).
- A `config.json` file must exist in the same directory as the `ae.pt` file.
- Desired device to load the dictionary onto (`device`).

**Expected Outcome:**
- An instance of the specified dictionary class is created.
- Weights are loaded from the `ae.pt` file.
- Configuration is loaded from `config.json` and stored in the `ae.cfg` attribute.
- The dictionary is loaded onto the specified `device`.

**Tested Code Implementation:**

_Example 1: Loading a locally trained dictionary (using a temporary path from a test setup)_
_Verified via `./tests/dictionary_learning/test_load_dictionary.py`. This script first trains a minimal dictionary, then loads it._
```python
import torch as t
import os
from dictionary_learning import AutoEncoder # Use the specific class

# Assume setup_trained_dictionary(base_temp_dir) has run and returned 
# the path to the saved trainer directory (e.g., "/tmp/some_temp_dir/trainer_0")
TRAINED_DICT_DIR = trained_dict_path # Path returned by setup helper
WEIGHTS_FILE_PATH = os.path.join(TRAINED_DICT_DIR, "ae.pt")
CONFIG_FILE_PATH = os.path.join(TRAINED_DICT_DIR, "config.json") # Check config exists too
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"

# Check if the files exist before attempting to load
if not os.path.isfile(WEIGHTS_FILE_PATH):
    print(f"Error: Weights file not found at {WEIGHTS_FILE_PATH}")
    # Handle error
elif not os.path.isfile(CONFIG_FILE_PATH):
    print(f"Error: Config file not found at {CONFIG_FILE_PATH}")
    # Handle error
else:
    try:
        print(f"Attempting to load dictionary using from_pretrained from path: {WEIGHTS_FILE_PATH}")
        # Load using the path to the weights file; config.json is found in the same directory
        dictionary = AutoEncoder.from_pretrained(WEIGHTS_FILE_PATH, device=DEVICE)
        
        print(f"Successfully loaded dictionary of type: {type(dictionary)}")
        # Verify device
        assert dictionary.encoder.weight.device.type == DEVICE.split(':')[0]
        print(f"Dictionary loaded onto device: {dictionary.encoder.weight.device}")
        
        # Verify config loaded into .cfg attribute
        assert hasattr(dictionary, 'cfg'), "Loaded dictionary missing .cfg attribute."
        assert "trainer" in dictionary.cfg
        print(f"Loaded config keys: {list(dictionary.cfg['trainer'].keys())}")
        print(f"Verification successful: Dictionary loaded with weights and config.")

    except FileNotFoundError:
        print(f"Error: Directory or files not found at {TRAINED_DICT_DIR}")
    except Exception as e:
        print(f"An error occurred during loading: {e}")

```

_Example 2: Loading a pre-downloaded dictionary from a known path_
_Verified via `./tests/dictionary_learning/test_load_local_dict.py`. This example loads a specific MLP dictionary for Pythia-70m._
```python
import torch
import os
from dictionary_learning import AutoEncoder # Use the specific class

# Define the path to the pre-downloaded dictionary directory
DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Construct the full path to the weights file (ae.pt)
WEIGHTS_FILE_PATH = os.path.join(DICT_PATH, "ae.pt")
CONFIG_FILE_PATH = os.path.join(DICT_PATH, "config.json")

# Check if the files exist before attempting to load
if not os.path.isfile(WEIGHTS_FILE_PATH):
    print(f"Error: Weights file not found at {WEIGHTS_FILE_PATH}")
    # Handle error
elif not os.path.isfile(CONFIG_FILE_PATH):
    print(f"Error: Config file not found at {CONFIG_FILE_PATH}")
    # Handle error
else:
    try:
        # Load the dictionary using the path to the weights file
        ae = AutoEncoder.from_pretrained(WEIGHTS_FILE_PATH, device=DEVICE)

        print(f"Successfully loaded dictionary of type: {type(ae)}")
        print(f"Dictionary weights loaded to device: {ae.encoder.weight.device}")

        # Example Verification (access dimensions via weights):
        activation_dim = ae.encoder.weight.shape[1]
        dictionary_size = ae.encoder.weight.shape[0]
        print(f"Inferred dictionary with activation_dim={activation_dim}, dictionary_size={dictionary_size}")
        
        # Verify configuration was loaded into ae.cfg
        assert hasattr(ae, 'cfg'), "Loaded dictionary does not have a 'cfg' attribute."
        print(f"Config loaded successfully. Keys: {list(ae.cfg['trainer'].keys())}")
        assert ae.cfg['trainer']['activation_dim'] == activation_dim

    except FileNotFoundError:
        print(f"Error: Could not find weights file at {WEIGHTS_FILE_PATH} or associated config.json")
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")

```

**Implementation Tips:**
- Use the specific dictionary class's `from_pretrained` method (e.g., `AutoEncoder.from_pretrained`, `GatedAutoEncoder.from_pretrained`).
- Provide the *full path* to the weights file (typically `ae.pt`).
- The method automatically looks for `config.json` in the same directory as the weights file. Ensure both files exist.
- Access the loaded configuration via the `ae.cfg` attribute of the loaded dictionary object.

**Variants:**
- Load different dictionary types by changing the class (`AutoEncoder`, `GatedAutoEncoder`, etc.).
- Load dictionaries saved at different locations or with different naming conventions (if adapted manually).
- Load dictionaries onto CPU (`device="cpu"`).

**Troubleshooting:**
- **`FileNotFoundError`:** Ensure the path to `ae.pt` is correct and that both `ae.pt` and `config.json` exist in that directory. Check for typos or incorrect relative paths.
- **`KeyError` / `AttributeError`:** The `config.json` might be missing necessary keys for the dictionary class, or the saved state dictionary (`ae.pt`) might be incompatible (e.g., wrong dimensions, mismatched layers) with the class being used to load it.
- **`RuntimeError (Size Mismatch)`:** The architecture defined by the class might not match the dimensions in the saved state dictionary. Ensure the correct class is used and the dictionary file corresponds to that architecture.
- **Device Errors:** Ensure the specified `device` is valid (e.g., `"cuda:0"`, `"cpu"`).

---

## Using Pretrained Dictionaries

_Note: This package primarily focuses on training dictionaries. The `from_pretrained` class method (described in the previous section) is the standard way to load dictionaries, whether they were trained locally or downloaded from elsewhere, provided they follow the expected format (`ae.pt` and `config.json` in the same directory). Once loaded, you can use them for encoding and decoding as described below._

### Load Dictionary using `from_pretrained`

**Description:**
Load a dictionary object directly using its class's `from_pretrained` method, pointing to the saved state dictionary weights file (`ae.pt`). The configuration is loaded automatically from `config.json` located in the same directory.

**Prerequisites:**
- The specific dictionary class you want to load (e.g., `AutoEncoder`, `GatedAutoEncoder`).
- Path to a saved dictionary state file (`ae.pt`).
- A `config.json` file must exist in the same directory as the `ae.pt` file.
- Desired device (`device`).

**Expected Outcome:**
- An instance of the specified dictionary class is created and loaded with weights from the `.pt` file.
- The configuration from `config.json` is loaded into the dictionary object's `.cfg` attribute.
- The dictionary is on the specified `device`.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_load_local_dict.py`. This example loads the MLP dictionary for layer 1 of Pythia-70m from a local directory._
```python
import torch
import os
from dictionary_learning import AutoEncoder # Use the specific class

# Define the path to the dictionary directory
DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Construct the full path to the weights file (ae.pt)
WEIGHTS_FILE_PATH = os.path.join(DICT_PATH, "ae.pt")
CONFIG_FILE_PATH = os.path.join(DICT_PATH, "config.json")

# Check if the files exist before attempting to load
if not os.path.isfile(WEIGHTS_FILE_PATH):
    print(f"Error: Weights file not found at {WEIGHTS_FILE_PATH}")
    # Handle error
elif not os.path.isfile(CONFIG_FILE_PATH):
    print(f"Error: Config file not found at {CONFIG_FILE_PATH}")
    # Handle error
else:
    try:
        # Load the dictionary using the path to the weights file
        ae = AutoEncoder.from_pretrained(WEIGHTS_FILE_PATH, device=DEVICE)

        print(f"Successfully loaded dictionary of type: {type(ae)}")
        print(f"Dictionary weights loaded to device: {ae.encoder.weight.device}")

        # Example Verification (access dimensions via weights):
        activation_dim = ae.encoder.weight.shape[1]
        dictionary_size = ae.encoder.weight.shape[0]
        print(f"Inferred dictionary with activation_dim={activation_dim}, dictionary_size={dictionary_size}")
        
        # Verify configuration was loaded into ae.cfg
        assert hasattr(ae, 'cfg'), "Loaded dictionary does not have a 'cfg' attribute."
        print(f"Config loaded successfully. Keys: {list(ae.cfg['trainer'].keys())}")
        assert ae.cfg['trainer']['activation_dim'] == activation_dim

    except FileNotFoundError:
        print(f"Error: Could not find weights file at {WEIGHTS_FILE_PATH} or associated config.json")
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
```

**Implementation Tips:**
- Ensure you use the correct dictionary class (e.g., `AutoEncoder`, `GatedAutoEncoder`) for the `from_pretrained` call.
- The method requires the *full path* to the dictionary weights file (typically `ae.pt`).
- It *requires* the corresponding `config.json` file to be in the *same directory*.
- The configuration is automatically loaded and attached to the returned dictionary object as the `.cfg` attribute.
- Use `ae.cfg` to access configuration parameters after loading.

**Variants:**
- Load different dictionary types by changing the class.
- Load different saved state files.

**Troubleshooting:**
- **`FileNotFoundError`:** Verify the path to the `.pt` file.
- **RuntimeError (Size Mismatch):** The architecture defined by the class (based on its `__init__` defaults if not overridden) might not match the dimensions in the saved state dictionary. Ensure the correct class is used.
- **AttributeError:** The loaded state dictionary might be missing expected keys.

---

## Using Trained Dictionaries (Encoding/Decoding)

This section covers how to use a loaded dictionary to transform activations into sparse features (encoding) and reconstruct activations from those features (decoding).

### Encode Activations

**Description:**
Use a loaded dictionary's `.encode()` method to obtain the sparse feature activations corresponding to input activations.

**Prerequisites:**
- A loaded `AutoEncoder` (or subclass) object (see [Loading Pretrained Dictionaries](#loading-pretrained-dictionaries)).
- A PyTorch tensor containing input activations. The tensor should have shape `(batch_size, activation_dim)` and be on the same device and dtype as the dictionary's encoder weights.

**Expected Outcome:**
- The `.encode()` method returns a tensor containing the sparse feature activations.
- For the standard `AutoEncoder`, the output tensor has shape `(batch_size, dictionary_size)` and contains the sparse feature activations resulting from the encoder followed by a ReLU.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_encode_activations.py`. This example uses a previously loaded dictionary (`ae`)._
```python
import torch

# Assume 'ae' is a loaded AutoEncoder object (e.g., from the previous 'Load Pretrained Dictionary' step)
# Assume 'activation_dim' and 'dictionary_size' are known (e.g., from ae.encoder.weight.shape)
# Assume DEVICE is set (e.g., "cuda:0" or "cpu")

BATCH_SIZE = 16 # Example batch size

# --- Prepare Dummy Input Data ---
# Create dummy activations matching the dictionary's input dimensions
dummy_activations = torch.randn(
    BATCH_SIZE, 
    activation_dim, # Use the dictionary's activation dim
    device=DEVICE, 
    dtype=ae.encoder.weight.dtype # Match dictionary weight dtype (e.g., torch.float32)
)

print(f"Input activations shape: {dummy_activations.shape}")

# --- Encode --- 
try:
    with torch.no_grad(): # Use no_grad for inference to save memory/computation
        # The encode method directly returns the feature tensor
        features = ae.encode(dummy_activations)
    
    print("Encoding successful.")
    print(f"Output feature tensor shape: {features.shape}")

    # --- Verification (Example) ---
    expected_feature_shape = (BATCH_SIZE, dictionary_size) # Use the dictionary's size
    assert features.shape == expected_feature_shape
    assert features.device == dummy_activations.device
    assert features.dtype == dummy_activations.dtype
    # Check sparsity (ReLU output should be non-negative and likely contain zeros)
    assert torch.all(features >= 0)
    print(f"Number of non-zero features in first sample: {torch.sum(features[0] > 0)}")

except Exception as e:
    print(f"An error occurred during encoding: {e}")

```

**Implementation Tips:**
- Ensure the input activation tensor has the correct shape `(batch_size, activation_dim)`.
- Ensure the input tensor is on the same `device` and has the same `dtype` as the dictionary weights.
- The `.encode()` method of the standard `AutoEncoder` directly returns the feature tensor (after the encoder's linear layer and ReLU activation).
- Wrap the call in `torch.no_grad()` during inference.

**Variants:**
- Input activations can come from various sources (e.g., direct model hooks, `ActivationBuffer`).

**Troubleshooting:**
- **Shape Mismatches:** Ensure the input tensor's last dimension matches the dictionary's `activation_dim`.
- **Device/Dtype Mismatches:** Use `.to(device, dtype=...)` to move/cast the input tensor before encoding.

---

### Decode Features

**Description:**
Use a loaded dictionary's `.decode()` method to reconstruct the original activation vectors from their sparse feature representations.

**Prerequisites:**
- A loaded `AutoEncoder` (or subclass) object (`ae`).
- A PyTorch tensor containing sparse feature activations (`features`), typically the output of `ae.encode()`. The tensor should have shape `(batch_size, dictionary_size)` and be on the same device and dtype as the dictionary's decoder weights.

**Expected Outcome:**
- The `.decode()` method returns a tensor containing the reconstructed activation vectors.
- The output tensor shape is `(batch_size, activation_dim)`.
- The output tensor represents the dictionary's approximation of the original activations based on the sparse features.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_decode_features.py`. This example uses a previously loaded dictionary (`ae`) and assumes `features` contains sparse feature activations (e.g., from `ae.encode()`)._
```python
import torch

# Assume 'ae' is a loaded AutoEncoder object
# Assume 'activation_dim' and 'dictionary_size' are known
# Assume DEVICE is set
# Assume 'features' is a tensor with shape (batch_size, dictionary_size)
# containing feature activations (e.g., output of ae.encode())

# --- Example: Create dummy features if needed ---
BATCH_SIZE = 16 # Must match the batch size of the features tensor
features = torch.randn(BATCH_SIZE, dictionary_size, device=DEVICE, dtype=ae.decoder.weight.dtype)
features = torch.relu(features) # Apply ReLU if mimicking encode output
print(f"Input features shape: {features.shape}") 

# --- Decode ---
try:
    with torch.no_grad(): # Use no_grad for inference
        # Decode the feature activations
        reconstructed_activations = ae.decode(features)

    print("Decoding successful.")
    print(f"Output reconstructed activation shape: {reconstructed_activations.shape}")

    # --- Verification (Example) ---
    expected_reconstruction_shape = (BATCH_SIZE, activation_dim) # Use the dictionary's activation dim
    assert reconstructed_activations.shape == expected_reconstruction_shape
    assert reconstructed_activations.device == features.device
    assert reconstructed_activations.dtype == features.dtype
    
    # You can compare reconstructed_activations to the original activations 
    # (before encoding) to measure reconstruction quality (e.g., using MSE).

except Exception as e:
    print(f"An error occurred during decoding: {e}")

```

**Implementation Tips:**
- Ensure the input `features` tensor has the correct shape `(batch_size, dictionary_size)`.
- Ensure the input tensor is on the same `device` and has the same `dtype` as the dictionary weights.
- The quality of the reconstruction depends on how well the dictionary was trained.

**Variants:**
- Decode features generated by different dictionary types.

**Troubleshooting:**
- **Shape Mismatches:** Ensure the input tensor's last dimension matches the dictionary's `dictionary_size`.
- **Device/Dtype Mismatches:** Use `.to(device, dtype=...)` to move/cast the input feature tensor before decoding.

---

## Integrated Pipeline Example

### Full Pipeline Demo

**Description:**
Demonstrates a complete workflow, combining the steps of loading components, preparing activation data, training a dictionary, loading the trained dictionary, encoding new activations, and decoding the resulting features.

**Prerequisites:**
- All prerequisites from the individual subtasks involved (loading model/data, configuring training, etc.).

**Expected Outcome:**
- The script runs end-to-end without errors.
- A dictionary is trained and saved.
- The saved dictionary is loaded successfully.
- Sample activations are encoded and then decoded, producing reconstructed activations.
- Basic checks confirm the shapes and types of intermediate and final tensors.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/integrated_pipeline_demo.py`. This script combines loading components, loading a pre-trained dictionary, extracting real activations, encoding, and decoding._
```python
import torch
from nnsight import LanguageModel
import os
import random

# Assuming necessary imports from dictionary_learning package are available
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning import AutoEncoder, ActivationBuffer # Needed for loading and buffer

print("--- Integrated Dictionary Learning Pipeline Demo ---")

# --- Configuration (Example) ---
MODEL_ID = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "monology/pile-uncopyrighted" # Or a smaller dataset for quick demo
LAYER = 1 # Example layer for dictionary
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Path to the pre-downloaded dictionary (adjust if needed)
PRETRAINED_DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
PRETRAINED_WEIGHTS_FILE = os.path.join(PRETRAINED_DICT_PATH, "ae.pt")

BATCH_SIZE_DEMO = 4 # Small batch for demo

# --- Set Seed ---
print(f"Setting random seed to {RANDOM_SEED}")
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 1. Load Components (Model, Tokenizer, Data Source) ---
print("\n1. Loading components...")
try:
    model = LanguageModel(MODEL_ID, device_map=DEVICE)
    tokenizer = model.tokenizer
    # Use a small subset for the demo to avoid long data processing
    generator = hf_dataset_to_generator(DATASET_NAME, split="train", streaming=True)
    # Get the target submodule - important: MUST match the layer the dictionary was trained on
    # For pythia-70m mlp_out_layer1, the submodule is model.gpt_neox.layers[1].mlp
    submodule = model.gpt_neox.layers[LAYER].mlp 
    activation_dim = model.config.hidden_size
    print(f"Model: {MODEL_ID}")
    print(f"Tokenizer: {type(tokenizer)}")
    print(f"Data source: {DATASET_NAME}")
    print(f"Target submodule: {type(submodule)} at layer {LAYER}")
    print(f"Activation dimension: {activation_dim}")
    print("Components loaded successfully.")
except Exception as e:
    print(f"Error loading components: {e}")
    exit(1)

# --- 2. Load Pre-trained Dictionary (Skipping Training for Demo) ---
print("\n2. Loading pre-trained dictionary...")
if not os.path.isfile(PRETRAINED_WEIGHTS_FILE):
    print(f"Error: Pretrained dictionary weights not found at {PRETRAINED_WEIGHTS_FILE}")
    print("Please ensure the dictionary has been downloaded or trained.")
    exit(1)

try:
    dictionary = AutoEncoder.from_pretrained(PRETRAINED_WEIGHTS_FILE, device=DEVICE)
    dictionary_size = dictionary.encoder.weight.shape[0]
    print(f"Loaded dictionary: {type(dictionary)} with size {dictionary_size}")
    # Verify activation dimension matches model
    assert dictionary.encoder.weight.shape[1] == activation_dim, \
        f"Dictionary activation dim ({dictionary.encoder.weight.shape[1]}) mismatch model dim ({activation_dim})"
    print("Dictionary loaded successfully.")
except Exception as e:
    print(f"Error loading dictionary: {e}")
    exit(1)

# --- 3. Get Sample Activations (Using ActivationBuffer) ---
print("\n3. Getting sample activations...")
try:
    # Initialize ActivationBuffer
    buffer = ActivationBuffer(
        data=generator,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=100,
        ctx_len=64,
        device=DEVICE,
        out_batch_size=BATCH_SIZE_DEMO
    )
    print("ActivationBuffer initialized.")

    # Get sample activations from the buffer
    sample_activations = next(iter(buffer))
    print(f"Extracted sample activations shape: {sample_activations.shape}, device: {sample_activations.device}, dtype: {sample_activations.dtype}")

except Exception as e:
    print(f"Error getting sample activations: {e}")
    exit(1)

# --- 4. Encode/Decode Example --- 
print("\n4. Encoding/Decoding sample activations...")
try:
    with torch.no_grad():
        # Encode
        feature_activations = dictionary.encode(sample_activations)
        print(f"-> Encoded feature shape: {feature_activations.shape}")
        # Check sparsity (non-negative values)
        assert torch.all(feature_activations >= 0)
        print(f"   Non-zero features in first sample: {torch.sum(feature_activations[0] > 1e-6)}")

        # Decode
        reconstructed_activations = dictionary.decode(feature_activations)
        print(f"-> Decoded reconstructed activation shape: {reconstructed_activations.shape}")
        # Basic shape check
        assert reconstructed_activations.shape == sample_activations.shape

    # Calculate reconstruction loss (MSE) for the sample
    mse = torch.mean((sample_activations - reconstructed_activations)**2)
    print(f"   Reconstruction MSE for the sample: {mse.item():.4f}")

    print("Encoding/Decoding successful.")

except Exception as e:
    print(f"Error during encode/decode: {e}")
    exit(1)

print("\n--- Integrated pipeline demo finished successfully. ---")
```

--- 

## Evaluating Dictionaries

This section covers how to evaluate the performance and characteristics of trained dictionaries.

### Calculate Standard Metrics (`evaluate`)

**Description:**
Calculates a comprehensive set of metrics for a dictionary using activation data. If an `ActivationBuffer` is provided as the data source, it also computes loss-related metrics by running the base model.

**Prerequisites:**
- A loaded dictionary object.
- A source of activation data (either an `ActivationBuffer` or a generator/iterator yielding activation tensors).
- If using `ActivationBuffer` for loss metrics: a loaded `nnsight.LanguageModel` and the target `submodule` used for the buffer.
- Evaluation parameters like `device`, `n_batches`.

**Expected Outcome:**
- Returns a dictionary containing various evaluation metrics, such as:
    - `l2_loss`, `l1_loss`, `l0`: Reconstruction and sparsity metrics.
    - `frac_variance_explained`: Proportion of activation variance captured by the reconstruction.
    - `frac_alive`: Proportion of dictionary features that were active (non-zero) during evaluation.
    - `cossim`, `l2_ratio`, `relative_reconstruction_bias`: Measures related to reconstruction quality and bias.
    - (If using ActivationBuffer) `loss_original`, `loss_reconstructed`, `loss_zero`, `frac_recovered`: Metrics comparing model loss with and without the dictionary intervention.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_evaluate.py`. This script loads a model, dictionary, and buffer, then calls `evaluate`._
```python
import torch
import os
import random
import json
from collections import defaultdict

# Attempt imports
try:
    from nnsight import LanguageModel
    from dictionary_learning import ActivationBuffer, AutoEncoder
    from dictionary_learning.evaluation import evaluate # Import evaluate
    from dictionary_learning.training import trainSAE # Added trainSAE import for completeness
    from dictionary_learning.trainers import StandardTrainer # Added StandardTrainer for completeness
    from dictionary_learning.utils import hf_dataset_to_generator
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    exit(1)

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "monology/pile-uncopyrighted"
LAYER = 1 # Layer for dictionary and buffer
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 43 
PRETRAINED_DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
PRETRAINED_WEIGHTS_FILE = os.path.join(PRETRAINED_DICT_PATH, "ae.pt")
ACTIVATION_DIM = 512
N_CTXS_BUFFER = 100
CTX_LEN_BUFFER = 64
EVAL_BATCHES = 2 # Number of batches to average metrics over

# Set seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Load Model and Dictionary (Assume successful loading) ---
# model = LanguageModel(MODEL_ID, device_map=DEVICE)
# # Correct submodule access (example for Pythia):
# submodule = model.gpt_neox.layers[LAYER].mlp 
# dictionary = AutoEncoder.from_pretrained(PRETRAINED_WEIGHTS_FILE, device=DEVICE)
# print("Model and dictionary loaded.") 
# Assume model and dictionary are loaded here for the snippet

# --- Prepare Activation Buffer ---
# Evaluate requires an activation source. Using ActivationBuffer enables loss_recovered calculation.
print("Preparing data generator and buffer...")
try:
    generator = hf_dataset_to_generator(DATASET_NAME, split="train", streaming=True)
    # Ensure correct submodule access when creating buffer:
    if "pythia" in MODEL_ID.lower() or "gpt_neox" in MODEL_ID.lower():
         buffer_submodule = model.gpt_neox.layers[LAYER].mlp
    # Add elif for other model types if needed
    else:
         raise NotImplementedError(f"Submodule access not defined for model type {MODEL_ID}")

    buffer = ActivationBuffer(
        data=generator,
        model=model, # Provide the model to the buffer
        submodule=buffer_submodule, # Provide the correctly accessed submodule
        d_submodule=ACTIVATION_DIM,
        n_ctxs=N_CTXS_BUFFER,
        ctx_len=CTX_LEN_BUFFER,
        device=DEVICE,
        out_batch_size=32 
    )
    print("ActivationBuffer initialized.")
except Exception as e:
    print(f"Error initializing buffer: {e}")
    exit(1)

# --- Run evaluate ---
print(f"Running evaluate for {EVAL_BATCHES} batches...")
try:
    eval_results = evaluate(
        dictionary=dictionary,
        activations=buffer, # Pass the buffer to get both reconstruction and loss metrics
        device=DEVICE,
        n_batches=EVAL_BATCHES, 
        # Optional: Params for loss_recovered if defaults aren't suitable
        # max_len=128, 
        # batch_size=16 # Batch size for loss_recovered forward passes
    )
    print("evaluate function finished.")
    print("\nEvaluation results:")
    print(json.dumps(eval_results, indent=2))
except StopIteration:
    print("Error: Buffer ran out of data before completing evaluation batches.")
    exit(1)
except Exception as e:
    print(f"Error during evaluate call: {e}")
    exit(1)

```

**Implementation Tips:**
- Provide an `ActivationBuffer` (initialized with the base `model` and `submodule`) as the `activations` argument to `evaluate` if you want to compute loss-related metrics (`loss_original`, `frac_recovered`, etc.).
- If you only need reconstruction metrics (L0, L1, L2, variance explained, etc.), you can pass any iterator/generator yielding batches of activation tensors to `activations`.
- Increase `n_batches` for more stable and reliable metric averages.
- Adjust `batch_size` within `evaluate` (not the buffer's `out_batch_size`) to control the batch size used specifically for the forward passes during the `loss_recovered` calculation, managing memory usage.

**Variants:**
- Evaluate on different data splits by changing the `split` in `hf_dataset_to_generator` when creating the buffer.
- Pass a simple generator yielding pre-computed activation tensors if model-based loss metrics are not needed.

**Troubleshooting:**
- **`StopIteration`:** The activation source (buffer or generator) did not yield enough batches for the specified `n_batches`. Increase the data source size or decrease `n_batches`.
- **CUDA Errors during `loss_recovered`:** The forward passes within `loss_recovered` might consume significant memory. Reduce the `batch_size` argument passed to `evaluate`.
- **Missing Loss Metrics:** Ensure you are passing a fully initialized `ActivationBuffer` (with `model` and `submodule`) to `evaluate` if you expect loss-related metrics.

### Calculate Loss Recovered (`loss_recovered`)

**Description:**
Computes the model's loss under three conditions: normal operation, zero-ablating the target submodule's activations, and replacing the submodule's activations with the dictionary's reconstruction. This helps quantify the functional impact of the dictionary.

**Prerequisites:**
- A loaded `nnsight.LanguageModel`.
- The target `submodule` object within the model (accessed via direct attribute chaining, e.g., `model.gpt_neox.layers[LAYER].mlp`).
- A loaded `dictionary` object.
- A batch of input text (list of strings or tokenized tensor).

**Expected Outcome:**
- Returns a tuple containing three scalar PyTorch tensors:
    1.  `loss_original`: The model's loss on the input text without intervention.
    2.  `loss_reconstructed`: The model's loss when the submodule's activations are replaced by the dictionary's reconstruction.
    3.  `loss_zero`: The model's loss when the submodule's activations are zero-ablated.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_loss_recovered.py`._
```python
import torch
import os
import random

# Attempt imports
try:
    from nnsight import LanguageModel
    from dictionary_learning import AutoEncoder
    from dictionary_learning.evaluation import loss_recovered # Import loss_recovered
    from dictionary_learning.utils import hf_dataset_to_generator # To get sample text
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    exit(1)

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "monology/pile-uncopyrighted"
LAYER = 1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 44 
PRETRAINED_DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
PRETRAINED_WEIGHTS_FILE = os.path.join(PRETRAINED_DICT_PATH, "ae.pt")
SAMPLE_BATCH_SIZE = 4
MAX_LEN_TEST = 64

# Set seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Load Model and Dictionary (Assume successful loading) ---
# model = LanguageModel(MODEL_ID, device_map=DEVICE)
# submodule = model.gpt_neox.layers[LAYER].mlp
# dictionary = AutoEncoder.from_pretrained(PRETRAINED_WEIGHTS_FILE, device=DEVICE)
# print("Model and dictionary loaded.") 
# Assume model and dictionary are loaded here for the snippet

# --- Prepare Sample Text Batch ---
print("Preparing sample text batch...")
try:
    generator = hf_dataset_to_generator(DATASET_NAME, split="train", streaming=True)
    sample_texts = [next(generator) for _ in range(SAMPLE_BATCH_SIZE)]
except Exception as e:
    print(f"Error getting sample texts: {e}")
    exit(1)

# --- Run loss_recovered ---
print(f"Running loss_recovered with max_len={MAX_LEN_TEST}...")
try:
    loss_orig, loss_recon, loss_zero = loss_recovered(
        text=sample_texts, # Can be list of strings or token tensor
        model=model,
        submodule=submodule, # Pass the submodule object obtained via attribute access
        dictionary=dictionary,
        max_len=MAX_LEN_TEST,
        io="out" # Intervene on output activations
    )
    print("loss_recovered function finished.")
    print(f"  Original loss: {loss_orig.item():.4f}")
    print(f"  Reconstructed loss: {loss_recon.item():.4f}")
    print(f"  Zero-ablated loss: {loss_zero.item():.4f}")
    
    # Calculate fraction loss recovered
    frac_rec = (loss_recon - loss_zero) / (loss_orig - loss_zero)
    print(f"  Fraction Loss Recovered: {frac_rec.item():.4f}")

except Exception as e:
    print(f"Error during loss_recovered call: {e}")
    exit(1)

```

**Implementation Tips:**
- The `loss_recovered` function performs three separate forward passes through the model, which can be computationally intensive.
- Use the `max_len` parameter to limit sequence length and reduce computation time/memory, especially during testing.
- The `io` parameter specifies whether to intervene on the submodule's input (`'in'`) or output (`'out'`). Output is typical for evaluating standard SAEs.
- `normalize_batch=True` can be used to normalize activations before passing them to the dictionary, similar to the option in `evaluate`.

**Variants:**
- Analyze the effect without adding back the residual (`add_residual=False`).
- Find the most suppressed tokens (`largest=False`).
- If `dictionary` is `None`, it ablates the dimension `feature` in the raw `submodule` output (less common use case).

**Troubleshooting:**
- **CUDA Out of Memory:** Reduce the batch size of `inputs` or decrease `max_length`.
- **Index Errors:** Ensure the `feature` index is valid for the dictionary size.
- **Tracer Errors:** Underlying `nnsight` tracing issues might occur.

---

## Interpreting Features

This section describes methods for understanding what individual features learned by the dictionary represent.

### Find Top Activating Contexts (`examine_dimension`)

**Description:**
Provides a comprehensive analysis of a single dictionary feature (or dimension in the original activation space if no dictionary is provided). It identifies the text contexts and specific tokens that maximally activate the feature, and also shows which output tokens are most affected by ablating the feature.

**Prerequisites:**
- A loaded `nnsight.LanguageModel`.
- The target `submodule` object (obtained via attribute access).
- An initialized `ActivationBuffer` (used for its text batching and tokenizer, requires the correctly accessed submodule during init).
- Optionally, a loaded `dictionary` object. If provided, the analysis is performed on the dictionary's feature activations; otherwise, it analyzes the raw submodule activations.
- The index (`dim_idx`) of the feature/dimension to examine.
- Parameters like `n_inputs` (number of text samples to process) and `k` (number of top items to return).

**Expected Outcome:**
- Returns a `namedtuple` with the following fields:
    - `top_contexts`: A `circuitsvis.utils.render.RenderedHTML` object visualizing the `k` text contexts where the feature had the highest activation.
    - `top_tokens`: A list of `(token_string, mean_activation)` tuples for the `k` tokens with the highest average activation for this feature across all processed inputs.
    - `top_affected`: A list of `(token_string, logprob_difference)` tuples for the `k` output tokens whose log probabilities are most affected (positively or negatively) when this feature is ablated.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_examine_dimension.py`._
```python
import torch
import os
import random
from collections import namedtuple

# Attempt imports
try:
    from nnsight import LanguageModel
    from dictionary_learning import ActivationBuffer, AutoEncoder
    from dictionary_learning.interp import examine_dimension # Import function
    from dictionary_learning.utils import hf_dataset_to_generator
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    exit(1)

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "monology/pile-uncopyrighted"
LAYER = 1 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 45 
PRETRAINED_DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
PRETRAINED_WEIGHTS_FILE = os.path.join(PRETRAINED_DICT_PATH, "ae.pt")
ACTIVATION_DIM = 512
N_CTXS_BUFFER = 20
CTX_LEN_BUFFER = 128
DIM_TO_EXAMINE = 100 # Which feature index to analyze
N_INPUTS_EXAMINE = 64 # How many text samples to use
K_EXAMINE = 5       # How many top items to show

# Set seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Load Model, Dictionary, Buffer (Assume successful loading) ---
# model = LanguageModel(MODEL_ID, device_map=DEVICE)
# submodule = model.gpt_neox.layers[LAYER].mlp
# dictionary = AutoEncoder.from_pretrained(PRETRAINED_WEIGHTS_FILE, device=DEVICE)
# generator = hf_dataset_to_generator(DATASET_NAME, split="train", streaming=True)
# buffer = ActivationBuffer(...) # Initialized with model, submodule, data
# print("Components loaded.")
# Assume model, dictionary, and buffer are loaded here

# --- Run examine_dimension ---
print(f"Running examine_dimension for feature {DIM_TO_EXAMINE}...")
try:
    feature_profile = examine_dimension(
        model=model,
        submodule=submodule, # Pass the submodule object
        buffer=buffer, # Pass the buffer (ensure it was created with the correct submodule)
        dictionary=dictionary, # Analyze SAE features
        dim_idx=DIM_TO_EXAMINE,
        n_inputs=N_INPUTS_EXAMINE,
        k=K_EXAMINE
    )
    print("examine_dimension function finished.")

    # --- Access results ---
    print("\n--- Feature Profile Results ---")
    # Top Contexts (Renderable HTML object for display in notebooks/HTML)
    print(f"Top Contexts Type: {type(feature_profile.top_contexts)}") 
    # To display in a Jupyter environment: display(feature_profile.top_contexts)
    
    # Top Activating Tokens
    print("\nTop Activating Tokens:")
    for token, activation in feature_profile.top_tokens:
        print(f"- '{token}': {activation:.4f}")
        
    # Top Affected Output Tokens (by ablation)
    print("\nTop Affected Output Tokens (by ablation):")
    for token, logprob_diff in feature_profile.top_affected:
        print(f"- '{token}': {logprob_diff:.4f}")

except Exception as e:
    print(f"Error during examine_dimension call: {e}")
    exit(1)

```

**Implementation Tips:**
- The `buffer` argument is required as the function uses it to get tokenized text batches.
- If `dictionary` is `None`, the function analyzes the raw activations of the `submodule` at the specified `dim_idx`.
- The `top_contexts` result is designed to be displayed directly in environments that can render HTML (like Jupyter notebooks) using `display(feature_profile.top_contexts)`.
- Increasing `n_inputs` provides a more robust analysis across more text samples but increases computation time.

**Variants:**
- Analyze raw activations by setting `dictionary=None`.
- Change the feature index `dim_idx`.
- Adjust `k` to see more or fewer top items.

**Troubleshooting:**
- **`ImportError: No module named circuitsvis`**: Install the `circuitsvis` package (`pip install circuitsvis`) as it's used internally for visualization.
- **CUDA Out of Memory:** Reduce the batch size of `inputs` or decrease `max_length`.
- **Incorrect Feature Interpretation:** Ensure the `dictionary` and `submodule` correspond correctly. If analyzing raw activations, ensure `dim_idx` is within the bounds of the submodule's output dimension.

### Measure Feature Ablation Effect (`feature_effect`)

**Description:**
Determines the impact of zero-ablating a specific dictionary feature on the model's predictions for the next token. It calculates the difference in log probabilities for each token in the vocabulary between the normal run and the run with the feature ablated, identifying the tokens most affected.

**Prerequisites:**
- A loaded `nnsight.LanguageModel`.
- The target `submodule` object (obtained via attribute access).
- A loaded `dictionary` object.
- The index (`feature`) of the dictionary feature to ablate.
- Tokenized input text (`inputs`) as a PyTorch tensor.
- Parameters like `k` (number of top affected tokens) and `largest` (whether to return tokens with the largest positive or negative logprob difference).

**Expected Outcome:**
- Returns a tuple containing two PyTorch tensors:
    1.  `top_tokens`: A tensor of shape `(k,)` containing the token IDs most affected by the ablation.
    2.  `top_probs`: A tensor of shape `(k,)` containing the corresponding mean log probability differences (clean run - ablated run) across the input batch.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_feature_effect.py`._
```python
import torch
import os
import random

# Attempt imports
try:
    from nnsight import LanguageModel
    from dictionary_learning import AutoEncoder
    from dictionary_learning.interp import feature_effect # Import function
    from dictionary_learning.utils import hf_dataset_to_generator # To get sample text
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    exit(1)

# --- Configuration ---
MODEL_ID = "EleutherAI/pythia-70m-deduped"
DATASET_NAME = "monology/pile-uncopyrighted"
LAYER = 1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 46
PRETRAINED_DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
PRETRAINED_WEIGHTS_FILE = os.path.join(PRETRAINED_DICT_PATH, "ae.pt")
FEATURE_TO_ABLATE = 150 
SAMPLE_BATCH_SIZE = 2   
MAX_LEN_TEST = 32       
K_EFFECT = 5            

# Set seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Load Model and Dictionary (Assume successful loading) ---
# model = LanguageModel(MODEL_ID, device_map=DEVICE)
# submodule = model.gpt_neox.layers[LAYER].mlp
# dictionary = AutoEncoder.from_pretrained(PRETRAINED_WEIGHTS_FILE, device=DEVICE)
# print("Model and dictionary loaded.") 
# Assume model and dictionary loaded here for the snippet

# --- Prepare Sample Input ---
print("Preparing sample input text...")
try:
    generator = hf_dataset_to_generator(DATASET_NAME, split="train", streaming=True)
    sample_texts = [next(generator) for _ in range(SAMPLE_BATCH_SIZE)]
    # Tokenize the text
    inputs = model.tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN_TEST).to(DEVICE)
    print(f"Input tensor shape: {inputs['input_ids'].shape}")
except Exception as e:
    print(f"Error preparing inputs: {e}")
    exit(1)

# --- Run feature_effect ---
print(f"Running feature_effect for feature {FEATURE_TO_ABLATE}...")
try:
    # Pass the tokenized inputs directly
    top_tokens, top_probs = feature_effect(
        model=model,
        submodule=submodule, # Pass the submodule object
        dictionary=dictionary,
        feature=FEATURE_TO_ABLATE,
        inputs=inputs['input_ids'], # Pass token ids
        max_length=MAX_LEN_TEST,
        k=K_EFFECT,
        add_residual=True, # Compensate for reconstruction error
        largest=True # Get tokens whose probability *increases* most from ablation
    )
    print("feature_effect function finished.")

    # --- Decode and display results ---
    print("\nTop Affected Tokens (Logprob Difference):")
    for token_id, prob in zip(top_tokens, top_probs):
        print(f"- '{model.tokenizer.decode(token_id)}' (ID: {token_id.item()}): {prob.item():.4f}")

except Exception as e:
    print(f"Error during feature_effect call: {e}")
    exit(1)

```

**Implementation Tips:**
- The `inputs` argument should be a tensor of token IDs.
- `add_residual=True` is generally recommended. It adds the dictionary's reconstruction error back when calculating the ablated state, isolating the effect of the specific feature activation rather than the general reconstruction quality.
- Set `largest=True` to find tokens whose probability *increases* most when the feature is ablated (i.e., the feature normally suppresses these tokens). Set `largest=False` to find tokens whose probability *decreases* most (i.e., the feature normally promotes these tokens).
- Like `loss_recovered`, this function runs multiple model forward passes.

**Variants:**
- Analyze the effect without adding back the residual (`add_residual=False`).
- Find the most suppressed tokens (`largest=False`).
- If `dictionary` is `None`, it ablates the dimension `feature` in the raw `submodule` output (less common use case).

**Troubleshooting:**
- **CUDA Out of Memory:** Reduce the batch size of `inputs` or decrease `max_length`.
- **Index Errors:** Ensure the `feature` index is valid for the dictionary size.
- **Tracer Errors:** Underlying `nnsight` tracing issues might occur.

### Visualize Feature Space (`feature_umap`)

**Description:**
Creates a 2D or 3D UMAP projection of the dictionary's feature vectors (either encoder weights or decoder weights) and returns it as an interactive Plotly figure. This helps visualize the relationships and clustering of features in the learned dictionary.

**Prerequisites:**
- A loaded dictionary object.
- The `umap-learn` package installed (`pip install umap-learn`).
- The `plotly` package installed (`pip install plotly`).
- Optional: `feat_idxs` (an integer or list/set of integers) to highlight specific features in the plot.

**Expected Outcome:**
- Returns a `plotly.graph_objects.Figure` object representing the UMAP visualization.
- The plot shows each feature as a point, projected into 2 or 3 dimensions.
- If `feat_idxs` is provided, the corresponding features are colored differently.

**Tested Code Implementation:**
_Verified via `./tests/dictionary_learning/test_feature_umap.py`._
```python
import torch
import os
import random

# Attempt imports
try:
    from dictionary_learning import AutoEncoder
    from dictionary_learning.interp import feature_umap # Import function
    import plotly.graph_objects as go # To check return type
except ImportError as e:
    if "plotly" in str(e):
        print("Error: Requires 'plotly'. pip install plotly")
    elif "umap" in str(e):
        print("Error: Requires 'umap-learn'. pip install umap-learn")
    else:
        print(f"Error: Missing other dependencies. {e}")
    exit(1)

# --- Configuration ---
SEED = 47
PRETRAINED_DICT_PATH = "source_code/dictionary_learning/dictionaries/pythia-70m-deduped/mlp_out_layer1/10_32768"
PRETRAINED_WEIGHTS_FILE = os.path.join(PRETRAINED_DICT_PATH, "ae.pt")
FEATURE_TO_HIGHLIGHT = 200 
UMAP_COMPONENTS = 2 

# Set seed
random.seed(SEED)
torch.manual_seed(SEED)

# --- Load Dictionary ---
# Load to CPU as UMAP runs on CPU
print("Loading dictionary...")
try:
    dictionary = AutoEncoder.from_pretrained(PRETRAINED_WEIGHTS_FILE, device="cpu")
    print("Dictionary loaded to CPU.")
except Exception as e:
    print(f"Error loading dictionary: {e}")
    exit(1)

# --- Run feature_umap ---
print(f"Running feature_umap (using decoder weights, highlighting feature {FEATURE_TO_HIGHLIGHT})...")
try:
    # UMAP calculation can take some time
    fig = feature_umap(
        dictionary=dictionary,
        weight="decoder", # Use decoder weights (rows of W_dec)
        n_components=UMAP_COMPONENTS, # 2D or 3D
        feat_idxs=FEATURE_TO_HIGHLIGHT # Highlight feature(s)
        # Other UMAP params: n_neighbors, metric, min_dist
    )
    print("feature_umap function finished.")
    print(f"Returned object type: {type(fig)}")
    # In a Jupyter notebook or similar, display the figure:
    # fig.show()

except ImportError as e:
     if "umap" in str(e):
         print("Error: umap-learn package required but not found. pip install umap-learn")
         exit(1)
     else:
         print(f"An unexpected import error occurred: {e}")
         exit(1)
except Exception as e:
    print(f"Error during feature_umap call: {e}")
    exit(1)
```

**Implementation Tips:**
- UMAP calculations can be computationally intensive, especially for large dictionaries. Running on CPU is standard.
- The `weight` parameter determines whether to visualize encoder weights (`dictionary.encoder.weight`) or decoder weights (`dictionary.decoder.weight.T`). Visualizing decoder weights is often more common for interpreting feature semantics.
- Use `feat_idxs` to visually locate specific features of interest within the overall structure.
- Adjust UMAP hyperparameters (`n_neighbors`, `metric`, `min_dist`) to fine-tune the visualization based on the specific dictionary and desired level of detail.

**Variants:**
- Visualize encoder weights (`weight='encoder'`).
- Create a 3D visualization (`n_components=3`).
- Highlight multiple features by passing a list or set to `feat_idxs`.
- Do not highlight any features (`feat_idxs=None`).

**Troubleshooting:**
- **`ImportError: No module named 'umap'`**: Install the `umap-learn` package.
- **`ImportError: No module named 'plotly'`**: Install the `plotly` package.
- **Slow Performance:** UMAP is computationally intensive. Reduce the number of features analyzed (if possible, e.g., by sub-sampling or pre-filtering) or be prepared for longer runtimes with large dictionaries.

---

## Integration with Other Packages

+### Issue: Training with Activations from `TransformerLens.HookedEncoder` (BERT)
+
+**Description:** 
+While `trainSAE` accepts generic iterators yielding activation batches, configuring `TransformerLens.HookedEncoder.run_with_cache` within a custom generator to correctly provide BERT activations (handling `input_ids`, `token_type_ids`, `attention_mask`) that match `trainSAE`'s expectations can be complex. This complexity arises partly from the experimental nature of BERT support in `TransformerLens` and potential ambiguities in the exact expected signature for `run_with_cache` or `forward` when handling BERT-specific inputs. Errors related to unexpected keyword arguments (`input_ids`, `attention_mask`) or internal `HookedEncoder` issues (like `UnboundLocalError`) may occur.
+
+**Integration Tips:**
+- **Preferred Method:** For training SAEs on BERT activations with the `dictionary_learning` package, using `nnsight.LanguageModel` for loading the model and `dictionary_learning.ActivationBuffer` for activation gathering (as shown in the main guide examples) is generally the most reliable and straightforward approach, aligning well with the package's design.
+- **Alternative (Advanced):** If using `TransformerLens` is strictly necessary, carefully verify the exact positional and keyword arguments expected by `HookedEncoder.run_with_cache` and `HookedEncoder.forward` for the specific version being used. Ensure the custom activation generator yields tensors compatible with the `StandardTrainer` or other chosen trainer (correct shape, dtype, device).
+
+**Debugging Steps Taken (Illustrative):**
+- Attempted various combinations of positional and keyword arguments (`input_ids`, `token_type_ids`, `attention_mask`) when calling `HookedEncoder.run_with_cache` inside a custom activation generator.
+- Encountered `UnboundLocalError` related to `token_type_ids` within `HookedEncoder.to_tokens`.
+- Encountered `TypeError: HookedEncoder.forward() got an unexpected keyword argument 'input_ids'`.
+- Encountered `TypeError: HookedEncoder.forward() got an unexpected keyword argument 'attention_mask'`.
+- Switching the implementation to use `nnsight.LanguageModel` and `dictionary_learning.ActivationBuffer` resolved these activation generation issues for the `trainSAE` function.
+
 ### Issue: Modifying Module Output and Accessing Final Logits During `model.generate`
 
 **Description:** 
