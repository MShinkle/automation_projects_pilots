# NNsight Package Examples and Usage

This document provides a comprehensive guide to using the `nnsight` package, covering key functionalities for interpreting and manipulating the internals of deep learning models.

## Table of Contents
1.  [Loading Models](#loading-models)
2.  [Basic Tracing (Single Prompt)](#basic-tracing-single-prompt)
3.  [Intervening in Model Execution](#intervening-in-model-execution)
4.  [Multi-Token Generation Tracing](#multi-token-generation-tracing)
5.  [Advanced Tracing](#advanced-tracing)
6.  [Remote Execution](#remote-execution) 
7.  [Utilities and Other Features](#utilities-and-other-features)

---

## Loading Models

### Load a Language Model

**Description:**
Load a Hugging Face language model using `nnsight`'s `LanguageModel` class, preparing it for tracing and interventions.

**Prerequisites:**
- A valid Hugging Face model identifier (e.g., `"openai-community/gpt2"`).
- Optional: Device mapping parameters (e.g., `device_map="auto"` or a specific device like `"cuda:0"`).

**Expected Outcome:**
- A `LanguageModel` object is successfully initialized without errors.
- The object wraps the underlying Hugging Face model.
- After running a minimal trace (e.g., `with model.trace("dummy"): pass`), model parameters are placed on the specified device.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_load_language_model.py`._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"

# Example using device_map='auto' (requires CUDA and accelerate)
try:
    model_auto = LanguageModel(MODEL_ID, device_map="auto")
    print(f"Initialized {MODEL_ID} with device_map='auto'.")
    # Run a minimal trace to force model loading and device placement
    with model_auto.trace("test"): 
        pass
    param_device = model_auto.transformer.wte.weight.device
    print(f"Parameter device after trace: {param_device}")
    assert param_device.type == 'cuda'
    print("Model loaded successfully on CUDA.")
except Exception as e:
    print(f"Failed loading with device_map='auto': {e}")

# Example loading explicitly to CPU
try:
    model_cpu = LanguageModel(MODEL_ID, device_map="cpu")
    print(f"\nInitialized {MODEL_ID} onto CPU.")
    # Run a minimal trace to force model loading and device placement
    with model_cpu.trace("test"):
        pass
    param_device = model_cpu.transformer.wte.weight.device
    print(f"Parameter device after trace: {param_device}")
    assert param_device.type == 'cpu'
    print("Model loaded successfully on CPU.")
except Exception as e:
    print(f"Failed loading onto CPU: {e}")

# You can print the model structure (proxy)
# print(model_cpu) 
```

**Implementation Tips:**
- The `LanguageModel` class handles loading models from Hugging Face Hub.
- Use `device_map="auto"` for automatic device placement (requires `accelerate` package). Specify a device like `"cuda"` or `"cpu"` for manual placement.
- **Important:** Actual model loading and device placement are often deferred until the first `trace` or `generate` context is executed. If you need to access parameters directly or verify their device *after* initialization but *before* a full trace, run a minimal trace (e.g., `with model.trace("dummy"): pass`) to force the model onto the specified device(s). Checking `parameter.device` immediately after initialization might yield unexpected results (e.g., 'meta' device) before the trace forces placement.
- Printing the `LanguageModel` object shows the model's architecture proxy, which is useful for understanding the module structure needed for targeting interventions.

**Variants:**
- Load different models by changing the model identifier.
- Specify different `device_map` options (e.g., specific GPU IDs).
- Pass additional arguments supported by `transformers.AutoModelForCausalLM.from_pretrained` via `model_kwargs` during initialization.

**Troubleshooting:**
- **Model Not Found:** Ensure the Hugging Face model identifier is correct and exists.
- **Dependency Issues:** Make sure `transformers`, `torch`, and potentially `accelerate` (for `device_map="auto"`) are installed.
- **Resource Errors:** Large models might require significant memory/GPU resources. Ensure your environment meets the requirements.
- **Device Errors:** If device checks fail, ensure you run a minimal trace *before* checking parameter devices, as loading is often lazy.

---

## Basic Tracing (Single Prompt)

### Run a Simple Trace and Save Output

**Description:**
Perform a basic forward pass on a loaded model for a single prompt using the `trace` context manager and save the final model output logits.

**Prerequisites:**
- An initialized `LanguageModel` object.
- A string containing the input prompt (e.g., `"The Eiffel Tower is in the city of"`).

**Expected Outcome:**
- The `trace` context executes without errors.
- Accessing `model.output.logits` within the trace provides a proxy to the output logits.
- The `.save()` method called on `model.output.logits` captures the model's final output logits tensor.
- The captured output tensor has the expected shape `(batch_size, sequence_length, vocab_size)` and is on the correct device.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_simple_trace_save_output.py`._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"
PROMPT = "The Eiffel Tower is in the city of"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assume 'model' is an initialized LanguageModel object (e.g., from 'Loading Models' section)
# model = LanguageModel(MODEL_ID, device_map=DEVICE)

print(f"Tracing prompt: '{PROMPT}'")

try:
    # Use the trace context manager with the prompt
    with model.trace(PROMPT) as tracer:
        # Access the .logits attribute of the model output and save it
        output_logits_proxy = model.output.logits.save()
    
    print("Trace completed.")

    # Access the saved output logits value after the context exits
    output_logits_value = output_logits_proxy.value

    print(f"Saved output logits type: {type(output_logits_value)}")
    
    # Verification
    assert isinstance(output_logits_value, torch.Tensor)
    print(f"Saved output logits shape: {output_logits_value.shape}")
    print(f"Saved output logits device type: {output_logits_value.device.type}")
    assert len(output_logits_value.shape) == 3
    assert output_logits_value.shape[0] == 1
    assert output_logits_value.shape[1] > 1
    assert output_logits_value.shape[2] == model.config.vocab_size
    assert output_logits_value.device.type == DEVICE
    print("Successfully saved and verified output logits.")

except Exception as e:
    print(f"An error occurred during trace: {e}")

**Implementation Tips:**
- The `model.trace()` context manager sets up the intervention graph. The actual model execution happens *after* the context block exits.
- `model.output` refers to the final output object of the model's `forward` method (often a Hugging Face `*Output` object).
- To get the actual logits tensor, access the `.logits` attribute: `model.output.logits`.
- `.save()` marks the value of the proxy object to be preserved and accessible after the trace execution via the `.value` attribute.

**Variants:**
- Use different input prompts.
- Trace different models.
- Access other components of the output object if needed (e.g., `model.output.hidden_states` if configured in the underlying model).

**Troubleshooting:**
- **AttributeError:** Ensure `model` is a valid `LanguageModel` instance. If accessing attributes like `.logits`, ensure the underlying model's forward pass actually returns an object with that attribute.
- **Runtime Errors during trace execution:** Issues might arise from the underlying model's forward pass or resource limitations. Check model compatibility and resource availability.

### Access and Save Internal Hidden States

**Description:**
Access and save the output (hidden states) of a specific internal module (e.g., a transformer layer) during a model trace.

**Prerequisites:**
- An initialized `LanguageModel` object.
- A string containing the input prompt.
- Knowledge of the path to the target module within the model architecture (e.g., `model.transformer.h[-1]`).

**Expected Outcome:**
- The `trace` context executes without errors.
- Accessing the module via its path (e.g., `model.transformer.h[-1]`) returns a proxy module.
- Accessing the `.output` attribute of the proxy module returns a proxy for its output.
- Indexing the proxy output (if necessary, e.g., `.output[0]`) works correctly.
- Calling `.save()` on the desired proxy output captures the tensor value.
- The captured tensor has the expected shape for the hidden states of that module and is on the correct device.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_save_internal_states.py`._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"
PROMPT = "The Eiffel Tower is in the city of"
TARGET_LAYER = -1 # Last layer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assume 'model' is an initialized LanguageModel object (e.g., from 'Loading Models' section)
# model = LanguageModel(MODEL_ID, device_map=DEVICE)

print(f"Tracing prompt: '{PROMPT}' to get layer {TARGET_LAYER} hidden states")

try:
    with model.trace(PROMPT) as tracer:
        # Access the target layer's output proxy
        # Path for GPT-2: model.transformer.h[layer_index]
        # Output of GPT2Block is a tuple: (hidden_states, ...) 
        # We want the first element (hidden_states)
        hidden_states_proxy = model.transformer.h[TARGET_LAYER].output[0].save()
        print(f"Proxy created for layer {TARGET_LAYER} hidden states.")

    print("Trace completed.")

    # Access the saved hidden state value
    hidden_states_value = hidden_states_proxy.value

    print(f"Saved hidden states type: {type(hidden_states_value)}")

    # Verification
    assert isinstance(hidden_states_value, torch.Tensor)
    print(f"Saved hidden states shape: {hidden_states_value.shape}")
    print(f"Saved hidden states device type: {hidden_states_value.device.type}")
    assert len(hidden_states_value.shape) == 3
    assert hidden_states_value.shape[0] == 1 # Batch size
    assert hidden_states_value.shape[1] > 1 # Sequence length
    assert hidden_states_value.shape[2] == model.config.n_embd # Hidden size
    assert hidden_states_value.device.type == DEVICE
    print("Successfully saved and verified hidden states.")

except Exception as e:
     print(f"An error occurred during trace: {e}")

```

**Implementation Tips:**
- Use `print(model)` to inspect the model's architecture and find the correct path to the desired module (e.g., `transformer.h[layer_index]`, `transformer.h[layer_index].mlp`, `transformer.h[layer_index].attn`).
- Module outputs might be tuples; inspect their structure (e.g., using `.shape` on the proxy) and index appropriately (e.g., `.output[0]`). Use the `_output` attribute to see the raw output before potential tuple unpacking if unsure.
- `.save()` makes the intermediate activation accessible outside the trace context via `.value`.

**Variants:**
- Save activations from different layers or modules (e.g., attention layers, MLPs, embeddings).
- Access module `.input` instead of `.output`.
- Save multiple intermediate values within the same trace.

**Troubleshooting:**
- **AttributeError:** Double-check the module path specified. Ensure it exists in the model architecture shown by `print(model)`.
- **IndexError:** If indexing `.output` (e.g., `.output[0]`), ensure the output is indeed a tuple/list and the index is valid. Check the `.shape` of the proxy or inspect `._output`.
- **Incorrect Shape:** Verify you are accessing the correct part of the module's output if it's complex (e.g., tuples containing hidden states, attention weights, etc.).

### Perform Operations on Proxies

**Description:**
Apply standard Python operations or PyTorch functions to proxy objects within a trace context to manipulate activations dynamically.

**Prerequisites:**
- An initialized `LanguageModel` object.
- A trace context (`model.trace(...)`).
- At least one proxy object (e.g., obtained from `module.output`).

**Expected Outcome:**
- Operations (e.g., `+`, `*`, `torch.sum`, `torch.mean`) applied to proxy objects create new proxy objects representing the result.
- The trace executes without errors.
- Saved results of these operations reflect the computed values.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_proxy_operations.py`._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"
PROMPT = "The Eiffel Tower is in the city of"
TARGET_LAYER = -1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assume 'model' is an initialized LanguageModel object (e.g., from 'Loading Models' section)
# model = LanguageModel(MODEL_ID, device_map=DEVICE)

print(f"Tracing prompt: '{PROMPT}' to test proxy operations")

try:
    with model.trace(PROMPT) as tracer:
        # Get the hidden states proxy (don't save yet)
        hidden_states_proxy = model.transformer.h[TARGET_LAYER].output[0]
        
        # Perform torch.sum operation on the proxy and save
        hs_sum_proxy = torch.sum(hidden_states_proxy).save()
        print(f"Proxy created for torch.sum(hidden_states)")

        # Perform addition between proxies/tensors and save
        # Create tensor on the correct device inside the context
        bias_tensor = torch.ones(1, device=DEVICE) 
        hs_added_proxy = (hidden_states_proxy + hs_sum_proxy + bias_tensor).save()
        print(f"Proxy created for hidden_states + sum + bias")
        
        # Save original hidden states for comparison
        original_hs_proxy = hidden_states_proxy.save()

    print("Trace completed.")

    # Access the saved values
    original_hs_value = original_hs_proxy.value
    hs_sum_value = hs_sum_proxy.value
    hs_added_value = hs_added_proxy.value

    print(f"Original HS shape: {original_hs_value.shape}, Sum value: {hs_sum_value.item():.2f}")
    print(f"Added HS shape: {hs_added_value.shape}")

    # Verification
    assert isinstance(hs_sum_value, torch.Tensor) and hs_sum_value.ndim == 0
    assert isinstance(hs_added_value, torch.Tensor)
    assert hs_added_value.shape == original_hs_value.shape
    assert hs_added_value.device.type == DEVICE

    # Check if the addition approximately worked 
    expected_added_value = original_hs_value + hs_sum_value + torch.ones(1, device=DEVICE)
    assert torch.allclose(hs_added_value, expected_added_value, atol=1e-5)
    print("Proxy operations successful and verified.")

except Exception as e:
     print(f"An error occurred during trace: {e}")

```

**Implementation Tips:**
- Most standard arithmetic operators and many `torch` functions work directly on `nnsight` proxy tensors.
- The operations are added to the computation graph and executed when the trace runs.
- Remember to `.save()` the final result if you want to access its value after the trace via `.value`.
- When operating with concrete tensors (like `bias_tensor` above), ensure they are on the same device as the proxy tensors they interact with. Creating them inside the context or explicitly moving them (`.to(proxy.device)`) is recommended.

**Variants:**
- Use different arithmetic operations.
- Apply various `torch` functions (e.g., `torch.relu`, `torch.norm`, slicing/indexing).
- Chain multiple operations together.

**Troubleshooting:**
- **TypeError/AttributeError:** Not all operations might be supported on proxies. Check `nnsight` documentation or examples for compatibility. Ensure operands have compatible shapes for the operation.
- **Shape Mismatches:** Verify the shapes of proxy tensors involved in operations using `.shape`.
- **Device Mismatches:** Ensure all tensors (proxies and concrete) involved in an operation are on the same device.

---

## Intervening in Model Execution

### Setting Module Outputs

**Description:**
Modify the forward pass by directly setting the output of a specific module during a trace.

**Prerequisites:**
- An initialized `LanguageModel` object.
- A trace context (`model.trace(...)`).
- The path to the target module.
- A tensor or proxy tensor containing the value to set the module's output to. This value must have the correct shape and dtype.

**Expected Outcome:**
- The trace executes without errors.
- The forward pass proceeds using the *modified* output for the specified module, potentially affecting subsequent layers and the final model output.
- Saving the module's output *after* the assignment confirms the change was applied within the trace.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_set_module_output.py`._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"
PROMPT = "The Eiffel Tower is in the city of"
TARGET_LAYER = -1 # Last layer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assume 'model' is an initialized LanguageModel object (e.g., from 'Loading Models' section)
# model = LanguageModel(MODEL_ID, device_map=DEVICE)

# Target the MLP output within the last layer
target_module_path = f"transformer.h[{TARGET_LAYER}].mlp"
print(f"Targeting module: {target_module_path}")
print(f"Tracing prompt: '{PROMPT}'")

try:
    with model.trace(PROMPT) as tracer:
        # Get the original output proxy to determine shape and dtype
        original_mlp_output_proxy = model.transformer.h[TARGET_LAYER].mlp.output
        original_shape = original_mlp_output_proxy.shape
        original_dtype = original_mlp_output_proxy.dtype
        print(f"Original MLP output shape proxy: {original_shape}")

        # Save the original output for comparison later
        original_mlp_output_proxy.save()

        # Create a new tensor (e.g., zeros) with the correct shape, dtype, and device
        zeros_tensor = torch.zeros(original_shape, dtype=original_dtype, device=DEVICE)
        print(f"Created zeros tensor with shape: {zeros_tensor.shape}")

        # Set the output of the target MLP module using the assignment operator
        model.transformer.h[TARGET_LAYER].mlp.output = zeros_tensor
        print(f"Intervention: Set {target_module_path}.output = zeros_tensor")

        # Save the output *after* setting it to verify the intervention
        modified_mlp_output_proxy = model.transformer.h[TARGET_LAYER].mlp.output.save()

    print("Trace completed.")

    # Access the saved values
    original_mlp_output_value = original_mlp_output_proxy.value
    modified_mlp_output_value = modified_mlp_output_proxy.value

    print(f"Original MLP output mean: {original_mlp_output_value.mean().item():.4f}")
    print(f"Modified MLP output mean: {modified_mlp_output_value.mean().item():.4f}")

    # Verification
    assert isinstance(modified_mlp_output_value, torch.Tensor)
    assert modified_mlp_output_value.shape == original_mlp_output_value.shape
    assert modified_mlp_output_value.device.type == DEVICE
    # Check if the output was indeed set to zeros
    assert torch.all(modified_mlp_output_value == 0), "Intervention failed: output is not all zeros"
    # Sanity check the original was not zero
    assert not torch.all(original_mlp_output_value == 0)
    print("Successfully set module output to zeros and verified.")

except Exception as e:
    print(f"An error occurred during trace: {e}")

```

**Implementation Tips:**
- Use the assignment operator (`=`) on the `.output` attribute of a module proxy (e.g., `model.transformer.h[-1].mlp.output = new_value`).
- The `new_value` can be a concrete `torch.Tensor` or another proxy tensor.
- **Crucially**, ensure the `new_value` has the exact shape and dtype expected as the output of the target module. You can get these from the original output proxy's `.shape` and `.dtype` attributes within the trace context.
- Ensure the `new_value` tensor is on the same device as the model's execution device.
- Use `.clone()` on the original output proxy if you need to modify it before setting (e.g., `original_output = model.module.output.clone(); model.module.output = original_output + noise`).

**Variants:**
- Set the output of different modules (e.g., attention layers, layer norms).
- Set the output using a pre-defined tensor or a dynamically computed proxy tensor (e.g., adding noise, scaling).
- Set module `.input` instead (use with caution, input structure might be complex, often a tuple).

**Troubleshooting:**
- **Shape/Dtype Mismatch:** The most common issue. Ensure the assigned tensor strictly matches the expected output shape and dtype.
- **Device Mismatch:** Ensure the assigned tensor is on the correct device. Create it within the context or move it explicitly.
- **AttributeError:** Verify the module path.
- **Unexpected Downstream Effects:** Interventions can significantly alter model behavior; analyze subsequent activations or the final model output to understand the impact.

### Modify Module Outputs In-Place

**Description:**
Modify parts of a module's output tensor directly using slice assignment during a trace.

**Prerequisites:**
- An initialized `LanguageModel` object.
- A trace context (`model.trace(...)`).
- The path to the target module's output proxy.
- A tensor or proxy tensor containing the value to assign, matching the shape and dtype of the *slice* being modified.
- Slice indices (e.g., specific token positions, hidden dimensions).

**Expected Outcome:**
- The trace executes without errors.
- The forward pass proceeds using the module output with the specified slice modified.
- Saving the module's output *after* the assignment confirms the targeted change was applied within the trace, while other parts remain unchanged.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_inplace_modification.py`._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"
PROMPT = "The Eiffel Tower is in the city of"
TARGET_LAYER = -1 # Last layer
TARGET_TOKEN_IDX = -1 # Last token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assume 'model' is an initialized LanguageModel object (e.g., from 'Loading Models' section)
# model = LanguageModel(MODEL_ID, device_map=DEVICE)

target_module_path = f"transformer.h[{TARGET_LAYER}].mlp"
print(f"Targeting module: {target_module_path}")
print(f"Tracing prompt: '{PROMPT}'")

try:
    with model.trace(PROMPT) as tracer:
        # Get the original output proxy 
        original_mlp_output_proxy = model.transformer.h[TARGET_LAYER].mlp.output
        
        # Save the original output for comparison later
        saved_original_output = original_mlp_output_proxy.clone().save()
        
        # Create a value to set (e.g., zeros)
        # We need to match the shape of the slice we are replacing
        # Shape is (batch_size, seq_len, hidden_size)
        # Target slice: (batch_size, 1, hidden_size) for a single token index
        batch_size = original_mlp_output_proxy.shape[0]
        hidden_size = original_mlp_output_proxy.shape[2]
        zeros_slice = torch.zeros((batch_size, 1, hidden_size), 
                                  dtype=original_mlp_output_proxy.dtype, 
                                  device=DEVICE)
        
        print(f"  - Modifying output at token index {TARGET_TOKEN_IDX} to zeros.")
        # Perform in-place modification using slice assignment (__setitem__)
        model.transformer.h[TARGET_LAYER].mlp.output[:, TARGET_TOKEN_IDX, :] = zeros_slice
        
        # Save the output *after* modification to verify
        saved_modified_output = model.transformer.h[TARGET_LAYER].mlp.output.save()

    print("Trace completed.")

    # Access the saved values
    original_value = saved_original_output.value
    modified_value = saved_modified_output.value

    print(f"Original output shape: {original_value.shape}")
    print(f"Modified output shape: {modified_value.shape}")

    # Verification
    assert isinstance(modified_value, torch.Tensor)
    assert modified_value.shape == original_value.shape
    # Check if the target slice was indeed set to zeros
    modified_slice = modified_value[:, TARGET_TOKEN_IDX, :]
    assert torch.all(modified_slice == 0)
    # Check that other parts were NOT set to zero (if applicable)
    if TARGET_TOKEN_IDX != 0: 
        original_prev_slice = original_value[:, TARGET_TOKEN_IDX -1, :]
        modified_prev_slice = modified_value[:, TARGET_TOKEN_IDX -1, :]
        assert torch.allclose(modified_prev_slice, original_prev_slice)
    
    print("Successfully modified module output in-place and verified.")

except Exception as e:
    print(f"An error occurred during trace: {e}")

```

**Implementation Tips:**
- Use standard PyTorch tensor slicing and assignment on the proxy object (e.g., `proxy[:, token_index, :] = new_value`). This implicitly uses the `__setitem__` operation defined for proxies.
- The `new_value` must match the shape and dtype of the *slice* being assigned to, not the entire tensor.
- Ensure the `new_value` tensor is on the correct device.
- Use `.clone()` on the original proxy if you need its unmodified value for comparison or calculation before the in-place modification.

**Variants:**
- Modify different slices (e.g., specific hidden dimensions, multiple tokens).
- Use dynamically computed values (proxies or tensors created within the trace) for the assignment.
- Apply mathematical operations in-place (e.g., `proxy[:, token_index, :] *= 0.1`).

**Troubleshooting:**
- **Shape/Dtype Mismatch:** Ensure the assigned tensor strictly matches the shape and dtype of the target slice.
- **Device Mismatch:** Ensure the assigned tensor is on the correct device.
- **Index Errors:** Double-check slice indices to ensure they are valid for the tensor dimensions.
- **AttributeError:** Verify the module path to the output proxy.

--- 

## Multi-Token Generation Tracing

### Save Intermediate States During Generation

**Description:**
Use the `model.generate()` context manager to save internal model states (like hidden states) at specific generation steps.

**Prerequisites:**
- An initialized `LanguageModel` object.
- A string containing the input prompt.
- Generation parameters (e.g., `max_new_tokens`).
- Knowledge of the path to the target module.

**Expected Outcome:**
- The `generate` context executes without errors.
- Accessing the module proxy and calling `.next()` advances the proxy to the subsequent generation step.
- Calling `.save()` on the proxy's output at different steps captures the internal state *for that specific step*.
- The captured tensors have the expected shape:
    - For the first step (processing the initial prompt): `(batch_size, prompt_sequence_length, hidden_size)`.
    - For subsequent generation steps (processing one token at a time): `(batch_size, 1, hidden_size)`.

**Tested Code Implementation:**
_Verified via `./tests/nnsight/test_multi_token_generation.py` (modified to focus only on intermediate state saving)._
```python
import torch
from nnsight import LanguageModel

MODEL_ID = "openai-community/gpt2"
PROMPT = "The Eiffel Tower"
TARGET_LAYER = -1 # Last layer
MAX_NEW_TOKENS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Assume 'model' is an initialized LanguageModel object (e.g., from 'Loading Models' section)
# model = LanguageModel(MODEL_ID, device_map=DEVICE)

print(f"Tracing generation for prompt: '{PROMPT}'")
hs_values = {}
try:
    # Use the generate context manager
    with model.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS, do_sample=False) as generator:
        # Get proxy for the target layer module
        module_proxy = generator.model.transformer.h[TARGET_LAYER]
        
        # Loop through desired generation steps (1 to MAX_NEW_TOKENS)
        # Step 1 corresponds to processing the prompt
        for i in range(MAX_NEW_TOKENS + 1): # +1 because step 0 isn't used, steps are 1-based
            step = i + 1
            if step == 1: # Step 1: Processing the prompt
                 # Save hidden states (output[0]) for the prompt processing step
                current_hs_proxy = module_proxy.output[0].save()
                print(f"Saved hidden states proxy for prompt processing (Step {step})")
                hs_values[f"step_{step}"] = current_hs_proxy
            elif step <= MAX_NEW_TOKENS + 1: # Steps 2 through MAX_NEW_TOKENS+1: Processing generated tokens
                 # Advance module proxy to the next generation step *before* saving
                 # This aligns saving with the state *after* generating token i (where i = step - 1)
                module_proxy = module_proxy.next()
                 # Save hidden states (output[0]) for the current generated token step
                current_hs_proxy = module_proxy.output[0].save()
                print(f"Saved hidden states proxy for generated token {step-1} (Step {step})")
                hs_values[f"step_{step}"] = current_hs_proxy

    print("Generation trace completed.")

    # Access and verify the saved hidden state values
    initial_tokens = model.tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]
    hidden_size = model.config.n_embd
    expected_hs_shape_step1 = (1, initial_tokens, hidden_size)
    expected_hs_shape_subsequent = (1, 1, hidden_size)

    for i in range(MAX_NEW_TOKENS + 1):
        step = i + 1
        key = f"step_{step}"
        hs_value = hs_values[key].value
        print(f"HS Step {step} shape: {hs_value.shape}")
        assert isinstance(hs_value, torch.Tensor)
        assert hs_value.device.type == DEVICE
        if step == 1:
            assert hs_value.shape == expected_hs_shape_step1, f"Shape mismatch for step {step}"
        else:
            assert hs_value.shape == expected_hs_shape_subsequent, f"Shape mismatch for step {step}"
            if step > 2:
                prev_key = f"step_{step-1}"
                prev_hs_value = hs_values[prev_key].value
                assert not torch.equal(hs_value, prev_hs_value), f"HS{step} and HS{step-1} are identical"
    
    print("Successfully saved and verified intermediate hidden states for prompt and each generation step.")

except Exception as e:
     print(f"An error occurred during generation trace: {e}")

```

**Implementation Tips:**
- Use `model.generate()` as a context manager for interventions during generation.
- Get a proxy for the *module* you want to observe (e.g., `generator.model.transformer.h[-1]`).
- Inside the context, use a loop or sequential calls:
    - Save the desired output (e.g., `.output[0]`) for the *first* step (prompt processing).
    - Then, for subsequent steps, first call `.next()` on the *module proxy* and *then* save the output. This aligns the saved state with the processing for each newly generated token.
- `.save()` captures the state for the tokens being processed at that step: the full prompt sequence for the first step, and a single token for subsequent steps.
- **Getting Final Output:** The `model.generate()` context manager (`generator`) is primarily designed for interventions and saving intermediate states. Retrieving the final generated sequence itself is typically done by calling `model.generate(...)` *without* the `with` context manager. If you need both intermediate states and the final sequence, consider running generation twice, or potentially inspecting the `generator` object after the context closes (though this is less standard).

**Variants:**
- Save states from different modules during generation.
- Intervene (e.g., set module outputs) at specific generation steps using `.next()` and assignment.
- Adjust `max_new_tokens` and other generation parameters.

**Troubleshooting:**
- **Shape Mismatches:** Remember that saving module output within the generation loop captures the state for the *full prompt* at step 1, and only the *current token* being processed in subsequent steps (sequence length of 1).

---

## Advanced Tracing
(Sections to be added: Cross-Prompt Interventions, etc.)

---

## Remote Execution
(Sections to be added based on potential library features)

---

## Utilities and Other Features
(Sections to be added based on potential library features)

---

## Integrated Workflow Example

The script `./tests/nnsight/integrated_pipeline_demo.py` demonstrates combining many of the functionalities described above (loading, tracing, intervening, generating) into a cohesive workflow. Refer to this script for an example of how these components work together in practice.

---

## Integration with Other Packages

### Issue: Modifying Module Output and Accessing Final Logits During `model.generate`

**Description:** 
When attempting to modify an intermediate module's output (e.g., `module_proxy.output = new_value`) and *also* access the final model output proxy (e.g., `generator.model.output.logits.save()`) within the same step of a `model.generate()` context loop, errors are highly likely. Debugging revealed that even after successfully setting an intermediate module's output, subsequent attempts to access `generator.model.output` (e.g., to save logits) within that same loop step consistently resulted in `AttributeError: 'Proxy' object has no attribute 'save'`. This occurred regardless of the order of operations (e.g., Intervene->Save vs. Intervene->Next->Save). However, saving the final logits *without* performing an intervention, or saving intermediate module activations *with* intervention, works correctly. This strongly suggests that modifying an intermediate module state via direct assignment (`.output = ...`) interferes with the construction or accessibility of the *final* model output proxy (`generator.model.output`) within that same generation step.

**Integration Tips & Workaround:**
- The fundamental issue appears to be the interaction between direct intermediate state modification and final model output access within the multi-step `generate` context.
- **Do not attempt to access `generator.model.output` (e.g., for saving logits) in the same `generate()` loop step where you modify an intermediate module's output using proxy assignment (`module_proxy.output = ...`).**
- **Workaround for Analysis:** As previously noted, if you need to analyze the effect of an intervention (like replacing activations with an SAE reconstruction), use the two-trace approach with `model.trace()`. This allows stable analysis of the intervention's impact on the *next single token prediction* but does not support analyzing a full generation sequence under continuous intervention.

**Debugging Steps Taken:**
- Verified intervention logic (`module.output = ...`) works in `model.trace`.
- Verified saving final logits (`generator.model.output.logits.save()`) works in `model.generate` *without* intervention.
- Attempted intervention combined with saving final logits within `model.generate` loop.
- Consistently encountered `AttributeError: 'Proxy' object has no attribute 'save'` when accessing `generator.model.output` *after* the intervention assignment within the loop step, irrespective of ordering relative to `.next()`.
- This confirms the limitation lies in combining intermediate module output assignment with final model output access within the `generate` context loop.

### Issue: Modifying Elements within Tupled Module Inputs/Outputs

**Description:** 
During attempts to intervene on modules whose inputs or outputs are tuples (common in Hugging Face transformer block implementations, e.g., `output=(hidden_states, present_key_value_states)` or `input=(hidden_states, attention_mask, ...)`), modifying specific elements of the tuple proxy within the `nnsight.trace` context (e.g., `module.output[0] = new_value` or `module.input[0][0] = new_value`) proved unreliable for the `openai-community/gpt2` model. Operations involving indexing (`getitem`) and assignment (`setitem`) on these tuple element proxies consistently led to runtime errors during trace execution (e.g., `TypeError: type object '_empty' has no attribute '__bool__'`, `IndexError`, or errors related to `getitem_X`/`setitem_X` nodes in the NNsight execution graph). This occurred both with direct assignment and arithmetic modification (e.g., `output[0] = original[0] + delta`).

**Integration Tips & Workaround:**
- **Avoid Modifying Tuple Elements:** When performing interventions with `nnsight`, prefer targeting modules whose inputs/outputs are single tensors rather than tuples, if possible. For example, intervening on the entire output of an MLP layer (`module.mlp.output = ...`) is generally more robust than trying to modify only the hidden state component within a block's output tuple (`module.output[0] = ...`).
- **Alternative Targets:** If you need to affect the residual stream before a specific block (e.g., corresponding to `hook_resid_pre`), consider intervening on the *entire output* of the preceding block or layer (e.g., set `model.transformer.h[N-1].output = ...` if the output is a single tensor, or target the MLP output `model.transformer.h[N-1].mlp.output = ...`). This modifies the input to the target block (Block N) indirectly but avoids interacting with potentially problematic tuple structures during the intervention assignment.
- **Check Module Output Structure:** Before attempting intervention, carefully inspect the target module's expected output structure (e.g., using `print(module)` or documentation). If it involves tuples, anticipate potential difficulties with direct element modification within the trace.

**Debugging Steps Taken:**
- Attempted to set `module.input[0][0]` using assignment. Resulted in `getitem` errors during trace execution.
- Attempted to set `module.output[0]` using assignment. Resulted in `setitem` errors during trace execution.
- Attempted to modify `module.output[0]` using arithmetic (`module.output[0] = original[0] + delta`). Still resulted in `getitem` errors (likely during the `original[0]` access or the arithmetic operation on the proxy).
- Successfully intervened by setting the *entire* output of an MLP module (`module.mlp.output = ...`) which returns a single tensor, confirming this method is more stable.

### Issue: Compatibility with Test Frameworks (e.g., pytest)

**Description:** 
Running `nnsight` traces (using `model.trace()` or `model.generate()`) within automated test frameworks like `pytest`, especially when using fixtures or complex test collection setups, can sometimes lead to unexpected hangs or errors. This may be due to interactions between `nnsight`'s internal context management, potential use of multiprocessing, and the test runner's execution model.

**Integration Tips & Workaround:**
- **Verify with Direct Execution:** If tests involving `nnsight` traces hang or fail unexpectedly, first try running the core logic in a standalone Python script (`python your_script.py`). If the script runs correctly outside the test framework, the issue likely lies in the interaction with the framework.
- **Standalone Test Scripts:** Consider structuring tests that heavily rely on `nnsight` tracing as standalone executable scripts that perform setup, run the trace, and use standard `assert` statements for verification, rather than relying on `pytest` fixtures and test functions. These scripts can still be integrated into a larger testing workflow if needed.

**Debugging Steps Taken:**
- Observed hangs when running `nnsight` traces within `pytest` fixtures.
- Successfully executed the same core logic via direct `python` script execution.
- Refactored tests to remove `pytest` dependency, allowing verification to proceed.

--- 