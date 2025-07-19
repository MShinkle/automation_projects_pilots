This is a structured sandbox setup for implementing and evaluating different SAE variants. It's designed such that aspects of the SAE architecture and/or optimization objectives can be easily changed, with these changes localized entirely to a single `.py` file for a given SAE variant. Concretely, custom SAE variants can be implemented by following these steps:
1. Create a new file named `[variant_name].py` in `./custom_saes/`.
2. Import the `BaseSAE` and `BaseTrainer` classes from `./custom_saes/base.py`.
3. Define two new classes called `CustomSAE` and `CustomTrainer`, which inherit from the base classes.
4. Re-define any of the components of base classes to implement the variant:
    * **`BaseSAE`:**
        * `encode(self, x)`: Change the encoding process, such as using a different non-linearity (instead of `ReLU`), adding normalization, or altering the linear transformation.
        * `decode(self, f)`: Modify the decoding process, potentially mirroring changes made to `encode` or using a different reconstruction approach.
        * `forward(self, x, output_features=False)`: Alter the overall forward pass, perhaps for SAEs with non-standard architectures or those requiring different input/output handling.
    * **`BaseTrainer`:**
        * `loss(self, x, step: int, logging=False, **kwargs)`: Implement custom loss functions, such as different sparsity penalties (e.g., L0, cosine similarity based), auxiliary losses, or modified reconstruction terms. Ensure `self.update_dead_neurons(f)` is still called unless you specifically want to disable dead neuron tracking.

## Workflow Scripts

Once you have created your custom SAE variant file (e.g., `./custom_saes/my_variant.py`), you should run the following three main scripts in order:

1. **`train_sae.py`**: Trains a specified SAE variant (`python train_sae.py [variant_name]`) using activations from `pythia-70m-deduped` layer 3 and saves the model to `./trained_saes/`.
2. **`benchmark_sae.py`**: Evaluates trained SAE models (`python benchmark_sae.py [optional_variant_name]`) from `./trained_saes/` using `sae-bench` and saves results to `./benchmark_results/`.
3. **`print_metrics.py`**: Loads and displays key metrics (`python print_metrics.py [optional_variant_name]`) from `./benchmark_results/` for specified SAE variants.

* Implementation Rules:
    * To maintain compatibility with the existing training and evaluation scripts, do not remove any of the input variables, and do not add or remove any of the output variables of the classes or methods.
    * Focus on streamlined changes--only change what is necessary to implement the variant. This might mean just changing a single line or method.
    * If necessary, you can add new properties to the init, e.g. a new threshold or scalar, but make sure the original properties are all inherited.
    * The core scripts (`train_sae.py`, `benchmark_sae.py`, `print_metrics.py`), the `utils.py` file, and any underlying source code within the `dictionary_learning` or `sae-bench` libraries should **not** be modified. Focus your changes exclusively within the custom SAE files in the `./custom_saes/` directory.
    * However, you should reference the source code if you run into any issues with your custom variant in order to understand the reason the variant is not working for a specific step.

The result should be an implementation of the described model variant localized entirely to a single `custom_saes/[variant_name].py` file, which has been successfully trained and benchmarked, with the benchmark results successfully printed.