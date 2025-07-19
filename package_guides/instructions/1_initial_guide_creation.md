**Overall Objective:**  
Generate a comprehensive, verified, and well-organized reference document for the specified Python package. The final document must be created as a markdown file in the directory `package_guides/[package_name]_guide.md`, and every code example included must be executed and verified in a live testing environment using efficient, real components. Always opt for built-in package functionality instead of custom solutions unless absolutely necessary, and reference the source code and official documentation frequently to ensure that implementations are as streamlined as possible. Include sections on implementation tips, variants, and troubleshooting.

---

**Phase 1: Survey and Initial Document Creation**

1. **Create the Documentation File:**  
   - Create a markdown file named `package_guides/[package_name]_guide.md`.
   - Thoroughly survey the package's source code, official documentation, demos, and examples.
   - Compile a complete list of all functionalities (use cases) that the package offers.

2. **Outline Capabilities:**  
   - List each identified function or use case as high-level tasks (e.g., "Training sparse autoencoders," "Loading pretrained models," "Evaluating models").
   - Use clear markdown headers (e.g., `## <Task Name>`) to demarcate each high-level task.

*Example for Phase 1 (Outline):*  
```markdown
# SAELens Package Examples and Usage

This document provides a comprehensive guide to using the SAELens package, covering key functionalities.

## Table of Contents
1. [Loading Pre-trained SAEs](#loading-pre-trained-saes)
2. [Training Sparse Autoencoders](#training-sparse-autoencoders)
3. [Analyzing SAE Features](#analyzing-sae-features)
4. [Evaluating SAE Performance](#evaluating-sae-performance)
5. [Advanced Analysis Tools](#advanced-analysis-tools)
```

---

**Phase 2: Breaking Down Tasks into Subtasks**

3. **Define Subtasks for Each Use Case:**  
   - For every high-level task, break it down into minimal, sequential subtasks.
   - Create a dedicated section for each subtask using markdown headers (e.g., `### <Subtask Name>`).

4. **Subtask Section Format:**  
   Each subtask section should include:
   - **Description:**  
     A concise, purpose-focused summary of the subtask.
   - **Prerequisites:**  
     A bullet list of required conditions or parameters (e.g., dataset identifiers, split names, model names).
   - **Expected Outcome:**  
     A clear statement of what a successful execution should produce.
   - **Tested Code Implementation:**  
     - **Step 1: Create and Execute Test Script:**  
       Immediately create a minimal Python test script (e.g., `./tests/[package_name]/test_<subtask_name>.py`) that leverages built-in package functionality—avoid custom workarounds unless absolutely necessary. Include the necessary imports, code snippet, and print statements or assertions to verify the expected outcome. Execute this test script in your live testing environment.
     - **Step 2: Document Verified Code:**  
       Once the test script runs successfully, update this documentation section with the verified code snippet. Clearly reference the test file used (e.g., "Verified via `./tests/[package_name]/test_load_dataset.py`").  
       **Important:** Use efficient, real components as implemented in the package; always reference the source code and official documentation to streamline your implementation.
   - **Implementation Tips:**  
     - Provide additional best practices or insights on effectively implementing the subtask, including recommendations to use built-in functionality whenever possible.
   - **Variants:**  
     - List possible variations (e.g., different dataset identifiers, alternate batch sizes, toggling options) and briefly explain when or why each might be useful.
   - **Troubleshooting:**  
     - Include common pitfalls and troubleshooting suggestions based on your testing experience.

*Example for Phase 2 (Subtask - Loading a Dataset):*  

---

### Load a Dataset from Hugging Face

**Description:**  
Load a dataset using Hugging Face's datasets library, convert necessary columns to PyTorch tensors, and create a DataLoader for batched iteration.

**Prerequisites:**  
- A valid dataset identifier (e.g., `"imdb"`).  
- A desired data split (e.g., `"train"`).  
- Configuration parameters such as batch size and column selection.

**Expected Outcome:**  
- A PyTorch DataLoader is created successfully.  
- Retrieving one batch from the DataLoader works without errors and includes expected keys (e.g., `"text"`, `"label"`).

**Tested Code Implementation:**  
_First, create and execute the test script `./tests/[package_name]/test_load_dataset.py` in your live testing environment. Once verified (using the built-in dataset functionality, without any dummy components), update this section with the confirmed code snippet:_
```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load the "imdb" training dataset.
dataset = load_dataset("imdb", split="train")

# Convert specified columns to PyTorch tensors.
dataset.set_format(type="torch", columns=["text", "label"])

# Create a DataLoader for batching; batch size is 32.
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Retrieve one batch to verify functionality.
batch = next(iter(dataloader))
# Verified via ./tests/[package_name]/test_load_dataset.py – DataLoader successfully created; batch contains keys 'text' and 'label'.
```

**Implementation Tips:**  
- Use the built-in `set_format` method to ensure that the dataset returns PyTorch tensors for easier downstream processing.
- Reference the package's source code and official documentation frequently to ensure that your approach is as streamlined as possible.
- Avoid custom solutions; opt for the package's native functionality unless no alternative exists.

**Variants:**  
- **Dataset Identifier and Split:** Substitute `"imdb"` with another valid dataset or change the split to `"test"` or `"validation"`.  
- **Batch Size:** Adjust the `batch_size` to suit memory constraints or performance needs.  
- **Shuffling:** Disable shuffling (`shuffle=False`) if deterministic output is required.

**Troubleshooting:**  
- **Dataset Errors:** Verify that the dataset identifier is correct and available in the Hugging Face registry.  
- **Data Format Issues:** Confirm that the specified columns exist in the dataset and that `set_format` converts them properly.  
- **Memory Constraints:** If memory issues arise, reduce the batch size accordingly.

---

**Phase 3: Iterative Testing-First Development**

5. **Test Each Subtask via Execution Tool:**  
   - For each subtask, immediately create and execute the corresponding test script as described above.  
   - Analyze `stdout` and `stderr`; if errors occur, debug and adjust the test script until the expected outcome is achieved.  
   - Update the documentation markdown with the verified code snippet only after successful test execution.
   - **Note on Blocked Prerequisites:** If testing a specific functionality (`Target Functionality`) is blocked by repeated failures in a prerequisite step (`Prerequisite Step`) involving external resources (e.g., downloading data/models, complex configurations):
     a. Briefly attempt obvious alternative methods for the `Prerequisite Step`.
     b. If still blocked, prioritize verifying the `Target Functionality` by temporarily modifying the setup (e.g., switching to simpler/standard inputs, mocking the prerequisite's output if appropriate).
     c. Clearly document the issue with the original `Prerequisite Step` and the workaround used in the test script and documentation (e.g., under 'Troubleshooting').

6. **Integrated Testing of All Subtasks:**  
   - Create an integrated demo script under the `./tests/[package_name]` directory (e.g., `./tests/[package_name]/integrated_pipeline_demo.py`) that combines all verified subtasks into a complete pipeline.  
   - Execute this integrated demo script in your live testing environment. If it fails, debug the integration (referring back to the individual test scripts) and repeat testing until the entire pipeline works as expected.

---

**Phase 4: Integrated Demo and Finalization**

7. **Create an Integrated Demo Script:**  
   - Develop an integrated demo script (e.g., `./tests/[package_name]/integrated_pipeline_demo.py`) that brings together all the verified code snippets into a complete and functional pipeline.  
   - Execute this integrated demo script in your live testing environment to ensure that all components work together seamlessly.  
   - If the integrated demo fails, debug it using insights from the individual test scripts and re-run until it passes.

8. **Finalize the Documentation:**  
   - Update each subtask section with any additional notes or improvements based on the integrated demo testing.  
   - Ensure that the final document maintains consistent formatting and includes all necessary import statements, either consolidated at the beginning or appropriately distributed within sections.  
   - Clearly indicate any known limitations or further testing recommendations for specific use cases.

---

**Final Notes:**  
- **Mandatory Testing:** All code snippets in the final documentation MUST originate from test scripts that have been executed successfully in your live environment. Do not include any code that hasn't been verified.
- **Live Environment Verification:** Execute all tests in your designated live environment, ensuring that all required dependencies are installed. Always use efficient, built-in package functionality rather than dummy or custom components unless absolutely necessary.
- **Modularity and Clarity:** Use clear, concise language to explain both the rationale and the implementation details of each code snippet.
- **Troubleshooting:** Include common pitfalls and troubleshooting suggestions based on your actual testing experience.
- **Autonomous Operation:** Proceed through all phases and subtasks without waiting for external feedback until the final documentation is complete; assume that the initial plan is approved.
- **Configuration Verification:** When debugging library functions whose behavior seems inconsistent with provided configuration files (e.g., YAML, JSON), environment variables, or internal mappings:
    a. Double-check the configuration source itself (file content, variable value).
    b. **Crucially, investigate the library's source code to understand precisely how it loads, parses, caches, and utilizes that specific configuration or mapping.** Pay attention to file paths, parsing logic, defaults, and caching.
    c. Note discrepancies between documented behavior and the actual implementation's loading/parsing logic, as these are common error sources.

By following these revised instructions—with integrated examples that illustrate each phase (from initial outline through individual subtask tests and integrated demo testing)—the final documentation will include detailed, structured, and clear reference material with verified code examples, implementation tips, variants, and troubleshooting. This process ensures that all examples are built with efficient, real package functionality, with frequent references to the source code and official documentation to keep implementations streamlined.