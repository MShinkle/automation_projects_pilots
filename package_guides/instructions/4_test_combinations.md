**Standalone Task: Implementing a Multi-Package Experiment Idea**

**Objective:**  
Implement one of the 10 brainstormed experiment ideas by leveraging the guides provided for all 5 packages. This task tests your ability to combine functionalities from multiple packages while also evaluating whether the guides contain sufficient information to implement a real experiment.

---

**Instructions:**

1. **Reference Guides Only:**  
   - Base your implementation entirely on the guides provided for the 5 packages.  
   - Do **not** read through the source code or previous test scripts unless absolutely necessary for debugging. This evaluation aims to determine if the guides alone provide enough detail for implementation.

2. **Select and Implement an Experiment Idea:**  
   - Choose one of the 10 experiment ideas that combines features from at least 2 packages.  
   - Develop a minimal, functional implementation plan using only the information from the guides.  
   - Ensure the experiment is realistic and uses manageable datasets or models (avoid large-scale setups).

3. **Project Status Updates:**  
   - **Initial Status:** Each experiment idea initially has a "Status" set to **"Not Implemented"**.  
   - **During Implementation:** Before beginning your work, update the status for the project you are working on to **"In Progress"**.  
   - **After Successful Completion:** If the task is successfully implemented in full, update the status to **"Completed"**.

4. **Testing and Verification:**  
   - Create test scripts for your new project under the directory `test_projects/[project_name]/` to verify that the implementation works as expected.  
   - Do not modify the original package source code; rely strictly on built-in functionality and the guides’ instructions.  
   - Update the guides based on any issues encountered during testing.

5. **Integration Troubleshooting:**  
   - If you run into issues integrating functionalities from multiple packages, add an **Integration with Other Packages** section at the bottom of the implementation guide.  
   - This section should include:  
     - **Description:** A brief summary of the integration challenge.  
     - **Integration Tips:** Recommendations and adjustments to resolve the issue.  
     - **Debugging Steps:** Specific actions taken to diagnose and fix the integration problems.

6. **Documentation Updates:**  
   - Actively debug and update the **Troubleshooting** and **Implementation Tips** sections in the guide with any new findings.  
   - Document any integration challenges and your resolutions in the **Integration with Other Packages** section.
   - You updates to the guides should be driven by this core question: what would have needed to be different in the guide(s) for you to have not made any mistakes or errors you did during your implementation? This should result in updates to the guides which will avoid the same issues in the future.

7. **Finalization:**  
   - Once the experiment is verified in your live testing environment, finalize the implementation guide for the experiment.  
   - Ensure all code snippets are verified and include references to the specific test scripts (for example, “Verified via `test_projects/[project_name]/experiment_test.py`”).  
   - Reiterate that testing was performed using only the built-in package functionality as described in the guides.  
   - Update the project status from "In Progress" to "Completed" only if the task is successfully implemented in full.

---

**Key Reminders:**

- **Guide Reliance:** The primary resource is the guides for the 5 packages. Do not rely on external source code or previous test scripts unless absolutely necessary for debugging.  
- **Debugging Protocol:** Only refer to source code or older tests as a last resort; the goal is to assess and refine the guides if integration issues are uncovered.  
- **Integration Section:** If functionalities do not integrate seamlessly, explicitly add an **Integration with Other Packages** section detailing your observations, tips, and resolutions.  
- **Testing Location:** All tests for the new project should reside under `test_projects/[project_name]/`.  
- **No Source Code Edits:** Do not modify the package source code; use built-in functionality as specified in the guides.  
- **Project Status Management:** Remember to update the project's status according to the implementation progress:  
  - Initially: **"Not Implemented"**  
  - While working: **"In Progress"**  
  - Upon full, successful implementation: **"Completed"**
- **Complete the Full Project Autonomously:** Do not pause to ask for approval or suggestions at any point unless absolutely necessary.

---

By following these instructions, you will implement a full research experiment using multiple packages while contributing to the refinement of the guides, which will have been updated as appropriate such that if another model tried to perform this task in the future, they would likely avoid any issues you encountered along the way.