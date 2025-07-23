# Example RepoMix Usage

This directory contains two things: 1) a script for converting Jupyter notebooks (`.ipynb` files) to Python (`.py`) files in place, preserving the original directory structure, and 2) a repomix config json tweaked for our application. 

## Basic Usage

First convert notebooks in a directory:
```bash
python move_and_convert_files.py [package_directory]
```

This will allow repomix to add them into the summary file without all the ipynb formatting and outputs. Next, compile a summary of all the .py, .md, and .txt files in the repo into a single output file via repomix. NOTE: IT IS CRITICAL THAT THIS IS RUN FROM THE SAME DIRECTORY AS YOUR `repomix.config.json` FILE.

```bash
npx repomix [package_directory] -o [package_name].txt
```

## Requirements

- Python 3.6+
- `nbconvert` (for Jupyter notebook conversion)
- `npx` and `repomix` (for documentation generation)

## Configuration

The directory includes a `repomix.config.json` file that can be used to configure how repomix processes the codebase. Key configuration options include:

- `output.style`: Output format (default: "plain")
- `output.compress`: Whether to remove empty lines and comments
- `output.fileSummary`: Include file summaries
- `output.directoryStructure`: Include directory structure information

You can control which files are included in the output through the following configuration options:

- `include`: List of glob patterns for files to include (default: `["**/*.py"]`)
- `ignore`: Configuration for excluding files:
  - `useGitignore`: Whether to respect .gitignore (default: false)
  - `useDefaultPatterns`: Whether to use default exclusion patterns (default: false)
  - `customPatterns`: List of glob patterns for files to exclude (default: `["**/.git/**", "**/.github/**"]`)
