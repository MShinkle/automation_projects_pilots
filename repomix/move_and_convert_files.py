#!/usr/bin/env python3
import os
import sys
import subprocess

def convert_notebooks(source_dir):
    """
    Walk through source_dir and convert any .ipynb file found to .py files
    in the same directory. This version invokes nbconvert as a module using
    the current Python interpreter.
    """
    source_dir = os.path.abspath(source_dir)
    for dirpath, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(dirpath, file)
                base_name = os.path.splitext(file)[0]
                command = [
                    sys.executable, "-m", "nbconvert", "--to", "script",
                    notebook_path,
                    "--output", base_name,
                    "--output-dir", dirpath
                ]
                print(f"Converting notebook: {notebook_path}")
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"Error converting {notebook_path}:\n{result.stderr}")
                else:
                    print(result.stdout)

def main():
    if len(sys.argv) != 2:
        print("Usage: python move_and_convert_files.py <source_directory>")
        sys.exit(1)
    
    source_directory = sys.argv[1]
    
    if not os.path.isdir(source_directory):
        print(f"Error: {source_directory} is not a valid directory.")
        sys.exit(1)
    
    print("Converting .ipynb files to .py files in place...")
    convert_notebooks(source_directory)
    
    print(f"\nAll notebooks have been converted to Python files in their original locations.")

if __name__ == "__main__":
    main() 