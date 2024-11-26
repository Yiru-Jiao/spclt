'''
This script is used to clean the __pycache__ folders in the current directory.
'''

import os
import shutil

# Define the directory to start searching from (e.g., current working directory)
root_dir = os.getcwd()

# Walk through all directories and files in the repository
for dirpath, dirnames, filenames in os.walk(root_dir):
    # If a __pycache__ directory is found, remove it
    if '__pycache__' in dirnames:
        cache_path = os.path.join(dirpath, '__pycache__')
        shutil.rmtree(cache_path)
        print(f"Removed {cache_path}")
