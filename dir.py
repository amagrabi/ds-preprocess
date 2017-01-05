#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper functions to process directories and files.

@author: amagrabi
"""

import os
import shutil
import magic

def delete_path(path):
    shutil.rmtree(path)
    
def create_folders(folder_list, path):
    for folder in folder_list:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

def print_filetypes(path):
    for path, subdirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(path, file)
            print(magic.from_file(filepath))