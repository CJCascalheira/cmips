#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
##This function list the contents of each directory and then compare the contents to find any differences. 
def compare_directories(dir1, dir2):
    dir1_contents = set(os.listdir(dir1))
    dir2_contents = set(os.listdir(dir2))
    
    missing_in_dir1 = dir2_contents - dir1_contents
    missing_in_dir2 = dir1_contents - dir2_contents
    
    if missing_in_dir1:
        print(f"Files/folders missing in {dir1}:")
        for item in missing_in_dir1:
            print(item)
    else:
        print("No files/folders missing in", dir1)
    
    if missing_in_dir2:
        print(f"\nFiles/folders missing in {dir2}:")
        for item in missing_in_dir2:
            print(item)
    else:
        print("No files/folders missing in", dir2)

# Paths to the directories to be compared
dir1_path = r'C:\Users\Ankit Chapagain\A'
dir2_path = r'C:\Users\Ankit Chapagain\B'

# Call the function to compare directories
compare_directories(dir1_path, dir2_path)

