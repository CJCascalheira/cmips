#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import json
import pandas as pd

root_folder_path = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'

def check_for_html_files(folder_path):
    if os.path.exists(folder_path):
        html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
        return len(html_files) > 0
    else:
        return False

for folder_name in os.listdir(root_folder_path):
    folder_path = os.path.join(root_folder_path, folder_name)
    if os.path.isdir(folder_path) and folder_name.endswith('Facebook'):
        subfolders = ['comments_and_reactions', 'posts']
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if subfolder == 'posts':
                if check_for_html_files(subfolder_path):
                    print("HTML files found in:", subfolder_path)
            elif subfolder == 'comments_and_reactions':
                if check_for_html_files(subfolder_path):
                    print("HTML files found in:", subfolder_path)               
        else:
            continue  # Go to the next item in the outer loop
            
    if os.path.isdir(folder_path) and folder_name.endswith('Twitter'):
        subfolders = ['data']
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if subfolder == 'data':
                if check_for_html_files(subfolder_path):
                    print("Found html in:", subfolder_path)
        else:
            continue 

