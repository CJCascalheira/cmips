#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import zipfile

def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        try:
            zf.extractall(dest_dir)
            print(f"Extracted: {source_filename} to {dest_dir}")
        except Exception as e:
            print(f"Error extracting {source_filename}: {e}")
            return source_filename

def extract_zip(zip_folder, extract_to):
    error_files = []  # List to keep track of error files
    # Create the directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Loop through all files in the zip folder
    for file_name in os.listdir(zip_folder):
        if file_name.endswith('.zip'):  # Check if file is a zip file
            zip_file_path = os.path.join(zip_folder, file_name)
            folder_name = os.path.splitext(file_name)[0]  # Extract folder name without the .zip extension
            extract_folder = os.path.join(extract_to, folder_name)  # Path to extract folder

            # Create the folder if it doesn't exist
            if not os.path.exists(extract_folder):
                os.makedirs(extract_folder)
                
            error_file = unzip(zip_file_path, extract_folder)
            if error_file:
                error_files.append(error_file)

    if error_files:
        print("Files that encountered errors during extraction:")
        for error_file in error_files:
            print(error_file)

# Path to the folder containing zip files
zip_folder = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\ZIP'
# Path to the folder where extracted contents will be saved
extract_to = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'

# Call the function to extract zip files
extract_zip(zip_folder, extract_to)


# In[ ]:




