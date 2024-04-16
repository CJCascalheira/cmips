import os
import json
import pandas as pd
from bs4 import BeautifulSoup

def parse_your_posts_html(html_file_path, profile_id):
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    post_data = []

    for comment_div in soup.find_all('div', class_='_3-95 _a6-g'):
        title_div = comment_div.find('div', class_='_2ph_ _a6-h _a6-i')
        if title_div:
            title = title_div.text.strip()
        else:
            title = " "
            
        post_divs = [div for div in comment_div.find_all('div', class_='_2pin') if not div.find(class_='_a7nf')]  # Exclude divs with class '_a7nf'
        post = " ".join(post_div.text.strip() for post_div in post_divs)
        
        timestamp = comment_div.find('div', class_='_a72d').text.strip()
        
        data = {
            'participant_id': profile_id,
            'title': title,
            'timestamp':timestamp,
            'post': post,
        }

        post_data.append(data)

    return pd.DataFrame(post_data)


def parse_comments_html(html_file_path, profile_id):
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    comments_data = []

    for comment_div in soup.find_all('div', class_='_3-95 _a6-g'):
        comment_info = comment_div.find('div', class_='_2ph_ _a6-h _a6-i').text.strip()
        comment_author = extract_author(comment_info)
        comment_content_div = comment_div.find('div', class_='_2pin')
        if comment_content_div:
            comment_content = comment_content_div.text.strip()
        else:
            comment_content = " "
        comment_timestamp = comment_div.find('div', class_='_a72d').text.strip()

        data = {
            'participant_id': profile_id,
            'title': comment_info,
            'comment_timestamp': comment_timestamp,
            'comment_author': comment_author,
            'comment_content': comment_content,
        }

        comments_data.append(data)

    return pd.DataFrame(comments_data)

def extract_author(comment_info):
    # Example: "El McCabe commented on Claire Misko's post."
    if ' commented on ' in comment_info:
        parts = comment_info.split(' commented on ')
    elif ' replied to ' in comment_info:
        parts = comment_info.split(' replied to ')
    else:
        parts = [comment_info, '']  
    return parts[0].strip()

# Define the root folder path
root_folder_path = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'

# Initialize an empty list to store profile IDs
profile_ids = []

# Initialize an empty DataFrame to store the html content and related information
post_data = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'description', 'post'])

comment_data = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'comment_timestamp', 'comment_content', 'comment_author'])

# Define a function to extract profile ID from folder name
def extract_profile_id(folder_name):
    parts = folder_name.split('_')
    if len(parts) >= 2:
        return f"CMIPS_{parts[1]}"
    else:
        return None

# Loop through folders
for folder_name in os.listdir(root_folder_path):
    folder_path = os.path.join(root_folder_path, folder_name)
    if os.path.isdir(folder_path) and any(folder_name.endswith(suffix) for suffix in ['Facebook', 'Facebook1', 'Facebook2','Facebook3','Facebook-1','Facebook-2']):
        # Extract and store the profile ID
        profile_id = extract_profile_id(folder_name)


        subfolders = ['comments_and_reactions', 'posts', 'your_activity_across_facebook']
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if subfolder == 'posts' and os.path.exists(os.path.join(subfolder_path, 'your_posts_1.html')):
                print("Found 'your_posts_1.html' in:", subfolder_path)
                your_posts_html_file_path = os.path.join(subfolder_path, 'your_posts_1.html')
                data_to_append_df = parse_your_posts_html(your_posts_html_file_path, str(profile_id))
                post_data = post_data.append(data_to_append_df, ignore_index=True)
    
                
            elif subfolder == 'comments_and_reactions' and os.path.exists(os.path.join(subfolder_path, 'comments.html')):
                print("Found 'comments.html' in:", subfolder_path)
                comments_html_file_path = os.path.join(subfolder_path, 'comments.html')
                data_to_append_df = parse_comments_html(comments_html_file_path, str(profile_id))
                # Append data_to_append to json_data
                comment_data = comment_data.append(data_to_append_df, ignore_index=True)
                
            if subfolder == 'your_activity_across_facebook':
                subsubfolders = ['comments_and_reactions', 'posts']
                folder_path = os.path.join(folder_path, subfolder)
                for subsubfolder in subsubfolders:   
                    subsubfolder_path = os.path.join(folder_path, subsubfolder)
                    if subsubfolder == 'posts' and os.path.exists(os.path.join(subsubfolder_path, 'your_posts_1.json')):
                        print("Found 'your_posts_1.json' in:", subsubfolder_path)
                        your_posts_html_file_path = os.path.join(subsubfolder_path, 'your_posts_1.json')
                        data_to_append_df = parse_your_posts_html(your_posts_html_file_path, str(profile_id))
                        post_data = json_data.append(data_to_append_df, ignore_index=True)
                    elif subsubfolder == 'comments_and_reactions' and os.path.exists(os.path.join(subsubfolder_path, 'comments.json')):
                        print("Found 'comments.json' in:", subsubfolder_path)
                        comments_html_file_path = os.path.join(subsubfolder_path, 'comments.json')
                        data_to_append_df = parse_comments_html(comments_html_file_path, str(profile_id))
                        # Append data_to_append to json_data
                        comment_data = comment_data.append(data_to_append_df, ignore_index=True)
       
        else:
            continue  # Go to the next item in the outer loop

post_data.to_csv('facebook_posts_html_data.csv',index=False)
comment_data.to_csv('facebook_comments_html_data.csv',index=False)





