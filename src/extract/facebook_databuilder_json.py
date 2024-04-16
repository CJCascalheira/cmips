import os
import json
import pandas as pd

def parse_your_posts_1_to_dataframe(json_file, profile_id):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize lists to store extracted data
    timestamp_list = []
    creation_timestamp_list = []
    title_list = []
    description_list = []
    post_list = []

    # Iterate over each entry in the JSON data
    for entry in data:
        # Extract post from 'data'
        if 'data' in entry and entry['data']:
            post = entry.get('data', [{}])[0].get('post')
            post_list.append(post)

     
        # Extract attachments
        attachments = entry.get('attachments', [])
        for attachment in attachments:
            attachment_data = attachment.get('data', [])
            for item in attachment_data:
                # Extract attributes from 'media'
                media = item.get('media', {})
                creation_timestamp_list.append(media.get('creation_timestamp'))
                title_list.append(media.get('title'))
                description_list.append(media.get('description'))
                # Append timestamp (assuming it's the same for all attachments in an entry)
                timestamp_list.append(entry['timestamp'])

    # Fill missing values with None
    max_length = max(len(timestamp_list), len(creation_timestamp_list),
                     len(title_list), len(description_list), len(post_list))
    timestamp_list.extend([None] * (max_length - len(timestamp_list)))
    creation_timestamp_list.extend([None] * (max_length - len(creation_timestamp_list)))
    title_list.extend([None] * (max_length - len(title_list)))
    description_list.extend([None] * (max_length - len(description_list)))
    post_list.extend([None] * (max_length - len(post_list)))

    # Create DataFrame
    df = pd.DataFrame({
        'participant_id': profile_id,
        'timestamp': timestamp_list,
        'creation_timestamp': creation_timestamp_list,
        'title': title_list,
        'description': description_list,
        'post': post_list
    })

    return df



def parse_comments_json(json_file, profile_id):
    with open(json_file, 'r') as file:
        data = json.load(file)

    comments_data = []

    for entry_key in data.keys():
        if entry_key == 'comments_v2':
            for entry in data.get(entry_key, []):
                timestamp = entry.get('timestamp')
                title = entry.get('title')

                for comment_entry in entry.get('data', []):
                    comment = comment_entry.get('comment', {})
                    comment_timestamp = comment.get('timestamp')
                    comment_content = comment.get('comment')
                    comment_author = comment.get('author')

                    comment_data = {
                        'participant_id': profile_id,
                        'timestamp': timestamp,
                        'title': title,
                        'comment_timestamp': comment_timestamp,
                        'comment_content': comment_content,
                        'comment_author': comment_author
                    }

                    comments_data.append(comment_data)

    return pd.DataFrame(comments_data)



# Define the root folder path
root_folder_path = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'

# Initialize an empty list to store profile IDs
profile_ids = []

# Initialize an empty DataFrame to store the html content and related information
json_data = pd.DataFrame(columns=['participant_id', 'timestamp',  'creation_timestamp', 'title', 'description', 'post'])

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
        if profile_id:
            profile_ids.append(profile_id)
        
        # Look for specific subfolders
        subfolders = ['comments_and_reactions', 'posts', 'your_activity_across_facebook']
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if subfolder == 'posts' and os.path.exists(os.path.join(subfolder_path, 'your_posts_1.json')):
                print("Found 'your_posts_1.json' in:", subfolder_path)
                ###PARSE your_posts_1.json
                json_file_path = os.path.join(subfolder_path, 'your_posts_1.json')
                data_to_append_df = parse_your_posts_1_to_dataframe(json_file_path, str(profile_id))
                 # Append data_to_append to json_data
                json_data = json_data.append(data_to_append_df, ignore_index=True)
                
            elif subfolder == 'comments_and_reactions' and os.path.exists(os.path.join(subfolder_path, 'comments.json')):
                print("Found 'comments.json' in:", subfolder_path)
                comments_json_file_path = os.path.join(subfolder_path, 'comments.json')
                data_to_append_df = parse_comments_json(comments_json_file_path, str(profile_id))
                # Append data_to_append to json_data
                comment_data = comment_data.append(data_to_append_df, ignore_index=True)
            
            if subfolder == 'your_activity_across_facebook':
                subsubfolders = ['comments_and_reactions', 'posts']
                folder_path = os.path.join(folder_path, subfolder)
                for subsubfolder in subsubfolders:   
                    subsubfolder_path = os.path.join(folder_path, subsubfolder)
                    if subsubfolder == 'posts' and os.path.exists(os.path.join(subsubfolder_path, 'your_posts_1.json')):
                        print("Found 'your_posts_1.json' in:", subsubfolder_path)
                        json_file_path = os.path.join(subsubfolder_path, 'your_posts_1.json')
                        data_to_append_df = parse_your_posts_1_to_dataframe(json_file_path, str(profile_id))
                        json_data = json_data.append(data_to_append_df, ignore_index=True)
                    elif subsubfolder == 'comments_and_reactions' and os.path.exists(os.path.join(subsubfolder_path, 'comments.json')):
                        print("Found 'comments.json' in:", subsubfolder_path)
                        comments_json_file_path = os.path.join(subsubfolder_path, 'comments.json')
                        data_to_append_df = parse_comments_json(comments_json_file_path, str(profile_id))
                        # Append data_to_append to json_data
                        comment_data = comment_data.append(data_to_append_df, ignore_index=True)


                
                  
        else:
            continue  # Go to the next item in the outer loop


print("Profile IDs:", profile_ids)

json_data.to_csv('facebook_posts_json_data.csv',index=False)
comment_data.to_csv('facebook_comments_json_data.csv',index=False)


