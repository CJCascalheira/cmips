import os
import json
import pandas as pd
from bs4 import BeautifulSoup


##JSON FUNCTIONS
def parse_your_posts_1_to_dataframe(json_file, profile_id):
    with open(json_file, 'r') as file:
        data = json.load(file)

    posts_data = []
    for entry in data:
        timestamp = entry.get('timestamp')
        
        post = None
        update_timestamp = None
        # Extract post and update_timestamp from 'data'
        if 'data' in entry and entry['data']:
            for item in entry['data']:
                if 'post' in item:
                    post = item['post']
                if 'update_timestamp' in item:
                    update_timestamp = item['update_timestamp']
        
        # Initialize variables
        creation_timestamp = None
        media_title = None
        media_description = None
        title = None
        
        if not title:
            title = entry.get('title')
        
        # Extract attachments
        attachments = entry.get('attachments', [])
        if attachments:
            for attachment in attachments:
                attachment_data = attachment.get('data', [])
                for item in attachment_data:
                    media = item.get('media', {})
                    media_creation_timestamp = media.get('creation_timestamp')
                    media_title = media.get('title')
                    media_description = media.get('description')
                    
                    df = {
                        'participant_id': profile_id,
                        'timestamp': timestamp,
                        'media_creation_timestamp': media_creation_timestamp,
                        'media_title': media_title,
                        'media_description': media_description
                    }

                    posts_data.append(df)
                    
        df = {
            'participant_id': profile_id,
            'timestamp': timestamp,
            'update_timestamp': update_timestamp,
            'post': post,
            'title': title
        }

        posts_data.append(df)
            

    return pd.DataFrame(posts_data)



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


def parse_likes_and_reactions_json(json_file, profile_id):
    with open(json_file, 'r') as file:
        data = json.load(file)

    reactions_data = []

    if isinstance(data, list):  # Handle case when root is a list
        for item in data:
            if isinstance(item, dict):
                reactions_data.extend(process_dict(item, profile_id))
    elif isinstance(data, dict):  # Handle case when root is a dictionary
        reactions_v2_data = data.get('reactions_v2')
        if reactions_v2_data:  # Check if reactions_v2 key exists
            for item in reactions_v2_data:
                if isinstance(item, dict):
                    reactions_data.extend(process_dict(item, profile_id))

    return pd.DataFrame(reactions_data)

def process_dict(data, profile_id):
    reactions_data = []

    timestamp = data.get('timestamp')
    title = data.get('title')

    for reaction_entry in data.get('data', []):
        reaction_json = reaction_entry.get('reaction', {})
        reaction = reaction_json.get('reaction')
        actor = reaction_json.get('actor')

        reaction_data = {
            'participant_id': profile_id,
            'timestamp': timestamp,
            'title': title,
            'reaction': reaction,
            'actor': actor
        }

        reactions_data.append(reaction_data)

    return reactions_data


# HTML Functions
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
    elif ' liked ' in comment_info:
        parts = comment_info.split(' liked ')
    elif ' reacted ' in comment_info:
        parts = comment_info.split(' reacted ')
    else:
        parts = [comment_info, '']  
    return parts[0].strip()



def parse_likes_and_reactions_html(html_file_path, profile_id):
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    comments_data = []

    for reaction_div in soup.find_all('div', class_='_3-95 _a6-g'):
        reaction_info = reaction_div.find('div', class_='_2ph_ _a6-h _a6-i').text.strip()
        actor = extract_author(reaction_info)
        reaction_content_div = reaction_div.find('div', class_='_2pin')
        if reaction_content_div:
            reaction_content = reaction_content_div.find('img')['src']
        else:
            reaction_content = " "
        timestamp = reaction_div.find('div', class_='_a72d').text.strip()

        data = {
            'participant_id': profile_id,
            'timestamp': timestamp,
            'title': reaction_info,
            'reaction': reaction_content,
            'actor': actor   
        }

        comments_data.append(data)

    return pd.DataFrame(comments_data)


# Define the root folder path
root_folder_path = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'

# Initialize an empty list to store profile IDs
profile_ids = []

# Initialize an empty DataFrame to store the html content and related information
post_data_json = pd.DataFrame(columns=['participant_id', 'timestamp',  'media_creation_timestamp', 'media_title', 'media_description', 'update_timestamp' 'post', 'title'])
comment_data_json = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'comment_timestamp', 'comment_content', 'comment_author'])
reaction_data_json = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'reaction','actor']) #####


post_data_html = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'description', 'post'])
comment_data_html = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'comment_timestamp', 'comment_content', 'comment_author'])
reaction_data_html = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'reaction','actor']) 

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
            #print(subfolder_path)
            
            if subfolder == 'posts' and os.path.exists(os.path.join(subfolder_path, 'your_posts_1.json')):
                print("Found 'your_posts_1.json' in:", subfolder_path)
                ###PARSE your_posts_1.json
                json_file_path = os.path.join(subfolder_path, 'your_posts_1.json')
                data_to_append_df = parse_your_posts_1_to_dataframe(json_file_path, str(profile_id))
                 # Append data_to_append to json_data
                post_data_json = post_data_json.append(data_to_append_df, ignore_index=True)
                
            if subfolder == 'posts' and os.path.exists(os.path.join(subfolder_path, 'your_posts_1.html')):
                print("Found 'your_posts_1.html' in:", subfolder_path)
                your_posts_html_file_path = os.path.join(subfolder_path, 'your_posts_1.html')
                data_to_append_df = parse_your_posts_html(your_posts_html_file_path, str(profile_id))
                post_data_html = post_data_html.append(data_to_append_df, ignore_index=True)
            
            if subfolder == 'comments_and_reactions' and os.path.exists(subfolder_path):
                if os.path.exists(os.path.join(subfolder_path, 'comments.html')):
                    print("Found 'comments.html' in:", subfolder_path)
                    comments_hmtl_file_path = os.path.join(subfolder_path, 'comments.html')
                    data_to_append_df = parse_comments_html(comments_hmtl_file_path, str(profile_id))
                    # Append data_to_append to json_data
                    comment_data_html = comment_data_html.append(data_to_append_df, ignore_index=True)
                    
                if os.path.exists(os.path.join(subfolder_path, 'comments.json')):
                    print("Found 'comments.json' in:", subfolder_path)
                    comments_json_file_path = os.path.join(subfolder_path, 'comments.json')
                    data_to_append_df = parse_comments_json(comments_json_file_path, str(profile_id))
                    # Append data_to_append to json_data
                    comment_data_json = comment_data_json.append(data_to_append_df, ignore_index=True)
                    
                for filename in os.listdir(subfolder_path):
                    #print(filename)
                    if filename.endswith('.json'):
                        #anyfile name other than comments.json is likes and reaction
                        if os.path.exists(os.path.join(subfolder_path, filename)) and filename != 'comments.json':
                            print(filename, "found in",subfolder_path )
                            reaction_json_file_path = os.path.join(subfolder_path, filename)
                            data_to_append_df = parse_likes_and_reactions_json(reaction_json_file_path, str(profile_id))
                            # Append data_to_append to json_data
                            reaction_data_json = reaction_data_json.append(data_to_append_df, ignore_index=True)
    
                    if filename.endswith('.html'):
                        #anyfile name other than comments.html is likes and reaction
                        if os.path.exists(os.path.join(subfolder_path, filename)) and filename != 'comments.html':
                            print(filename, "found in",subfolder_path )
                            reaction_html_file_path = os.path.join(subfolder_path, filename)
                            data_to_append_df = parse_likes_and_reactions_html(reaction_html_file_path, str(profile_id))
                            # Append data_to_append to json_data
                            reaction_data_html = reaction_data_html.append(data_to_append_df, ignore_index=True)
                            
            #there are some files inside this folder                
            if subfolder == 'your_activity_across_facebook' and os.path.exists(subfolder_path):
                subsubfolders = ['comments_and_reactions', 'posts']
                folder_path = os.path.join(folder_path, subfolder)
                for subsubfolder in subsubfolders:   
                    subsubfolder_path = os.path.join(folder_path, subsubfolder)
                    if subsubfolder == 'posts' and os.path.exists(os.path.join(subsubfolder_path, 'your_posts_1.json')):
                        print("Found 'your_posts_1.json' in:", subsubfolder_path)
                        json_file_path = os.path.join(subsubfolder_path, 'your_posts_1.json')
                        data_to_append_df = parse_your_posts_1_to_dataframe(json_file_path, str(profile_id))
                        post_data_json = post_data_json.append(data_to_append_df, ignore_index=True)
                        
                        
                    if subfolder == 'posts' and os.path.exists(os.path.join(subsubfolder_path, 'your_posts_1.html')):
                        print("Found 'your_posts_1.html' in:", subsubfolder_path)
                        your_posts_html_file_path = os.path.join(subsubfolder_path, 'your_posts_1.html')
                        data_to_append_df = parse_your_posts_html(your_posts_html_file_path, str(profile_id))
                        post_data_html = post_data_html.append(data_to_append_df, ignore_index=True)

                    if subsubfolder == 'comments_and_reactions' and os.path.exists(subsubfolder_path):
                        if os.path.exists(os.path.join(subsubfolder_path, 'comments.json')):
                            print("Found 'comments.json' in:", subsubfolder_path)
                            comments_json_file_path = os.path.join(subsubfolder_path, 'comments.json')
                            data_to_append_df = parse_comments_json(comments_json_file_path, str(profile_id))
                            # Append data_to_append to json_data
                            comment_data_json = comment_data_json.append(data_to_append_df, ignore_index=True) 
                            
                        if os.path.exists(os.path.join(subsubfolder_path, 'comments.html')):
                            print("Found 'comments.html' in:", subsubfolder_path)
                            comments_hmtl_file_path = os.path.join(subsubfolder_path, 'comments.html')
                            data_to_append_df = parse_comments_html(comments_hmtl_file_path, str(profile_id))
                            # Append data_to_append to json_data
                            comment_data_html = comment_data_html.append(data_to_append_df, ignore_index=True)
                            
                        for filename in os.listdir(subsubfolder_path):
                            #print(filename)
                            if filename.endswith('.json'):
                                #anyfile name other than comments.json is likes and reaction
                                if os.path.exists(os.path.join(subsubfolder_path, filename)) and filename != 'comments.json':
                                    print(filename, "found in",subsubfolder_path )
                                    reaction_json_file_path = os.path.join(subsubfolder_path, filename)
                                    data_to_append_df = parse_likes_and_reactions_json(reaction_json_file_path, str(profile_id))
                                    # Append data_to_append to json_data
                                    reaction_data_json = reaction_data_json.append(data_to_append_df, ignore_index=True)       
                            if filename.endswith('.html'):
                                #anyfile name other than comments.html is likes and reaction
                                if os.path.exists(os.path.join(subsubfolder_path, filename)) and filename != 'comments.html':
                                    print(filename, "found in",subsubfolder_path )
                                    reaction_html_file_path = os.path.join(subsubfolder_path, filename)
                                    data_to_append_df = parse_likes_and_reactions_html(reaction_html_file_path, str(profile_id))
                                    # Append data_to_append to json_data
                                    reaction_data_html = reaction_data_html.append(data_to_append_df, ignore_index=True)

        else:
            continue  # Go to the next item in the outer loop


print("Profile IDs:", profile_ids)

post_data_json.to_csv('facebook_posts_json_data.csv',index=False)
comment_data_json.to_csv('facebook_comments_json_data.csv',index=False)
reaction_data_json.to_csv('facebook_likes_and_reactions_json_data.csv',index=False)


post_data_html.to_csv('facebook_posts_html_data.csv',index=False)
comment_data_html.to_csv('facebook_comments_html_data.csv',index=False) 
reaction_data_html.to_csv('facebook_likes_and_reactions_html_data.csv',index=False) 



