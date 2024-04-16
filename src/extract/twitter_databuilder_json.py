import json
import os
import pandas as pd

def extract_tweet_data(js_file, profile_id):
    with open(js_file, 'r', encoding='utf-8') as file:
        data = file.read()
        # Find the start and end indices of the tweet data within the JavaScript file
        start_index = data.find('[')
        end_index = data.rfind(']') + 1
        # Extract the tweet data substring
        tweets_data_str = data[start_index:end_index]

    # Load the tweet data as JSON
    tweets_data = json.loads(tweets_data_str)

    tweet_list = []

    for tweet_info in tweets_data:
        tweet = tweet_info.get('tweet', {})
        tweet_id = tweet.get('id')
        created_at = tweet.get('created_at')
        full_text = tweet.get('full_text')
        source = tweet.get('source')
        retweeted = tweet.get('retweeted', False)

        entities = tweet.get('entities', {})
        hashtags = [tag['text'] for tag in entities.get('hashtags', [])]

        tweet_data = {
            'participant_id': profile_id,
            'tweet_id': tweet_id,
            'created_at': created_at,
            'full_text': full_text,
            'source': source,
            'retweeted': retweeted,
            'hashtags': hashtags
        }

        tweet_list.append(tweet_data)

    return pd.DataFrame(tweet_list)


def extract_like_data(js_file, profile_id):
    with open(js_file, 'r', encoding='utf-8') as file:
        data = file.read()
        start_index = data.find('[')
        end_index = data.rfind(']') + 1
        like_data_str = data[start_index:end_index]

    like_data = json.loads(like_data_str)

    like_list = []

    for like_info in like_data:
        like = like_info.get('like', {})
        tweet_id = like.get('tweetId')
        full_text = like.get('fullText')
        expanded_url = like.get('expandedUrl')

        like_data = {
            'participant_id': profile_id,
            'tweet_id': tweet_id,
            'full_text': full_text,
            'expanded_url': expanded_url
        }

        like_list.append(like_data)

    return pd.DataFrame(like_list)

# Define the root folder path
root_folder_path = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'
# Initialize an empty list to store profile IDs
profile_ids = []
tweet_data = pd.DataFrame(columns=['participant_id', 'tweet_id', 'created_at', 'full_text', 'source', 'retweeted','hashtags'])
like_data = pd.DataFrame(columns=['participant_id', 'tweet_id',  'full_text', 'expanded_url'])

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
    if os.path.isdir(folder_path) and folder_name.endswith('Twitter'):
        # Extract and store the profile ID
        profile_id = extract_profile_id(folder_name)
        if profile_id:
            profile_ids.append(profile_id)
        
        # Look for specific subfolders
        subfolders = ['data']
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if subfolder == 'data' and os.path.exists(os.path.join(subfolder_path, 'tweets.js')):
                print("Found 'tweets.js' in:", subfolder_path)
                tweet_js_file_path = os.path.join(subfolder_path, 'tweets.js')
                data_to_append_df = extract_tweet_data(tweet_js_file_path, str(profile_id))
                 # Append data_to_append to json_data
                tweet_data = tweet_data.append(data_to_append_df, ignore_index=True)   
                
            if subfolder == 'data' and os.path.exists(os.path.join(subfolder_path, 'like.js')):
                print("Found 'like.js' in:", subfolder_path)
                like_js_file_path = os.path.join(subfolder_path, 'like.js')
                data_to_append_df = extract_like_data(like_js_file_path, str(profile_id))
                 # Append data_to_append to json_data
                like_data = like_data.append(data_to_append_df, ignore_index=True)  
        else:
            continue  # Go to the next item in the outer loop

# Print profile IDs
print("Profile IDs:", profile_ids)

tweet_data.to_csv('twitter_tweet_js_data.csv',index=False)
like_data.to_csv('twitter_like_js_data.csv',index=False)


