import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from facebook_databuilder import extract_profile_id, parse_your_posts_1_to_dataframe, parse_comments_json, parse_likes_and_reactions_json



def main():
    root_folder_path = r'C:\Users\Ankit Chapagain\OneDrive - USU\CMIPS\Social Media Data\UNZIP'
    profile_ids = []
    post_data_json = pd.DataFrame(columns=['participant_id', 'timestamp',  'media_creation_timestamp', 'media_title', 'media_description', 'update_timestamp' 'post', 'title'])
    comment_data_json = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'comment_timestamp', 'comment_content', 'comment_author'])
    reaction_data_json = pd.DataFrame(columns=['participant_id', 'timestamp', 'title', 'reaction','actor']) #####

    for folder_name in os.listdir(root_folder_path):
        folder_path = os.path.join(root_folder_path, folder_name)
        if os.path.isdir(folder_path) and any(folder_name.endswith(suffix) for suffix in ['Facebook', 'Facebook1', 'Facebook2','Facebook3','Facebook-1','Facebook-2']):
            # Extract and store the profile ID
            profile_id = extract_profile_id(folder_name)
            if profile_id:
                profile_ids.append(profile_id)
            
            # Look for specific subfolders
            subfolders = ["this_profile's_activity_across_facebook"]
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                #print(subfolder_path)
                
                #there are some files inside this folder 0376 Profile/ Json                
                if subfolder == "this_profile's_activity_across_facebook" and os.path.exists(subfolder_path):
                    subsubfolders = ['comments_and_reactions', 'posts']
                    folder_path = os.path.join(folder_path, subfolder)
                    for subsubfolder in subsubfolders:   
                        subsubfolder_path = os.path.join(folder_path, subsubfolder)
                        if subsubfolder == 'posts' and os.path.exists(os.path.join(subsubfolder_path, 'profile_posts_1.json')):
                            print("Found 'profile_posts_1.json' in:", subsubfolder_path)
                            json_file_path = os.path.join(subsubfolder_path, 'profile_posts_1.json')
                            data_to_append_df = parse_your_posts_1_to_dataframe(json_file_path, str(profile_id))
                            post_data_json = post_data_json.append(data_to_append_df, ignore_index=True)
                            
                        if subsubfolder == 'comments_and_reactions' and os.path.exists(subsubfolder_path):
                            if os.path.exists(os.path.join(subsubfolder_path, 'comments.json')):
                                print("Found 'comments.json' in:", subsubfolder_path)
                                comments_json_file_path = os.path.join(subsubfolder_path, 'comments.json')
                                data_to_append_df = parse_comments_json(comments_json_file_path, str(profile_id))
                                # Append data_to_append to json_data
                                comment_data_json = comment_data_json.append(data_to_append_df, ignore_index=True)
                                
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

            else:
                continue  # Go to the next item in the outer loop


    print("Profile IDs:", profile_ids)

    post_data_json.to_csv('facebook_posts_json_data_036.csv',index=False)
    comment_data_json.to_csv('facebook_comments_json_data_036.csv',index=False)
    reaction_data_json.to_csv('facebook_likes_and_reactions_json_data_036.csv',index=False)

if __name__ == "__main__":
    main()



