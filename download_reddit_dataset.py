from minedojo.data import RedditDataset
import os

# TODO: Replace these with your actual credentials
client_id = "{your_client_id}"
client_secret = "{your_client_secret}"
user_agent = "{your_user_agent}"
download_dir = "./minedojo_data" 

print(f"Downloading to {download_dir}...")

if client_id == "{your_client_id}":
    print("Please update the credentials in the script or provide them to the agent.")
    exit(1)

reddit_dataset = RedditDataset(
  client_id=client_id, 
  client_secret=client_secret, 
  user_agent=user_agent,
  download=True,
  download_dir=download_dir
) 
print(f"Dataset length: {len(reddit_dataset)}")
