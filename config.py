import os

current_file_path = os.path.abspath(__file__)
dir = os.path.dirname(current_file_path)

staging_area = os.path.join(dir, 'training_data/unprocessed')
liked_images = os.path.join(dir, 'training_data/unprocessed/liked')
disliked_images = os.path.join(dir, 'training_data/unprocessed/disliked')

mliked_images = os.path.join(dir, 'training_data/validation/liked')
mdisliked_images = os.path.join(dir, 'training_data/validation/disliked')

firefox_profile = r''

total_likes = 100
total_runs = 250
