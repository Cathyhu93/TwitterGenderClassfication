import pandas as pd
import urllib
import twitter
import os

df = pd.read_csv('gender-classifier-DFE-791531.csv', encoding = "ISO-8859-1")


api = twitter.Api(consumer_key='xKdXWK0l6Y6KRBfLn23AidPti',
                  consumer_secret='wXvW70Vz2dcWUeFwf2Sc3S7NFkNNPVu2Kl4o2aDjJdUYGrSnO4',
                  access_token_key='1868022308-h4bI5hlhAF3KF4H3serPBovoUtNFaWQUIKzBklx',
                  access_token_secret='5VJmo9LPjKhvht7yr9ZCNBIqOlZXyoagGAN1FXDuXge6C')

# data = api.GetUser(screen_name='GabrieleNeher').profile_image_url
# print data

img_dir = "./images_original/"
for idx, row in df.iterrows():
    target_file = img_dir + str(idx) + '.png'
    if os.path.isfile(target_file):
        continue

    try:
        user = api.GetUser(screen_name=row['name'])
    except:
        print row['name']
        continue

    if user.default_profile_image:
        continue  # skip default images
    url = user.profile_image_url
    idx_to_remove = url.rfind('_normal') # normal to original size
    if idx_to_remove == -1:
        print("Error URL: " + url)
        continue
    url = url[:idx_to_remove] + url[idx_to_remove + 7:]
    
    try:
        urllib.urlretrieve(url, target_file)
    except:
        continue