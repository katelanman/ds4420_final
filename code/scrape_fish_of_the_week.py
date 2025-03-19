import requests
import os
import pandas as pd

from bs4 import BeautifulSoup

SAVEDIR = "data/original/fish_of_the_week/"
URL = "https://visdeurbel.nl/en/fish-updates/"

page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")

data = []
pictures = soup.find_all(class_="fish-of-the-week__media")
for picture in pictures:
    # extract images
    img = picture.find("img")['src']
    img_filename = img.split('/')[-1].strip('.jpeg') + '.jpg' # extract unique filename
    
    with open(os.path.join(SAVEDIR, 'imgs/',img_filename), 'wb') as infile:
        infile.write(requests.get(img).content)

    # grab picture descriptions if any
    picture_idx = picture["data-slide-index"]
    content = soup.find(class_="fish-of-the-week__slide-content", attrs={"data-slide-index" : picture_idx})
    descs = content.find_all(class_="fish-of-the-week__description")
    if descs is not None:
        descs = ';'.join([desc.string for desc in descs])
    
    data.append([img_filename.strip('.jpg'), img, descs])

data = pd.DataFrame(data, columns=["id", "source", "description"])
data.to_csv(os.path.join(SAVEDIR, "img_data.csv"))
