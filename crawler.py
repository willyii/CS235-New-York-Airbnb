# #This file used for initialize a crawler to get data from Airbnb.com to get data.


import pandas as pd 

raw = pd.read_csv("AB_NYC_2019.csv")
id_list = raw["id"].tolist()



## selenium capture 
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import requests
from tqdm import tqdm


# browser = webdriver.Chrome() # Get local session of firefox

count = 0
rates = []
reviews = []

for id in tqdm(id_list):

    url = "https://www.airbnb.com/rooms/" + str(id) + "?source_impression_id=p3_1572300638_xHmYO4dmxnnrWjDt"
    # print(url)
    
    # browser.get(url) # Load page
    rep = requests.get(url)
    # time.sleep(2)
    soup = BeautifulSoup(rep.text, "lxml")

    try:
        overall_rate = soup.find("div", id="reviews").find("div", class_="_10za72m2").get_text()
    except AttributeError:
        print("this is the "+ str(count+1)+ " error")
        print(url)
        count += 1
        overall_rate = 0
    try:
        review = soup.find("div", id="reviews").find_all("div", class_="_czm8crp")[0].get_text()
    except :
        review = ""
        
    rates.append(overall_rate)
    reviews.append(review)
    
rate_df = pd.DataFrame(rates,columns = ["rate"])
rate_df.to_csv("rate.csv", index = False)

review_df = pd.DataFrame(reviews,columns = ["review"])
review_df.to_csv("review.csv", index = False)
