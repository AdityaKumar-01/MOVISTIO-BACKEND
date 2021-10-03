from flask import Flask, render_template,request 
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
import math
import re
from collections import Counter
import pickle
import json
import bs4 as bs
import urllib.request

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
    
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


    
@app.route('/',methods=["GET"])
def displayHome():
    x = {
        "name":"Aditya",
        "reg":"19BIT0139"
    }
    return x

@app.route('/getRecommendations', methods=["POST"])
def displayRecommendations():
    data= request.get_json()
    inp = ""
    for i in data["movie"]:
        inp+= i+" "
        
    inp+=data["cast"][0]["original_name"]+ " "
    inp+=data["cast"][1]["original_name"]+ " "
    inp+=data["cast"][2]["original_name"]+ " "
    
    inp = inp.lower()
    df = pd.read_csv("../Dataset/reqAttr.csv")
   
    corr = []
    vector1 = text_to_vector(inp)
    for ind, row in df.iterrows():
        vector2 = text_to_vector(row["combined"].lower())
        cosine = get_cosine(vector1, vector2)
        corr.append((row["movie_title"],cosine))
    # print(corr)
    corr = sorted(corr,key=lambda x: x[1], reverse=True)

    res = {
        "data":corr[1:7],
        "status":200
    }
    return res

@app.route("/filterReviews", methods=["POST"])
def filterReviews():
    data= request.get_json()
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(data["id"])).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})
    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = []
    for i in range(len(reviews_status)):
        movie_reviews.append((reviews_list[i], reviews_status[i]))     
   
    res = {
        "staus":200,
        "data":movie_reviews
    }
   
    return res
if __name__ == '__main__':
    app.run(debug=True,port=5000)