from flask import Flask, render_template,request 
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin
import math
import re
from collections import Counter
import pickle

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
    title= request.get_json()
    # print(title["cast"])
    # title = title.lower()
    # df = pd.read_csv("../Dataset/reqAttr.csv")
    # print(title)
    # corr = []
    # vector1 = text_to_vector(title)
    # for ind, row in df.iterrows():
    #     vector2 = text_to_vector(row["combined"].lower())
    #     cosine = get_cosine(vector1, vector2)
    #     corr.append((row["movie_title"],cosine))
    # print(corr)
    # corr = sorted(corr,key=lambda x: x[1], reverse=True)
    # print(corr[0])
    res = {
        "data":"aditya",
        "status":200
    }
    return res

@app.route("/filterReviews", methods=["POST"])
def filterReviews():
    reviews_status=[]
    data= request.get_json()
    movie_review_list = np.array(data)
    movie_vector = vectorizer.transform(movie_review_list)
    pred = clf.predict(movie_vector)
    for i in range(pred):
        reviews_status.append((data[i], pred[i]))
    res = {
        "staus":200,
        "data":reviews_status
    }
    return res
if __name__ == '__main__':
    app.run(debug=True,port=5000)