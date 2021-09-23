from flask import Flask, render_template,request 
import pandas as pd
from flask_cors import CORS
import math
import re
from collections import Counter

app = Flask(__name__)
CORS(app)
cors = CORS(app,resources = {
    r"/*":{
        "origins":"*"
    }
})


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

@app.route('/getRecommendations', methods=["GET"])
def displayRecommendations():
    title= request.args.get('title')
    title = title.lower()
    df = pd.read_csv("../Dataset/reqAttr.csv")

    corr = []
    vector1 = text_to_vector(title)
    for ind, row in df.iterrows():
        vector2 = text_to_vector(row["combined"].lower())
        cosine = get_cosine(vector1, vector2)
        corr.append((row["movie_title"],cosine))
    print(len(corr))
    corr = sorted(corr,key=lambda x: x[1], reverse=True)
    res = {
        "data":corr[0:11],
        "status":200
    }
    return res


if __name__ == '__main__':
    app.run(debug=True,port=5000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def getMovieNames(name):
#     data = pd.read_csv("Dataset/reqAttr.csv")
#     for i in range(len(data)):
#         if name == data["movie_title"][i]:
#             cntVec = CountVectorizer()
#             cntMat = cntVec.fit_transform(data['combined'])
#             similarity = cosine_similarity(cntMat)
#             i = data.loc[data['movie_title']==name].index[0]
#             lst = list(enumerate(similarity[i]))
#             lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
#             lst = lst[1:21] # excluding first item since it is the requested movie itself
#             l = []
#             for i in range(len(lst)):
#                 a = lst[i][0]
#                 l.append(data['movie_title'][a])
#             l=list(set(l))
#             if name in l:
#                 l.remove(name)
#             l = l[0:10]
#             s = "---".join(l)
#             return s
#     return "404"

# @app.route('/',methods=["GET"])
# def displayHome():
#     x = {
#         "name":"Aditya",
#         "reg":"19BIT0139"
#     }
#     return x

# @app.route('/rcmdByTitle', methods=["POST"])
# def genre():
#     title= request.form['title']
#     listMovies = getMovieNames(title)
#     return listMovies


