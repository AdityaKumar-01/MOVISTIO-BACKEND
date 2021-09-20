from flask import Flask, render_template,request 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cors = CORS(app,resources = {
    r"/*":{
        "origins":"*"
    }
})

def getMovieNames(name):
    data = pd.read_csv("Dataset/reqAttr.csv")
    for i in range(len(data)):
        if name == data["movie_title"][i]:
            cntVec = CountVectorizer()
            cntMat = cntVec.fit_transform(data['combined'])
            similarity = cosine_similarity(cntMat)
            i = data.loc[data['movie_title']==name].index[0]
            lst = list(enumerate(similarity[i]))
            lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
            lst = lst[1:21] # excluding first item since it is the requested movie itself
            l = []
            for i in range(len(lst)):
                a = lst[i][0]
                l.append(data['movie_title'][a])
            l=list(set(l))
            if name in l:
                l.remove(name)
            l = l[0:10]
            s = "---".join(l)
            return s
    return "404"

@app.route('/',methods=["GET"])
def displayHome():
    x = {
        "name":"Aditya",
        "reg":"19BIT0139"
    }
    return x

@app.route('/rcmdByTitle', methods=["POST"])
def genre():
    title= request.form['title']
    listMovies = getMovieNames(title)
    return listMovies

if __name__ == '__main__':
    app.run(debug=True,port=5000)
