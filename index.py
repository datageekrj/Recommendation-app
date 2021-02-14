import os
from flask import Flask,request,render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
folder_for_data = os.path.join(app.instance_path, "static")
ds1 = pd.read_csv(folder_for_data + "/Course Description.csv")

ds = ds1.copy()
def main_function_to_recommend(name,num):
    global ds
    def item(id):
        return ds.loc[ds['id'] == id]['description'].to_list()[0].split('- ')[0]
    list_of_names = []
    for id in ds.id:
        list_of_names.append(item(id))
    dictionary_to_map = {j:i+1 for i,j in enumerate(list_of_names)}
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

        results[row['id']] = similar_items[1:]
    # Just reads the results out of the dictionary.
    def recommend(item_id, num):
        print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
        print("-------")
        recs = results[item_id][:num]
        res = {}
        for rec in recs:
            res[item(rec[1])] = str(rec[0])
            print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
        return res
    def recommend_from_name(name, num):
        id = dictionary_to_map[name]
        return recommend(id,num)
    return recommend_from_name(name, num)


@app.route("/", methods = ["GET", "POST"])
def home():
    return render_template("home.html")
@app.route("/course_list", methods = ["GET", "POST"])
def course_list():
    global ds
    if request.method == "POST":
        job_id = int(list(request.form.keys())[0])
        ds = ds[~ds["id"].isin([job_id])]
        ds["id"] = list(range(1, ds.shape[0]+1))
        ds.reset_index(inplace = True, drop=True)
        return render_template("course.html", data = ds,delete_id = job_id)
    return render_template("course.html", data = ds)

@app.route("/dropdown", methods = ["GET", "POST"])
def dropdown():
    global ds
    if request.method == "POST":
        name = request.form.get("course")
        num = request.form.get("number")
        try:
            num = int(num)
        except:
            num = 5
        results = main_function_to_recommend(name,int(num))
        return render_template("dropdown.html", results = results, name = name) 
    course_list = []
    for des in ds["description"]:
        course_list.append(des.split('- ')[0])
    return render_template("dropdown.html", data = course_list)

@app.route("/add_course", methods = ["GET", "POST"])
def add_course():
    global ds
    if request.method == "POST":
        title = request.form.get("title")
        desc = request.form.get("desc")
        s = pd.DataFrame({"id":[ds.shape[0]+1],"description":[title + " - " + desc]})
        ds = pd.concat((ds, s))
        ds.reset_index(inplace = True, drop=True)
        return render_template("add.html", title = title)    
    return render_template("add.html")

if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0")