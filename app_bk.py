from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    total_unique = 0
    total_words = 0
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([text])

            words = vectorizer.get_feature_names_out()
            counts = X.toarray()[0]

            results = sorted(zip(words, counts), key=lambda x: (-x[1], x[0]))
            total_unique = len(words)
            total_words = int(sum(counts))

    return render_template(
        "index.html",
        results=results,
        total_unique=total_unique,
        total_words=total_words,
        text=text,
    )
