from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import boto3

app = Flask(__name__)
textract = boto3.client("textract", region_name="us-east-2")


def make_bow(text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])

    words = vectorizer.get_feature_names_out()
    counts = X.toarray()[0]

    return sorted(zip(words, counts), key=lambda x: (-x[1], x[0]))


def extract_text_from_image(file_bytes):
    response = textract.detect_document_text(Document={"Bytes": file_bytes})
    lines = []

    for block in response.get("Blocks", []):
        if block.get("BlockType") == "LINE":
            lines.append(block.get("Text", ""))

    return "\n".join(lines)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/bow", methods=["GET", "POST"])
def bow():
    results = []
    total_unique = 0
    total_words = 0
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            results = make_bow(text)
            total_unique = len(results)
            total_words = sum(count for _, count in results)

    return render_template(
        "bow.html",
        text=text,
        results=results,
        total_unique=total_unique,
        total_words=total_words,
    )


@app.route("/ocr", methods=["GET", "POST"])
def ocr():
    extracted_text = ""
    results = []
    total_unique = 0
    total_words = 0

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file and uploaded_file.filename:
            file_bytes = uploaded_file.read()
            extracted_text = extract_text_from_image(file_bytes)

            if extracted_text.strip():
                results = make_bow(extracted_text)
                total_unique = len(results)
                total_words = sum(count for _, count in results)

    return render_template(
        "ocr.html",
        extracted_text=extracted_text,
        results=results,
        total_unique=total_unique,
        total_words=total_words,
    )


if __name__ == "__main__":
    app.run()
