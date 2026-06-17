from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import boto3
import os
import psycopg2

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


def save_bow_summary_to_postgres(source, input_text, total_unique, total_words):
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        return

    conn = psycopg2.connect(database_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bow_runs (
                        id SERIAL PRIMARY KEY,
                        source TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        total_unique INTEGER NOT NULL,
                        total_words INTEGER NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO bow_runs (source, input_text, total_unique, total_words)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (source, input_text, total_unique, total_words),
                )
    finally:
        conn.close()


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
            save_bow_summary_to_postgres("bow", text, total_unique, total_words)

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
                save_bow_summary_to_postgres("ocr", extracted_text, total_unique, total_words)

    return render_template(
        "ocr.html",
        extracted_text=extracted_text,
        results=results,
        total_unique=total_unique,
        total_words=total_words,
    )


if __name__ == "__main__":
    app.run()
