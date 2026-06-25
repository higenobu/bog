from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import boto3
import os
import psycopg2

app = Flask(__name__)
textract = boto3.client("textract", region_name="us-east-2")
from janome.tokenizer import Tokenizer

def tokenize_japanese(text):
    tokenizer = Tokenizer()
    # 名詞、動詞、形容詞のみを抽出してリスト化
    tokens = tokenizer.tokenize(text)
    words = [token.surface for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
    return words


def make_jbow(text):
    vectorizer = CountVectorizer(analyzer=tokenize_japanese)
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    counts = X.toarray()[0]

    return sorted(zip(words, counts), key=lambda x: (-x[1], x[0]))


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


def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    return psycopg2.connect(database_url)


def save_bow_to_postgres(source, input_text, results):
    conn = get_db_connection()
    if conn is None:
        return

    normalized_results = [(word, int(count)) for word, count in results]
    total_unique = int(len(normalized_results))
    total_words = int(sum(count for _, count in normalized_results))

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
                    CREATE TABLE IF NOT EXISTS bow_items (
                        id SERIAL PRIMARY KEY,
                        run_id INTEGER NOT NULL REFERENCES bow_runs(id) ON DELETE CASCADE,
                        word TEXT NOT NULL,
                        count INTEGER NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO bow_runs (source, input_text, total_unique, total_words)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (source, input_text, total_unique, total_words),
                )
                run_id = cur.fetchone()[0]

                if normalized_results:
                    cur.executemany(
                        """
                        INSERT INTO bow_items (run_id, word, count)
                        VALUES (%s, %s, %s)
                        """,
                        [(run_id, word, count) for word, count in normalized_results],
                    )
    finally:
        conn.close()


def get_history(limit=50):
    conn = get_db_connection()
    if conn is None:
        return []

    runs = []
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, source, input_text, total_unique, total_words, created_at
                    FROM bow_runs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                run_rows = cur.fetchall()

                for run_id, source, input_text, total_unique, total_words, created_at in run_rows:
                    cur.execute(
                        """
                        SELECT word, count
                        FROM bow_items
                        WHERE run_id = %s
                        ORDER BY count DESC, word ASC
                        """,
                        (run_id,),
                    )
                    item_rows = cur.fetchall()
                    items = [(word, int(count)) for word, count in item_rows]

                    runs.append(
                        {
                            "id": run_id,
                            "source": source,
                            "input_text": input_text,
                            "total_unique": int(total_unique),
                            "total_words": int(total_words),
                            "created_at": created_at,
                            "items": items,
                        }
                    )
    finally:
        conn.close()

    return runs


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
            results = make_jbow(text)
            total_unique = len(results)
            total_words = sum(count for _, count in results)
            save_bow_to_postgres("bow", text, results)

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
                save_bow_to_postgres("ocr", extracted_text, results)

    return render_template(
        "ocr.html",
        extracted_text=extracted_text,
        results=results,
        total_unique=total_unique,
        total_words=total_words,
    )


@app.route("/history")
def history():
    runs = get_history(limit=50)
    return render_template("history.html", runs=runs)


if __name__ == "__main__":
    app.run()
