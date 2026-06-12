import boto3

textract = boto3.client("textract", region_name="us-east-2")

with open("test.jpg", "rb") as f:
    file_bytes = f.read()

response = textract.detect_document_text(Document={"Bytes": file_bytes})
print(response)
