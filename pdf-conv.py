import fitz
import os

def convert_jpeg_to_pdf(jpeg_paths, output_pdf_path):
    """
    JPEG画像をPDFに変換する関数
    Parameters:
        jpeg_path(list): JPEGファイルのパス
        output_pdf_path (str): 出力するPDFファイルのパス
    """
    # 新しいPDFドキュメントを作成
    doc = fitz.open()

    for img_path in jpeg_paths:
        # ファイルを開く
        img = fitz.Pixmap(img_path)

        # RGBでない場合はRGBに変換
        if img.n > 4:
            img = fitz.Pixmap(fitz.csRGB, img)

        # PDFに変換
        rect = fitz.Rect(0, 0, img.width, img.height)
        page = doc.new_page(width=img.width, height=img.height)
        page.insert_image(rect, pixmap=img)

    # 保存して閉じる
    doc.save(output_pdf_path)
    doc.close()
    print(f"[I] PDFに変換完了: {output_pdf_path}")

# 使用例
if __name__ == "__main__":
    jpeg_paths = ["test.jpg"]
    output_pdf = "test.pdf"
    convert_jpeg_to_pdf(jpeg_paths, output_pdf)
