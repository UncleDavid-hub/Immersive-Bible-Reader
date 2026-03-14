import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Ensure static directory exists to prevent startup crash
STATIC_DIR = "src/static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory="src/templates")

# Filename provided by Architect
PDF_PATH = "data/master/kjv-bible-1881.pdf"

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/page/{page_num}")
async def get_page_data(page_num: int):
    if not os.path.exists(PDF_PATH):
        raise HTTPException(status_code=404, detail=f"PDF file not found at {PDF_PATH}")

    try:
        doc = fitz.open(PDF_PATH)
        if page_num < 0 or page_num >= len(doc):
            raise HTTPException(status_code=400, detail="Invalid page number")

        page = doc.load_page(page_num)

        # Render page to image at high DPI for better OCR
        # 300 DPI is usually good for OCR (72 * 4.16... = 300)
        matrix = fitz.Matrix(4, 4)
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Perform OCR with pytesseract to get detailed data
        # Note: Tesseract's standard 'image_to_data' doesn't easily expose italics
        # without custom configuration/training. For Phase 1, we provide the text
        # and positioning. Italics can be refined in Phase 2 with HOCR or specialized models.
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        words = []
        n_boxes = len(ocr_data['text'])

        page_width = pix.width
        page_height = pix.height

        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text:
                # Heuristic: Tesseract sometimes includes font style info in 'conf' or other fields,
                # but 'italic' is not a default output in 'image_to_data' dict for all versions.
                # If we were using HOCR or Tesseract 5.0+ with specific params, we'd get more.
                is_italic = False

                # Check if 'italic' exists in the dictionary (some Tesseract versions/wrappers)
                if 'italic' in ocr_data and ocr_data['italic'][i] == 1:
                    is_italic = True

                word = {
                    "text": text,
                    "left": ocr_data['left'][i],
                    "top": ocr_data['top'][i],
                    "width": ocr_data['width'][i],
                    "height": ocr_data['height'][i],
                    "italic": is_italic
                }
                words.append(word)

        doc.close()

        return {
            "page": page_num,
            "width": page_width,
            "height": page_height,
            "words": words
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
