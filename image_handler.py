import base64
import asyncio
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract


def encode_image_to_base64(image):
    from io import BytesIO

    buffer = BytesIO()

    image.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_from_base64(base64_image):
    from io import BytesIO

    img_bytes = base64.b64decode(base64_image)
    img = Image.open(BytesIO(img_bytes))

    plt.imshow(img)
    plt.axis("off")
    plt.show()


def image_ocr(pil_image, min_text_length=200):
    text = pytesseract.image_to_string(pil_image)

    text_length = len(text.strip().replace(" ", "").replace("\n", "").replace("\t", ""))

    return text_length >= min_text_length, text
