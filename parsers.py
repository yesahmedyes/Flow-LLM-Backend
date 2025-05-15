import asyncio
from pdfplumber import open as open_pdf
from PIL import Image
from image_handler import image_ocr, encode_image_to_base64, decode_image_from_base64
from model_calls import parse_page_gpt, create_caption


async def parse_image(image):
    needs_ocr, text = await asyncio.to_thread(image_ocr, image)

    base64_image = await asyncio.to_thread(encode_image_to_base64, image)
    data_url = f"data:image/{image.format};base64,{base64_image}"

    if needs_ocr:
        result = await asyncio.to_thread(parse_page_gpt, data_url, text)

        cleaned_text, caption = result.cleaned_text, result.caption
        text = f"\nIMAGE CAPTION: {caption}\nIMAGE TEXT: {cleaned_text}\n"

        return image, text, needs_ocr
    else:
        caption = await asyncio.to_thread(create_caption, data_url)

        return image, caption, needs_ocr


async def parse_pdf_page(page):
    all_text = ""
    images = []
    captions = []

    text = page.extract_text()

    if text is not None:
        all_text += text

    tasks = []

    for img in page.images:
        dim = (img["x0"], img["top"], img["x1"], img["bottom"])

        image = page.crop(dim, strict=False).to_image(resolution=300).original

        tasks.append(parse_image(image))

    results = await asyncio.gather(*tasks)

    for image, caption, needs_ocr in results:
        if needs_ocr:
            all_text += caption
        else:
            images.append(image)
            captions.append(caption)

    return all_text, images, captions


async def parse_pdf(pdf_path):
    with open_pdf(pdf_path) as pdf:
        all_text = ""
        all_images, all_captions = [], []

        tasks = [parse_pdf_page(page) for page in pdf.pages]

        results = await asyncio.gather(*tasks)

        for text, images, captions in results:
            all_text += text
            all_images.extend(images)
            all_captions.extend(captions)

    return all_text, all_images, all_captions
