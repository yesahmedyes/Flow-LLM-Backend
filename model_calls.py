from pydantic import BaseModel, Field

from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


class GPTResponse(BaseModel):
    cleaned_text: str = Field(
        ..., description="The cleaned text extracted from the document."
    )
    caption: str = Field(..., description="A descriptive caption for the image.")


def parse_page_gpt(data_url, parsed_text):
    system_prompt = """You are a document text cleaner and formatter. Your goal is to process raw OCR text from a document, using the original document image to correct errors and restore the correct text flow and structure.

You will be provided with:
1.  The original document image.
2.  The raw, uncorrected text from OCR.

Compare the OCR text to the document image. Your task is to:
- Identify and correct recognition errors (typos, incorrect characters).
- Correctly handle line breaks and paragraph breaks based on the document's layout.
- Merge fragmented words or lines if they appear as single entities in the image.
- Present the final text in a clean, readable format that accurately reflects the original document's content and basic structure (e.g., paragraphs, lists).

Also generate a short caption for the image provided.

Your caption should include:
- The primary subject(s) or objects in the image.
- Any visible document type (e.g., invoice, form, letter, screenshot, handwritten note).
- A description of the layout or prominent visual features of the document.
- Mention of significant elements within the document (e.g., signatures, logos, tables, charts).
- If clearly visible and concise to summarize, a brief note about the topic or key information present in the text.
"""

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": parsed_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        response_format=GPTResponse,
    )

    return completion.choices[0].message.parsed


def create_caption(data_url):
    system_prompt = """You are a detailed image caption generator. Your goal is to generate a DETAILED AND DESCRIPTIVE CAPTION for an image that accurately portrays its key visual elements.
Focus on identifying and describing:
- The main subjects (people, objects, animals).
- Any actions or activities taking place.
- The setting or environment (location, background, context).
- Notable details like colors, lighting, expressions, or atmosphere.
Generate a clear and informative caption based on these elements.
"""

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the image you need to caption: ",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
    )

    return completion.choices[0].message.content
