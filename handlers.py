import asyncio
import cohere
from PIL import Image
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

from s3_handler import upload_images_to_s3
from parsers import parse_pdf, parse_image

from pinecone import Pinecone

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


co = cohere.AsyncClient(COHERE_API_KEY)


pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("flowllm-files")


def chunk_text(texts):
    docs = [Document(page_content=texts)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    chunks = splitter.split_documents(docs)
    chunks = [chunk.page_content for chunk in chunks]

    return chunks


async def handle_pdf(object_name):
    logger.info("Parsing PDF")
    document_name = object_name.split(".")[0]

    all_text, images, captions = await parse_pdf(object_name)

    chunks = chunk_text(all_text)

    logger.info("Uploading images")

    upload_task = upload_images_to_s3(document_name, images)

    logger.info("Embedding text")

    embed_task = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        texts=chunks + captions,
    )

    _, embeddings = await asyncio.gather(upload_task, embed_task)

    vectors = []

    for chunk in chunks:
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embeddings.embeddings.float_[chunks.index(chunk)],
                "metadata": {"text": chunk, "object_name": object_name},
            }
        )

    for i, caption in enumerate(captions):
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embeddings.embeddings.float_[len(chunks) + i],
                "metadata": {
                    "text": caption,
                    "image_path": f"{document_name}/image{i}.png",
                    "object_name": object_name,
                },
            }
        )

    return vectors


async def handle_image(object_name):
    image = Image.open(object_name)

    logger.info("Parsing image")

    image, text, needs_ocr = await parse_image(image)

    if needs_ocr:
        chunks = chunk_text(text)
    else:
        chunks = [text]

    logger.info("Embedding text")

    embeddings = await co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        texts=chunks,
    )

    vectors = []

    if needs_ocr:
        for chunk in chunks:
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embeddings.embeddings.float_[chunks.index(chunk)],
                    "metadata": {"text": chunk, "object_name": object_name},
                }
            )
    else:
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embeddings.embeddings.float_[0],
                "metadata": {
                    "text": text,
                    "image_path": object_name,
                    "object_name": object_name,
                },
            }
        )

    return vectors


async def handle_text(object_name):
    logger.info("Reading text")

    with open(object_name, "r") as f:
        text = f.read()

    chunks = chunk_text(text)

    logger.info("Embedding text")

    embeddings = await co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        texts=chunks,
    )

    vectors = []

    for chunk in chunks:
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embeddings.embeddings.float_[chunks.index(chunk)],
                "metadata": {"text": chunk, "object_name": object_name},
            }
        )

    return vectors


async def upload_vectors(user_id, vector_embeddings):
    await asyncio.to_thread(index.upsert, vectors=vector_embeddings, namespace=user_id)
