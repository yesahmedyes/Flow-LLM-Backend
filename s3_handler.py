from io import BytesIO
import os
import boto3
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)


def download_file_from_s3(bucket, object_name, file_name=None):
    if file_name is None:
        file_name = object_name
    try:
        s3.download_file(bucket, object_name, file_name)
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise e


async def upload_single_image(document_name, image, i):
    def upload():
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        s3.upload_fileobj(buffer, "flowllm-bucket", f"{document_name}/image{i}.png")

    await asyncio.to_thread(upload)


async def upload_images_to_s3(document_name, images):
    tasks = [
        upload_single_image(document_name, image, i) for i, image in enumerate(images)
    ]

    await asyncio.gather(*tasks)
