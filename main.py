import asyncio
import json
import os
from fastapi import FastAPI
from dotenv import load_dotenv
from handlers import process_item
from pydantic import BaseModel
from upstash_redis import Redis
import logging

load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParseRequest(BaseModel):
    object_name: str
    user_id: str


redis_url = os.getenv("UPSTASH_REDIS_URL")
redis_token = os.getenv("UPSTASH_REDIS_TOKEN")

redis = Redis(url=redis_url, token=redis_token)


QUEUE_NAME = "files-to-process"


@app.post("/api/parse-and-embed")
async def parse_and_embed() -> None:
    while True:
        item = redis.rpop(QUEUE_NAME)

        if not item:
            break

        data = json.loads(item)

        file_url = data["fileUrl"]
        user_id = data["userId"]

        if not file_url or not user_id:
            logger.error(f"Invalid item: {item}")
            continue

        logger.info(f"Processing item: {file_url} for user: {user_id}")

        await process_item(file_url, user_id)

    return {"message": "All items processed"}
