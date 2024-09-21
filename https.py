from fastapi import FastAPI
import os
import uvicorn
from dotenv import load_dotenv

app = FastAPI(title="HTTPS 적용 테스트")

load_dotenv()


@app.get("/")
async def https():
  return {"message" : "yabbi"}

ssl_keyfile = os.getenv("SSL_KEYFILE")
ssl_certfile = os.getenv("SSL_CERTFILE")

if __name__ == "__main__":
    uvicorn.run(
        "https:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )