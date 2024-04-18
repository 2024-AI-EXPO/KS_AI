FROM python:3.10

WORKDIR /ai

COPY ./requirements.txt /test/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /test/requirements.txt

COPY ./app /ai/app

WORKDIR /test/myapp

CMD ["uvicorn", "app.test:app", "--host", "0.0.0.0", "--port", "8000"]