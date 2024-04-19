FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /test/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /test/requirements.txt
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx


COPY ./test.py /code/

CMD ["uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8000"]