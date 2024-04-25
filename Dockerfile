FROM python:3.111

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","prepare_dataset.py"]

CMD [ "python", "train.py" ]