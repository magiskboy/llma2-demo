FROM python:3.11-slim-buster

WORKDIR /app

COPY . .

RUN pip install .

EXPOSE 8000

ENTRYPOINT "./scripts/start.sh"
