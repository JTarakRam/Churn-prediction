FROM python:3.9-slim-buster

WORKDIR /web_application

COPY requirements.txt .
COPY artifacts/ artifacts/
COPY src/ src/
COPY config/ config/
COPY web_application/main.py .
COPY web_application/pages/ pages/
COPY web_application/reports/ reports/

RUN pip install --upgrade pip && \
    apt-get update && \
    apt-get install -y gcc python3-dev git && \
    pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]
