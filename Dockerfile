FROM python:3.9-slim-buster

WORKDIR /web_application/

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y gcc python3-dev git

COPY requirements.txt /web_application/
COPY reports /web_application/reports
COPY pages /web_application/pages
COPY artifacts /web_application/artifacts
COPY src /web_application/src
COPY config /web_application/config
COPY main.py /web_application/

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "--server.port", "8501", "main.py"]
