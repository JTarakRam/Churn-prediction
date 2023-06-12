
FROM python:3.10.10

WORKDIR /web_application/


RUN pip install --upgrade pip


RUN apt-get update && apt-get install -y gcc python3-dev git


COPY requirements.txt /web_application/


RUN pip install -r requirements.txt


COPY . /web_application/


EXPOSE 8501


CMD ["streamlit", "run", "main.py"]
