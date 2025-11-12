#FROM python:3.8-slim-buster

#WORKDIR /app
#COPY . /app

#RUN apt update -y && apt install awscli -y

#RUN pip install -r requirements.txt
#CMD ["python3" , "app.py"]

FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

# Install AWS CLI via pip (simpler approach)
RUN pip install awscli

RUN pip install -r requirements.txt

# Use JSON array format for CMD
CMD ["python3", "app.py"]