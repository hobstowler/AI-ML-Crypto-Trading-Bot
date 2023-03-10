FROM python:3.9
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENV PORT=7890
ENV GOOGLE_APPLICATION_CREDENTIALS='./application_default_credentials.json'
ENV GCLOUD_PROJECT='ai-ml-bitcoin-bot'
EXPOSE ${PORT}
CMD [ "python", "main.py" ]
