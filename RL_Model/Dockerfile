FROM gcr.io/deeplearning-platform-release/pytorch-gpu
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENV PORT=7890
EXPOSE ${PORT}
CMD [ "python", "main.py" ]
