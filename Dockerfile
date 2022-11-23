FROM python:3.9
ADD requirements.txt /
ADD model_final.pth / 
ADD config.yaml /
RUN pip install -r /requirements.txt
ADD rooftop-classifier.py /
ENV PYTHONUNBUFFERED=1
EXPOSE $PORT
CMD [ "python", "./rooftop-classifier.py" ]
