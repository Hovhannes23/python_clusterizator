FROM python:3.8

ADD engine.py .
ADD utils.py .

RUN pip install opencv-python numpy

CMD [ "python", "./image_controller.py" ]