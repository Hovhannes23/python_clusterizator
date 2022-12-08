FROM ubuntu:18.04
RUN apt-get update -y && apt-get install -y python-pip python-dev
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 5000
CMD ["python", "./image_controller.py"]

#FROM python:3.8
#RUN apt-get update -y
#COPY . /app
#WORKDIR /app
#RUN pip install numpy flasgger Flask==0.12
#EXPOSE 5000
#ENTRYPOINT ["python"]
#CMD ["image_controller.py"]






#FROM python:3.8
#
#ADD image_controller.py .
#ADD response .
#RUN /usr/local/bin/python -m pip install --upgrade pip
#RUN #apt-get update && apt-get install -y python3-opencv
#RUN #pip install opencv-python
#RUN pip install numpy flasgger flask
#WORKDIR .
#CMD [ "python3", "./image_controller.py" ]
