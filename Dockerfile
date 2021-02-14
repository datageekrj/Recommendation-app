FROM python:3.6-slim
WORKDIR /recommendation-app
ADD . /recommendation-app 
RUN pip install --upgrade pip
RUN pip install flask Werkzeug==0.16.1 numpy sklearn pandas
CMD ["python", "index.py"]