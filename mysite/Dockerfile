FROM python:3.7

ENV PYTHONUNBUFFERED 1

RUN mkdir /DL_deploy_django

WORKDIR /DL_deploy_django/

ADD ./requirements.txt .

RUN pip install -r requirements.txt

ADD . /DL_deploy_django



