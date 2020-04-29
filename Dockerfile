FROM python:3.7

ENV PYTHONUNBUFFERED 1

RUN mkdir /DL_deploy_django

WORKDIR /DL_deploy_django

ADD . /DL_deploy_django/

RUN pip install -r requirements.txt
