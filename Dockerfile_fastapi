FROM python:3.10

RUN mkdir /fastapi_app

WORKDIR /fastapi_app

COPY new_req.txt .

RUN ls

RUN pip install --root-user-action=ignore -r new_req.txt

COPY . .

#RUN chmod a+x docker/*.sh

WORKDIR ./fastapi

CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000
