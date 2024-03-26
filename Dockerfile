FROM python:3.10

RUN mkdir /mm_coreset

WORKDIR /mm_coreset

COPY requirements.txt .

RUN ls

RUN pip install --root-user-action=ignore -r requirements.txt

COPY . .

#RUN chmod a+x docker/*.sh

WORKDIR .

