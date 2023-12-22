FROM continuumio/miniconda3

WORKDIR /home/app

# Tu mets ce qui est dans le répertoire courant dans /home/app
COPY . /home/app

RUN pip install -r requirements.txt

CMD streamlit --server.port $PORT run app.py