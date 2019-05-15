FROM python

RUN python3 -m venv fxcm
CMD ["source", "fxcm/bin/activate"] 

WORKDIR /fxcm
COPY . .

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

RUN \
  pip install wheel && \
  pip install pandas && \
  pip install scikit-learn && \
  pip install tensorflow && \
  tar -zxvf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib && \
  ./configure --prefix=/usr && \
  make && \
  make install && \
  pip install Ta-Lib && \
  pip install fxcmpy

CMD ["python", "ml.py"]
