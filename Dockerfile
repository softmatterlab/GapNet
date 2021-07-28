FROM tensorflow/tensorflow:nightly-jupyter

COPY requirements.txt .
RUN apt-get install -y python3-pydot
RUN apt-get install -y graphviz
RUN pip3 install -r requirements.txt

WORKDIR /tf

EXPOSE 8888
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
