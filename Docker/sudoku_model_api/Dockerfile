FROM python:3.6.9

RUN apt update
RUN apt install -y libgl1-mesa-glx

# copy over all dependency files
RUN mkdir /home/tensorflow_api
RUN mkdir /home/tensorflow_api/api
RUN mkdir /home/tensorflow_api/model

COPY ./Python/* /home/tensorflow_api/api/
COPY ./model/* home/tensorflow_api/model/
COPY ./Docker/sudoku_model_api/pip_requirements /home/tensorflow_api/

# install all python dependencies from requirements file without cacheing to prevent memory issues
RUN pip install -r /home/tensorflow_api/pip_requirements --no-cache-dir

# expose api port
EXPOSE 8000

# create mount point for shared drive and make it readable for python api root process
RUN mkdir /sudoku_volume
RUN chmod o+wxr /sudoku_volume

# set working directory
WORKDIR /home/tensorflow_api/api

# launch the sudoku solver model REST API
RUN chmod +x ./host_sudoku_solver_api.sh
#CMD  './host_sudoku_solver_api.sh'
