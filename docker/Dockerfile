FROM python:3.8-buster

WORKDIR skportfolio
COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip==22.1.1 && \
    pip install "poetry==1.1.13" && \
    pip install jupyterlab && \
    poetry install -E optionals --no-root

COPY . .


# Usage examples:
# Build
# from root of repo:
# docker build -f docker/Dockerfile . -t skportfolio
#
# Run
# iPython interpreter:
# docker run -it skportfolio poetry run ipython
# Jupyter notebook server:
# docker run -it -p 8888:8888 skportfolio poetry run jupyter notebook --allow-root --no-browser --ip 0.0.0.0
# click on http://127.0.0.1:8888/?token=xxx
# Pytest
# docker run -t skportfolio poetry run pytest
# Bash
# docker run -it skportfolio bash

