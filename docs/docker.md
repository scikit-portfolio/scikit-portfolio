# Docker

You can use scikit-portfolio through docker, provided you have docker installed or available in your system.
Docker makes development efficient and predictable. Docker takes away repetitive, mundane configuration tasks and is used throughout the development lifecycle for fast, easy and portable application development – desktop and cloud. Docker’s comprehensive end to end platform includes UIs, CLIs, APIs and security that are engineered to work together across the entire application delivery lifecycle.

## Docker build
From root of repo issue the following command

```shell
docker build -f docker/Dockerfile . -t skportfolio
```


## Docker run an interpreter
You can start an IPython interpreter

```shell
docker run -it skportfolio poetry run ipython
```

or even a **Jupyter notebook** that you can remotely connect to!

```
docker run -it -p 8888:8888 skportfolio poetry run jupyter notebook --allow-root --no-browser --ip 0.0.0.0
```

Then open a browser pointing at `http://127.0.0.1:8888/?token=xxx` where `xxx` is the token indicated by the output of the above command.

### Docker run pytest

```shell
docker run -t skportfolio poetry run pytest
```

### Docker open a bash shell

```
docker run -it skportfolio bash
```


