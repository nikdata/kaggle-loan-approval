ARG PYTHON_VER

# define base image
FROM python:${PYTHON_VER:-3.12}-slim-bookworm AS builder
# see why I did this here: https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/

# see devcontainer.json to see argument values
ARG PYTHON_ENV
ARG SETUP_FOLDER

ENV PYTHON_ENV=$PYTHON_ENV
ENV SETUP_FOLDER=$SETUP_FOLDER

# create directory
RUN mkdir $SETUP_FOLDER

# copy files
COPY update-linux.sh install-python-packages.sh packages.txt /$SETUP_FOLDER/

# setup environment
RUN bash ./${SETUP_FOLDER}/update-linux.sh
RUN bash ./${SETUP_FOLDER}/install-python-packages.sh $PYTHON_ENV $SETUP_FOLDER

# expose port
EXPOSE 8888

# final stage
FROM python:${PYTHON_VER:-3.12}-slim-bookworm

ARG PYTHON_ENV
ENV PYTHON_ENV=${PYTHON_ENV}

COPY --from=builder /opt/${PYTHON_ENV} /opt/${PYTHON_ENV}

# expose port
EXPOSE 8888
ENV PATH="/opt/${PYTHON_ENV}/bin:$PATH"
# ENV PATH="/opt/venv/bin:$PATH"