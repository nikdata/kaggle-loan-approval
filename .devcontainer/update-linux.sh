#!/usr/bin/env bash

# install base linux dependecies
 apt-get update && apt-get install -y --no-install-recommends \
    python3-launchpadlib \
    nano \
    curl \
    wget \
    gdebi \
    xdg-utils \
    git \
    gcc \
    gfortran \
    libglpk40 \
    htop \
    jq \
    libxtst6 \
    libxt6 \
    xdg-utils \
    tar \
    bzip2 \
    unzip \
    openssh-server \
    build-essential \
    libedit-dev \
    && rm -rf /var/lib/apt/lists/*