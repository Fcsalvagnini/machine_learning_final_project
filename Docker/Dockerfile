FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Install all dependencies to use PyTorch/TensorFlow GPU and to compile and use ift libraries
RUN apt update && apt upgrade -y \
    && apt install -y \
        build-essential git subversion sudo cmake zlib1g-dev libjpeg-dev \
        libpq-dev xvfb ffmpeg xorg-dev libboost-all-dev libsdl2-dev \
        swig unzip zip wget libopenblas-dev libatlas-base-dev \
        bison clang flex gperf libatspi2.0-dev libclang-dev \
        libdrm-dev libegl1-mesa-dev libfontconfig1-dev \
        libfreetype6-dev libgstreamer1.0-dev libhunspell-dev \
        libnss3-dev libopengl-dev libpulse-dev libssl-dev \
        libts-dev libx11-dev libx11-xcb-dev libxcb-glx0-dev \
        libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev \
        libxcb-randr0-dev libxcb-render-util0-dev libxcb-shape0-dev \
        libxcb-shm0-dev libxcb-sync-dev libxcb-util-dev itksnap \
        libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-xkb-dev \
        libxcb1-dev libxcomposite-dev libxcursor-dev libxdamage-dev \
        libxext-dev libxfixes-dev libxi-dev libxkbcommon-dev \
        libxkbcommon-x11-dev libxkbfile-dev libxrandr-dev libgl1-mesa-dev \
        libxrender-dev libxshmfence-dev libxshmfence1 llvm ninja-build \
        nodejs libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PATH /opt/conda/bin:$PATH
CMD [ "/bin/bash" ]
ARG CONDA_VERSION=py39_4.9.2
ARG CONDA_MD5=b4e46fcc8029e2cfa731b788f25b1d36

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_MD5}  miniconda.sh" > miniconda.md5 && \
    if ! md5sum --status -c miniconda.md5; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Copy the environment file at volume mounted
# Conda clean (t removes cached package tarballs and y doesn't ask for confirmation)
COPY environment.yml /tmp/
RUN conda update -y -n base conda \
    && conda env create -f /tmp/environment.yml \
    && conda clean -y -t \
    && rm /tmp/environment.yml

# Define args
ARG user_name
ARG user_id
ARG home=/home/${user_name}
ARG workdir=${home}

RUN adduser ${user_name} --uid ${user_id} --gecos '' --disabled-password \
    && echo "${user_name} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${user_name} \
    && chmod 0440 /etc/sudoers.d/${user_name}

# Change the group and user ownership of WORKDIR to created user
USER ${user_name}
WORKDIR ${workdir}
RUN chown -R ${user_name}:${user_name} ${home}

# Defines environment variables
ENV PATH=/opt/conda/envs/master/bin:$PATH

RUN conda init \
    && echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc