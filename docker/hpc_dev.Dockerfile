FROM nvcr.io/nvidia/nvhpc:23.9-devel-cuda_multi-ubuntu22.04 as HPC

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel && \
    rm -rf /var/lib/apt/lists/*

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git-lfs

# Install Miniconda and make it the default Python
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
    # echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    # echo "conda activate base" >> ~/.bashrc

COPY .condarc /opt/conda/

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

WORKDIR /workdir
