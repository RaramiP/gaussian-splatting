FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"


RUN apt-get update && apt-get install -y \
    curl git cmake wget build-essential ninja-build neovim nano ffmpeg\
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
    libboost-iostreams-dev libsuitesparse-dev libfreeimage-dev libgoogle-glog-dev \
    libgflags-dev libglew-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    libflann-dev libsqlite3-dev liblz4-dev libmetis-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout 3.8 && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89" && \
    ninja install


RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:/usr/local/bin:$PATH

WORKDIR /app
COPY . .

RUN conda env create -f environment.yml
RUN conda env create -f sam_environment.yaml


RUN conda init bash
ENTRYPOINT ["/bin/bash"]

