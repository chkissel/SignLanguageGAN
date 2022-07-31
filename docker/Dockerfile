FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
RUN conda install --quiet --yes \ 
    numpy \
    pillow=6.1 \
    matplotlib \
    scipy \
    && conda clean -ya
RUN conda install --quiet --yes -c conda-forge \ 
    opencv \
    && conda clean -ya
CMD [ "sh", "-c","./scripts/$RUN_SCRIPT" ]