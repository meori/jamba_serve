FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# install git-lfs for model cloning
RUN apt update && apt install -y git-lfs
RUN git lfs install

# cd to workspace
RUN mkdir /ml_workspace && cd /ml_workspace
WORKDIR /ml_workspace

# install dependencies
RUN pip install transformers
RUN pip install git+https://github.com/Dao-AILab/causal-conv1d
RUN pip install git+https://github.com/state-spaces/mamba

RUN pip install flash-attn --no-build-isolation
RUN pip install accelerate

RUN cd /ml_workspace

# install Jamba with model and tokenizer
RUN git clone https://huggingface.co/ai21labs/Jamba-v0.1

# install API server
RUN pip install fastapi "uvicorn[standard]"

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./main.py /ml_workspace/main.py

CMD ["/start.sh"]
