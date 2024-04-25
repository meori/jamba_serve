FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
ARG jamba_repo=https://huggingface.co/ai21labs/Jamba-v0.1
ARG jamba_dir=Jamba-v0.1

# install git-lfs for model cloning
RUN apt update

# cd to workspace
RUN mkdir /ml_workspace && cd /ml_workspace
WORKDIR /ml_workspace

# install dependencies
RUN pip install torch packaging
RUN pip install einops transformers
RUN pip install git+https://github.com/Dao-AILab/causal-conv1d
RUN pip install git+https://github.com/state-spaces/mamba

RUN pip install flash-attn --no-build-isolation
RUN pip install accelerate

RUN cd /ml_workspace

# install Jamba with model and tokenizer
RUN git clone ${jamba_repo}

# manually install each safetensors file - because putting them in one step created a layer too large
RUN wget ${jamba_repo}/resolve/main/model-00001-of-00021.safetensors?download=true -O ${jamba_dir}/model-00001-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00002-of-00021.safetensors?download=true -O ${jamba_dir}/model-00002-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00003-of-00021.safetensors?download=true -O ${jamba_dir}/model-00003-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00004-of-00021.safetensors?download=true -O ${jamba_dir}/model-00004-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00005-of-00021.safetensors?download=true -O ${jamba_dir}/model-00005-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00006-of-00021.safetensors?download=true -O ${jamba_dir}/model-00006-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00007-of-00021.safetensors?download=true -O ${jamba_dir}/model-00007-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00008-of-00021.safetensors?download=true -O ${jamba_dir}/model-00008-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00009-of-00021.safetensors?download=true -O ${jamba_dir}/model-00009-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00010-of-00021.safetensors?download=true -O ${jamba_dir}/model-00010-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00011-of-00021.safetensors?download=true -O ${jamba_dir}/model-00011-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00012-of-00021.safetensors?download=true -O ${jamba_dir}/model-00012-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00013-of-00021.safetensors?download=true -O ${jamba_dir}/model-00013-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00014-of-00021.safetensors?download=true -O ${jamba_dir}/model-00014-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00015-of-00021.safetensors?download=true -O ${jamba_dir}/model-00015-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00016-of-00021.safetensors?download=true -O ${jamba_dir}/model-00016-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00017-of-00021.safetensors?download=true -O ${jamba_dir}/model-00017-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00018-of-00021.safetensors?download=true -O ${jamba_dir}/model-00018-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00019-of-00021.safetensors?download=true -O ${jamba_dir}/model-00019-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00020-of-00021.safetensors?download=true -O ${jamba_dir}/model-00020-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/model-00021-of-00021.safetensors?download=true -O ${jamba_dir}/model-00021-of-00021.safetensors
RUN wget ${jamba_repo}/resolve/main/tokenizer.model?download=true -O ${jamba_dir}/tokenizer.model


# install API server
RUN pip install fastapi "uvicorn[standard]"

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./main.py /ml_workspace/main.py

CMD ["/start.sh"]
