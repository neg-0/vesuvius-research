# Ink Prospector. CPU image by default; for a GPU image build with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124   and run with --gpus all ... scan --device cuda
FROM python:3.13-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY . /opt/prospector-src
# setup.sh builds the work folder: venv, pinned villa reader code, Reader V3 weights (sha256-checked), smoke test.
RUN TORCH_INDEX=$TORCH_INDEX bash /opt/prospector-src/setup.sh /opt/prospector && rm -rf /root/.cache
ENV PROSPECTOR_HOME=/opt/prospector PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["/opt/prospector/venv/bin/python", "/opt/prospector-src/prospect.py"]
CMD ["--help"]
