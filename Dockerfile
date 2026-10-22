# syntax=docker/dockerfile:1.7
# Dependencies only; the repository is mounted at /app at run time (see docker-compose.yml).
#
#   docker compose build [--build-arg PIP_INDEX_URL=<mirror>]   # downloads are checksum-verified
#   docker compose run --rm frds python -m frds all
#   docker compose up api dashboard

FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

# libgomp: OpenMP runtime for XGBoost and LightGBM.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-extra.txt docker/verify_wheels.py ./

# CPU-only torch from its own index, then every other wheel downloaded to /wheels,
# checksum-verified against the official index and installed offline.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip freeze | grep -i '^torch==' > /tmp/torch.txt \
 && pip download -d /wheels --only-binary=:all: -c /tmp/torch.txt \
      -r requirements.txt -r requirements-extra.txt \
      --index-url "$PIP_INDEX_URL" --extra-index-url https://download.pytorch.org/whl/cpu \
 && rm -f /wheels/torch-* \
 && python verify_wheels.py /wheels \
 && pip install --no-index --find-links /wheels -r requirements.txt -r requirements-extra.txt \
 && rm -rf /wheels \
 && ! pip list 2>/dev/null | grep -i '^nvidia'

ENV PYTHONPATH=/app/src:/app/serving \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

EXPOSE 8000 8501
CMD ["python", "-m", "frds", "--help"]
