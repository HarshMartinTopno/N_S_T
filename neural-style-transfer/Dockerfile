FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch

WORKDIR /app

# CPU-only torch keeps the image around 1 GB instead of 6 GB.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

COPY nst/ ./nst/
COPY static/ ./static/
COPY app.py cli.py ./

# Bake the VGG19 weights into the image so the first request is not a download.
RUN python -c "from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.IMAGENET1K_V1)"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
