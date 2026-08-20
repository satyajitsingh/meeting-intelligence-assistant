# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime

# Slim rather than alpine: fastembed pulls onnxruntime, which ships manylinux
# wheels but not musl ones, so alpine would force a source build.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # fastembed caches the ONNX model here; mounted as a volume in compose so a
    # rebuild does not re-download it.
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed

WORKDIR /app

# Installed from pyproject.toml rather than a hand-copied requirement list, so
# the image can never drift from the project's declared dependencies.
COPY pyproject.toml ./
COPY app ./app
RUN python -m pip install --upgrade pip && python -m pip install .

# Run unprivileged.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p "$FASTEMBED_CACHE_PATH" \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8000

# No curl in the slim image, so probe with the interpreter that is already here.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
