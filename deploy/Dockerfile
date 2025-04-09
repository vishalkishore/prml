FROM python:3.10-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ./app /app/app
COPY ./pyproject.toml /app/

COPY ./pyproject.toml /app/project_pyproject.toml
COPY ./uv.lock /app/uv.lock

RUN uv sync --frozen --no-cache && rm -rf /root/.cache

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
