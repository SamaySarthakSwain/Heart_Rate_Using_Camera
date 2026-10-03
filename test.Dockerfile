FROM python:3.10-slim AS deps
WORKDIR /install
COPY reqs.txt .
RUN pip install --no-cache-dir --prefix=/install/pkg -r reqs.txt

FROM python:3.10-slim
COPY --from=deps /install/pkg /usr/local
RUN uvicorn --version
