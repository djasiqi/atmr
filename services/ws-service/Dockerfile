FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py
COPY auth_claims.py /app/auth_claims.py
COPY dedup.py /app/dedup.py
COPY delivery_metrics.py /app/delivery_metrics.py
COPY kafka_lag_metrics.py /app/kafka_lag_metrics.py
COPY event_contract.py /app/event_contract.py
COPY gps_ingest.py /app/gps_ingest.py
COPY kill_switch.py /app/kill_switch.py
COPY rooms.py /app/rooms.py

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
