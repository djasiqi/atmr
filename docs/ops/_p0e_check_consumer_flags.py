import os

from services.tracking import ingest_consumer as m

print("env_OUTBOX", os.getenv("TRACKING_PERSIST_WITH_OUTBOX"))
print("env_MODE", os.getenv("TRACKING_INGEST_MODE"))
print("env_PERSIST", os.getenv("TRACKING_INGEST_PERSIST_ENABLED"))
print("env_PG_FIRST", os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED"))
print("eff_OUTBOX", m.TRACKING_PERSIST_WITH_OUTBOX)
print("eff_MODE", getattr(m, "TRACKING_INGEST_MODE", None))
