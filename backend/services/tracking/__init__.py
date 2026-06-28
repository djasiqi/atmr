"""Services tracking ingest async."""

from .ingest_producer import enqueue_tracking_event, enqueue_tracking_event_nowait

__all__ = ["enqueue_tracking_event", "enqueue_tracking_event_nowait"]
