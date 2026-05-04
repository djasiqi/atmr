"""Services tracking ingest async."""

from .ingest_producer import enqueue_tracking_event

__all__ = ["enqueue_tracking_event"]
