from .dedupe import (
    compute_dedupe_hash,
    current_window_bucket,
    find_recent_duplicate,
    normalize_message,
)
from .dispatcher import (
    CONTACT_CATEGORY_TO_ENV,
    build_contact_email_body,
    get_destination_email,
    send_contact_notification,
)
from .scoring import compute_priority
from .spam_guard import (
    CONTACT_RATE_LIMITS,
    in_cooldown,
    is_silent_spam,
    minimal_spam_payload,
)

__all__ = [
    "CONTACT_CATEGORY_TO_ENV",
    "CONTACT_RATE_LIMITS",
    "build_contact_email_body",
    "compute_dedupe_hash",
    "compute_priority",
    "current_window_bucket",
    "find_recent_duplicate",
    "get_destination_email",
    "in_cooldown",
    "is_silent_spam",
    "minimal_spam_payload",
    "normalize_message",
    "send_contact_notification",
]
