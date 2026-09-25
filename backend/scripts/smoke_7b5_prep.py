"""Smoke prep: publish synthetic channel caps + print flag status."""

from app import create_app
from ext import db
from services.legal.portal_channel_cancellation_caps import (
    get_current_channel_cancellation_policy,
    publish_channel_cancellation_policy,
    synthetic_test_channel_caps,
)
from services.legal.portal_double_validation import (
    is_portal_conditional_order_enabled,
    is_portal_double_validation_enabled,
)
from services.legal.portal_terms_catalog import effective_portal_terms_version

app = create_app()
with app.app_context():
    print("DV", is_portal_double_validation_enabled())
    print("CO", is_portal_conditional_order_enabled())
    print("TERMS", effective_portal_terms_version())
    cur = get_current_channel_cancellation_policy()
    if cur is None:
        r = publish_channel_cancellation_policy(body_json=synthetic_test_channel_caps())
        db.session.commit()
        print("CAPS_PUBLISH", r.ok, getattr(r.policy, "version", None), r.error)
    else:
        print("CAPS_EXISTING", cur.version, cur.content_hash[:16])
