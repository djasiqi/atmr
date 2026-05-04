from __future__ import annotations

import os

from flask_restx import Namespace, Resource, fields

from ext import redis_client

realtime_gateway_ns = Namespace(
    "realtime-gateway", description="Pilotage canary de la couche realtime dediee."
)

health_model = realtime_gateway_ns.model(
    "RealtimeGatewayHealth",
    {
        "ok": fields.Boolean(required=True),
        "canary_enabled": fields.Boolean(required=True),
        "presence_keys": fields.Integer(required=True),
    },
)


def _canary_enabled() -> bool:
    return os.getenv("REALTIME_GATEWAY_CANARY_ENABLED", "false").lower() == "true"


@realtime_gateway_ns.route("/health")
class RealtimeGatewayHealth(Resource):
    @realtime_gateway_ns.marshal_with(health_model)
    def get(self):
        presence_keys = 0
        if redis_client is not None:
            try:
                presence_keys = sum(
                    1 for _ in redis_client.scan_iter(match="presence:*", count=200)
                )
            except Exception:
                presence_keys = -1
        return {
            "ok": True,
            "canary_enabled": _canary_enabled(),
            "presence_keys": presence_keys,
        }, 200


@realtime_gateway_ns.route("/canary")
class RealtimeGatewayCanary(Resource):
    def get(self):
        return {
            "enabled": _canary_enabled(),
            "flag": "REALTIME_GATEWAY_CANARY_ENABLED",
        }, 200
