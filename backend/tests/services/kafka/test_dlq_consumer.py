"""Couverture de ``services.kafka.dlq_consumer``."""

from __future__ import annotations

import json
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.kafka import dlq_consumer as dlq


@pytest.fixture
def consumer(monkeypatch, tmp_path):
    monkeypatch.setattr(dlq, "KAFKA_ENABLED", False)
    monkeypatch.setattr(dlq, "KAFKA_DLQ_STORAGE_PATH", str(tmp_path / "dlq.jsonl"))
    monkeypatch.setattr(dlq.signal, "signal", lambda *_a, **_k: None)
    inst = dlq.KafkaDlqConsumer()
    inst._persist_path = tmp_path / "dlq.jsonl"
    return inst


def test_security_config_complet(monkeypatch):
    monkeypatch.setattr(dlq, "KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    monkeypatch.setattr(dlq, "KAFKA_SASL_MECHANISM", "PLAIN")
    monkeypatch.setattr(dlq, "KAFKA_SASL_USERNAME", "user")
    monkeypatch.setattr(dlq, "KAFKA_SASL_PASSWORD", "pass")
    monkeypatch.setattr(dlq, "KAFKA_SSL_CAFILE", "/ca.pem")
    monkeypatch.setattr(dlq, "KAFKA_SSL_CERTFILE", "/cert.pem")
    monkeypatch.setattr(dlq, "KAFKA_SSL_KEYFILE", "/key.pem")
    cfg = dlq._kafka_security_config()
    assert cfg["security_protocol"] == "SASL_SSL"
    assert cfg["sasl_mechanism"] == "PLAIN"
    assert cfg["ssl_cafile"] == "/ca.pem"


def test_security_config_minimal():
    cfg = dlq._kafka_security_config()
    assert cfg == {"security_protocol": dlq.KAFKA_SECURITY_PROTOCOL}


def test_init_disabled_et_property(consumer):
    assert consumer.initialized is False
    assert consumer._consumer is None


def test_init_appelle_init_si_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(dlq, "KAFKA_ENABLED", True)
    monkeypatch.setattr(dlq, "KAFKA_DLQ_STORAGE_PATH", str(tmp_path / "dlq.jsonl"))
    monkeypatch.setattr(dlq.signal, "signal", lambda *_a, **_k: None)
    called = {"n": 0}

    def _init(self):
        called["n"] += 1
        self._initialized = True

    monkeypatch.setattr(dlq.KafkaDlqConsumer, "_init_consumer", _init)
    inst = dlq.KafkaDlqConsumer()
    assert called["n"] == 1
    assert inst.initialized is True


def test_init_consumer_ok_et_deserializers(consumer, monkeypatch):
    captured: dict = {}

    class FakeKC:
        def __init__(self, *topics, **kwargs):
            captured["topics"] = topics
            captured["kwargs"] = kwargs

    kafka_mod = types.ModuleType("kafka")
    kafka_mod.KafkaConsumer = FakeKC  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "kafka", kafka_mod)
    monkeypatch.setattr(
        "services.kafka.bootstrap_retry.run_with_kafka_bootstrap_retry",
        lambda **kwargs: kwargs["fn"](),
    )
    consumer._init_consumer()
    assert consumer.initialized is True
    assert consumer._persist_path.parent.exists()
    value_ds = captured["kwargs"]["value_deserializer"]
    key_ds = captured["kwargs"]["key_deserializer"]
    assert value_ds(json.dumps({"x": 1}).encode()) == {"x": 1}
    assert key_ds(None) is None
    assert key_ds(b"abc") == "abc"


def test_init_consumer_importerror_et_exception(consumer, monkeypatch):
    kafka_mod = types.ModuleType("kafka")
    monkeypatch.setitem(__import__("sys").modules, "kafka", kafka_mod)
    consumer._init_consumer()
    assert consumer.initialized is False

    kafka_mod.KafkaConsumer = object  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "services.kafka.bootstrap_retry.run_with_kafka_bootstrap_retry",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    captured = {"n": 0}

    def _ok(_exc):
        captured["n"] += 1

    monkeypatch.setattr("shared.sentry_init.capture_kafka_error", _ok)
    consumer._init_consumer()
    assert captured["n"] == 1

    def _sentry_fail(_exc):
        raise RuntimeError("sentry")

    monkeypatch.setattr("shared.sentry_init.capture_kafka_error", _sentry_fail)
    consumer._init_consumer()


def test_persist_event(consumer):
    record = SimpleNamespace(
        topic="notifications.dlq",
        partition=0,
        offset=7,
        key="k1",
        value={"hello": "é"},
    )
    consumer._persist_event(record)
    lines = consumer._persist_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])
    assert payload["topic"] == "notifications.dlq"
    assert payload["offset"] == 7
    assert payload["value"] == {"hello": "é"}


def test_update_dlq_metric(consumer, monkeypatch):
    class TP:
        def __init__(self, topic, partition_id):
            self.topic = topic
            self.partition = partition_id

        def __hash__(self):
            return hash((self.topic, self.partition))

        def __eq__(self, other):
            return (self.topic, self.partition) == (other.topic, other.partition)

    kafka_mod = types.ModuleType("kafka")
    kafka_mod.TopicPartition = TP  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "kafka", kafka_mod)
    set_size = MagicMock()
    monkeypatch.setattr("services.notifications.metrics.set_dlq_size", set_size)

    consumer._consumer = MagicMock()
    consumer._consumer.partitions_for_topic.return_value = None
    consumer._update_dlq_metric("notifications.dlq")
    set_size.assert_called_with(source="notifications.dlq", size=0, region="global")

    consumer._consumer.partitions_for_topic.return_value = {0, 1}

    def _end_offsets(tps):
        return {tps[0]: 10}

    consumer._consumer.end_offsets.side_effect = _end_offsets
    consumer._consumer.committed.return_value = 3
    consumer._update_dlq_metric("notifications.dlq")
    assert set_size.call_args.kwargs["size"] == 14

    consumer._consumer.partitions_for_topic.side_effect = RuntimeError("metric")
    consumer._update_dlq_metric("notifications.dlq")


def test_start_not_initialized(consumer, caplog):
    consumer.start()
    assert "cannot start" in caplog.text


def test_start_poll_vide_puis_stop(consumer):
    consumer._initialized = True
    mock_c = MagicMock()

    def _poll(timeout_ms=1000):
        consumer._running = False
        return {}

    mock_c.poll.side_effect = _poll
    consumer._consumer = mock_c
    consumer.start()
    mock_c.close.assert_called_once()


def test_start_persist_commit_et_metric(consumer, monkeypatch):
    consumer._initialized = True
    record = SimpleNamespace(
        topic="notifications.dlq", partition=0, offset=1, key=None, value={"a": 1}
    )
    mock_c = MagicMock()

    def _poll(timeout_ms=1000):
        consumer._running = False
        return {"tp0": [record]}

    mock_c.poll.side_effect = _poll
    consumer._consumer = mock_c
    metric = MagicMock()
    monkeypatch.setattr(consumer, "_update_dlq_metric", metric)
    consumer.start()
    mock_c.commit.assert_called_once()
    metric.assert_called_once_with("notifications.dlq")
    assert consumer._persist_path.exists()


def test_start_persist_echec_pas_de_commit(consumer):
    consumer._initialized = True
    record = SimpleNamespace(topic="t", partition=0, offset=1, key=None, value={})
    mock_c = MagicMock()

    def _poll(timeout_ms=1000):
        consumer._running = False
        return {"tp": [record]}

    mock_c.poll.side_effect = _poll
    consumer._consumer = mock_c
    consumer._persist_path = consumer._persist_path.parent / "missing" / "x.jsonl"
    consumer.start()
    mock_c.commit.assert_not_called()


def test_start_poll_race_et_commit_race(consumer, monkeypatch):
    consumer._initialized = True
    record = SimpleNamespace(topic="t", partition=0, offset=2, key=None, value={"v": 1})
    mock_c = MagicMock()
    state = {"n": 0}

    def _poll(timeout_ms=1000):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("Task is already done!")
        consumer._running = False
        return {"tp": [record]}

    mock_c.poll.side_effect = _poll
    mock_c.commit.side_effect = RuntimeError("task is already done")
    consumer._consumer = mock_c
    monkeypatch.setattr(consumer, "_update_dlq_metric", lambda _t: None)
    consumer.start()
    assert mock_c.commit.called


def test_start_poll_runtime_autre(consumer, monkeypatch):
    consumer._initialized = True
    mock_c = MagicMock()
    mock_c.poll.side_effect = RuntimeError("autre erreur")
    consumer._consumer = mock_c
    monkeypatch.setattr(
        "shared.sentry_init.is_kafka_connection_error", lambda _e: False
    )
    with pytest.raises(RuntimeError, match="autre erreur"):
        consumer.start()
    mock_c.close.assert_called_once()


def test_start_commit_runtime_autre_et_connexion(consumer, monkeypatch):
    consumer._initialized = True
    record = SimpleNamespace(topic="t", partition=0, offset=3, key=None, value={})
    mock_c = MagicMock()

    def _poll(timeout_ms=1000):
        consumer._running = False
        return {"tp": [record]}

    mock_c.poll.side_effect = _poll
    mock_c.commit.side_effect = RuntimeError("commit boom")
    consumer._consumer = mock_c
    captured = {"n": 0}
    monkeypatch.setattr("shared.sentry_init.is_kafka_connection_error", lambda _e: True)
    monkeypatch.setattr(
        "shared.sentry_init.capture_kafka_error",
        lambda _e: captured.__setitem__("n", 1),
    )
    with pytest.raises(RuntimeError, match="commit boom"):
        consumer.start()
    assert captured["n"] == 1


def test_shutdown_et_close(consumer):
    consumer._running = True
    consumer._shutdown_signal(15, None)
    assert consumer._running is False
    consumer._consumer = MagicMock()
    consumer.close()
    consumer._consumer.close.assert_called_once()
    consumer._consumer = None
    consumer.close()


def test_install_noise_filter():
    dlq._install_kafka_log_noise_filter()


def test_run_disabled(monkeypatch):
    monkeypatch.setattr(dlq, "KAFKA_ENABLED", False)
    monkeypatch.setattr(dlq, "_install_kafka_log_noise_filter", lambda: None)
    monkeypatch.setattr("shared.sentry_init.init_sentry", lambda: None)
    with pytest.raises(SystemExit) as exc:
        dlq.run_kafka_dlq_consumer()
    assert exc.value.code == 0


def test_run_not_initialized(monkeypatch):
    monkeypatch.setattr(dlq, "KAFKA_ENABLED", True)
    monkeypatch.setattr(dlq, "_install_kafka_log_noise_filter", lambda: None)
    monkeypatch.setattr("shared.sentry_init.init_sentry", lambda: None)
    monkeypatch.setattr(
        dlq, "KafkaDlqConsumer", lambda: SimpleNamespace(initialized=False)
    )
    with pytest.raises(SystemExit) as exc:
        dlq.run_kafka_dlq_consumer()
    assert exc.value.code == 1


def test_run_start(monkeypatch):
    monkeypatch.setattr(dlq, "KAFKA_ENABLED", True)
    monkeypatch.setattr(dlq, "_install_kafka_log_noise_filter", lambda: None)
    monkeypatch.setattr("shared.sentry_init.init_sentry", lambda: None)
    started = {"n": 0}
    monkeypatch.setattr(
        dlq,
        "KafkaDlqConsumer",
        lambda: SimpleNamespace(
            initialized=True, start=lambda: started.__setitem__("n", 1)
        ),
    )
    dlq.run_kafka_dlq_consumer()
    assert started["n"] == 1
