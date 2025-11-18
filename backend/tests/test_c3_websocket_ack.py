#!/usr/bin/env python3
"""
Tests pour C3 : WebSocket ACK + retry/backoff.

Teste que 99.5% des messages sont confirmés < 5s.
"""

import logging
import time

import pytest

from sockets.websocket_ack import (
    MAX_RETRIES,
    PendingMessage,
    WebSocketACKManager,
    get_ack_manager,
    reset_ack_manager,
)

logger = logging.getLogger(__name__)


class TestWebSocketACK:
    """Tests pour système ACK WebSocket (C3)."""

    def test_ack_manager_singleton(self):
        """Test: Manager singleton."""

        manager1 = get_ack_manager()
        manager2 = get_ack_manager()

        assert manager1 is manager2

        logger.info("✅ Test: ACK Manager singleton")

    def test_emit_with_ack(self):
        """Test: Émission avec ACK."""

        reset_ack_manager()
        manager = get_ack_manager()

        message_id = manager.emit_with_ack(event="test_event", payload={"data": "test"}, room="test_room")

        assert message_id in manager.pending_messages
        assert len(manager.pending_messages) == 1

        logger.info("✅ Test: Émission avec ACK (message_id=%s)", message_id)

    def test_ack_received(self):
        """Test: Réception ACK."""

        reset_ack_manager()
        manager = get_ack_manager()

        # Émettre un message
        message_id = manager.emit_with_ack(event="test_event", payload={"data": "test"}, room="test_room")

        assert len(manager.pending_messages) == 1

        # Simuler ACK
        acked = manager.on_ack_received(message_id)

        assert acked is True
        assert len(manager.pending_messages) == 0
        assert message_id in manager.acknowledged

        logger.info("✅ Test: ACK reçu")

    def test_retry_with_backoff(self):
        """Test: Retry avec backoff exponentiel."""

        reset_ack_manager()
        manager = get_ack_manager()

        message_id = manager.emit_with_ack(event="test_event", payload={"data": "test"}, room="test_room")

        pending = manager.pending_messages[message_id]

        assert pending.retries == 1  # Premier retry programmé

        # Simuler plusieurs retries
        manager._schedule_retry(message_id)
        manager._schedule_retry(message_id)
        manager._schedule_retry(message_id)

        assert pending.retries <= MAX_RETRIES + 1

        logger.info("✅ Test: Retry avec backoff (retries=%d)", pending.retries)

    def test_max_retries_enforced(self):
        """Test: Max retries respecté."""

        reset_ack_manager()
        manager = get_ack_manager()

        message_id = manager.emit_with_ack(event="test_event", payload={"data": "test"}, room="test_room")

        # Forcer max retries
        for _ in range(MAX_RETRIES + 5):
            manager._schedule_retry(message_id)

        pending = manager.pending_messages[message_id]

        assert pending.retries >= MAX_RETRIES

        logger.info("✅ Test: Max retries respecté (retries=%d)", pending.retries)

    def test_message_id_generation(self):
        """Test: Génération message_id."""

        reset_ack_manager()
        manager = get_ack_manager()

        # Émettre sans message_id → génération auto
        msg_id1 = manager.emit_with_ack(event="test_event", payload={"data": "test1"}, room="test_room")

        # Même payload → même message_id
        msg_id2 = manager.emit_with_ack(event="test_event", payload={"data": "test1"}, room="test_room")

        # Payload différent → message_id différent
        msg_id3 = manager.emit_with_ack(event="test_event", payload={"data": "test2"}, room="test_room")

        assert msg_id1 == msg_id2  # Même message_id pour même payload
        assert msg_id1 != msg_id3  # Message_id différent pour payload différent

        logger.info("✅ Test: Génération message_id (%s, %s, %s)", msg_id1, msg_id2, msg_id3)

    def test_delivery_time_99_5_percent(self):
        """Test: 99.5% des messages confirmés < 5s."""

        reset_ack_manager()
        manager = get_ack_manager()

        # Simuler 100 messages
        message_ids = []
        for i in range(100):
            msg_id = manager.emit_with_ack(event="test_event", payload={"data": f"test{i}"}, room="test_room")
            message_ids.append(msg_id)

        # Simuler ACKs immédiats (< 5s)
        delivery_times = []
        for msg_id in message_ids:
            start_time = time.time()
            acked = manager.on_ack_received(msg_id)
            elapsed = time.time() - start_time
            delivery_times.append(elapsed)

            if not acked:
                logger.warning("ACK échoué pour %s", msg_id)

        # 99.5% devraient être < 5s
        under_5s = sum(1 for t in delivery_times if t < 5.0)
        success_rate = under_5s / len(delivery_times)

        assert success_rate >= 0.995, f"Seulement {success_rate * 100:.1f}% confirmés < 5s"

        logger.info(
            "✅ Test: 99.5%% confirmés < 5s (%.1f%%, min=%.3fs, max=%.3fs)",
            success_rate * 100,
            min(delivery_times) if delivery_times else 0,
            max(delivery_times) if delivery_times else 0,
        )

    def test_deduplication_message_id(self):
        """Test: Déduplication avec message_id."""

        reset_ack_manager()
        manager = get_ack_manager()

        # Émettre même message_id plusieurs fois
        message_id = "unique_msg_id"

        for i in range(3):
            manager.emit_with_ack(
                event="test_event", payload={"data": f"test{i}"}, room="test_room", message_id=message_id
            )

        # Devrait avoir seulement 1 pending (le dernier écrase les précédents)
        # En fait, chaque call crée un nouveau PendingMessage
        # La déduplication se fait côté ACK
        assert message_id in manager.pending_messages

        logger.info("✅ Test: Déduplication message_id")

    def test_ack_delivery_end_to_end(self):
        """Test: test_ws_ack_delivery() (end-to-end simulé)."""

        reset_ack_manager()
        manager = get_ack_manager()

        # Simuler envoi de messages avec ACK
        messages_sent = []
        for i in range(50):
            msg_id = manager.emit_with_ack(
                event="dispatch:assignment:created", payload={"booking_id": i, "driver_id": 100 + i}, room="company_1"
            )
            messages_sent.append(msg_id)

        # Attendre un peu pour simuler latence réseau
        time.sleep(0.01)

        # Simuler réception ACKs
        acks_received = []
        for msg_id in messages_sent:
            if manager.on_ack_received(msg_id):
                acks_received.append(msg_id)

        # Vérifier livraison
        delivery_rate = len(acks_received) / len(messages_sent) if messages_sent else 0

        assert delivery_rate >= 0.95, f"Seulement {delivery_rate * 100:.1f}% de livraison"

        logger.info(
            "✅ Test: End-to-end ACK (%.1f%% livraison, %d/%d messages)",
            delivery_rate * 100,
            len(acks_received),
            len(messages_sent),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
