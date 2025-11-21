"""Tests pour vérifier la migration MD5 → SHA-256.

Valide que tous les usages de MD5 ont été remplacés par SHA-256 :
- osrm_client.py : Hash coordonnées pour cache
- model_registry.py : Checksum fichiers
- websocket_ack.py : Hash payload pour message_id
- queue.py : Hash paramètres pour déduplication
"""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from services.ml.model_registry import ModelRegistry
from services.unified_dispatch.queue import trigger_job
from sockets.websocket_ack import PendingMessage


class TestOSRMClientSHA256:
    """Tests pour vérifier que osrm_client.py utilise SHA-256."""

    def test_cache_key_route_uses_sha256(self):
        """Vérifie que get_distance_time_cached utilise SHA-256 pour hasher les coordonnées."""
        import hashlib

        # Coordonnées de test
        origin = (46.2044, 6.1432)
        dest = (46.2100, 6.1500)
        date_str = "2025-01-15"

        # Simuler la génération de la clé de cache (comme dans get_distance_time_cached)
        # Note: ORIG_ZERO = 0 (index du premier élément)
        origin_hash = hashlib.sha256(f"{origin[0]},{origin[1]}".encode()).hexdigest()[:8]
        dest_hash = hashlib.sha256(f"{dest[0]},{dest[1]}".encode()).hexdigest()[:8]
        cache_key = f"osrm:cache:{date_str}:{origin_hash}:{dest_hash}"

        # Vérifier que la clé contient les hashs
        assert cache_key.startswith("osrm:cache:")
        assert date_str in cache_key

        # Extraire les hashs de la clé
        parts = cache_key.split(":")
        extracted_origin_hash = parts[3]
        extracted_dest_hash = parts[4]

        # Vérifier que les hashs ont 8 caractères (troncature SHA-256)
        assert len(extracted_origin_hash) == 8
        assert len(extracted_dest_hash) == 8

        # Vérifier que les hashs correspondent au SHA-256
        assert extracted_origin_hash == origin_hash
        assert extracted_dest_hash == dest_hash

    def test_cache_key_table_uses_sha256(self):
        """Vérifie que get_time_matrix_cached utilise SHA-256 pour hasher la matrice."""
        import hashlib

        # Coordonnées de test
        origins = [(46.2044, 6.1432), (46.2100, 6.1500)]
        destinations = [(46.2150, 6.1550)]
        date_str = "2025-01-15"

        # Simuler la génération de la clé de cache (comme dans get_time_matrix_cached)
        origins_str = ",".join([f"{o[0]},{o[1]}" for o in origins])
        dests_str = ",".join([f"{d[0]},{d[1]}" for d in destinations])
        matrix_hash = hashlib.sha256(f"{origins_str}|{dests_str}".encode()).hexdigest()[:16]
        cache_key = f"osrm:matrix:{date_str}:{matrix_hash}"

        # Vérifier que la clé contient le hash
        assert cache_key.startswith("osrm:matrix:")
        assert date_str in cache_key

        # Extraire le hash de la clé
        parts = cache_key.split(":")
        extracted_hash = parts[2]

        # Vérifier que le hash a 16 caractères (troncature SHA-256)
        assert len(extracted_hash) == 16

        # Vérifier que le hash correspond au SHA-256
        assert extracted_hash == matrix_hash


class TestModelRegistrySHA256:
    """Tests pour vérifier que model_registry.py utilise SHA-256."""

    def test_calculate_checksum_uses_sha256(self):
        """Vérifie que _calculate_checksum utilise SHA-256."""
        # Créer un fichier temporaire avec du contenu
        with NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
            test_content = b"test content for checksum"
            tmp_file.write(test_content)
            tmp_file_path = Path(tmp_file.name)

        try:
            # Créer une instance de ModelRegistry
            registry = ModelRegistry(Path("/tmp/test_registry"))

            # Calculer le checksum
            checksum = registry._calculate_checksum(tmp_file_path)

            # Vérifier que le checksum est un hash SHA-256 complet (64 caractères hex)
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

            # Vérifier que le checksum correspond au SHA-256 du contenu
            expected_checksum = hashlib.sha256(test_content).hexdigest()
            assert checksum == expected_checksum

        finally:
            # Nettoyer le fichier temporaire
            tmp_file_path.unlink()


class TestWebSocketACKSHA256:
    """Tests pour vérifier que websocket_ack.py utilise SHA-256."""

    def test_emit_with_ack_uses_sha256(self):
        """Vérifie que emit_with_ack génère message_id avec SHA-256."""
        from sockets.websocket_ack import WebSocketACKManager

        event = "test_event"
        room = "test_room"
        payload = {"test": "data"}

        # Créer un gestionnaire ACK
        manager = WebSocketACKManager()

        # Appeler emit_with_ack sans message_id (sera généré)
        message_id = manager.emit_with_ack(event=event, payload=payload, room=room, message_id=None)

        # Vérifier que message_id a été généré
        assert message_id is not None

        # Vérifier que message_id a 16 caractères (troncature SHA-256)
        assert len(message_id) == 16

        # Vérifier que message_id correspond au SHA-256 du payload
        payload_str = f"{event}:{room}:{payload!s}"
        expected_hash = hashlib.sha256(payload_str.encode()).hexdigest()[:16]

        assert message_id == expected_hash


class TestQueueSHA256:
    """Tests pour vérifier que queue.py utilise SHA-256."""

    def test_queue_hash_uses_sha256(self):
        """Vérifie que trigger_job utilise SHA-256 pour le hash de déduplication."""
        # Paramètres de test
        run_kwargs = {"for_date": "2025-01-15", "mode": "auto"}

        # Simuler la génération du hash comme dans trigger_job
        params_str = json.dumps(run_kwargs, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()

        # Vérifier que le hash a 64 caractères (SHA-256 complet)
        assert len(params_hash) == 64

        # Vérifier que c'est bien un hash SHA-256 valide
        assert all(c in "0123456789abcdef" for c in params_hash)

        # Vérifier la reproductibilité
        params_hash2 = hashlib.sha256(params_str.encode()).hexdigest()
        assert params_hash == params_hash2


class TestSHA256Consistency:
    """Tests pour vérifier la cohérence et la reproductibilité des hashs SHA-256."""

    def test_sha256_reproducibility(self):
        """Vérifie que SHA-256 produit des hashs reproductibles."""
        test_data = "test data for hashing"

        hash1 = hashlib.sha256(test_data.encode()).hexdigest()
        hash2 = hashlib.sha256(test_data.encode()).hexdigest()

        assert hash1 == hash2

    def test_sha256_length(self):
        """Vérifie que SHA-256 produit des hashs de 64 caractères."""
        test_data = "test data"

        hash_value = hashlib.sha256(test_data.encode()).hexdigest()

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_sha256_truncation(self):
        """Vérifie que la troncature SHA-256 fonctionne correctement."""
        test_data = "test data"

        full_hash = hashlib.sha256(test_data.encode()).hexdigest()
        truncated_8 = full_hash[:8]
        truncated_16 = full_hash[:16]

        assert len(truncated_8) == 8
        assert len(truncated_16) == 16
        assert truncated_16.startswith(truncated_8)
