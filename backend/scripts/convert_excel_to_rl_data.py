#!/usr/bin/env python3
# ruff: noqa: T201, DTZ001, DTZ007, DTZ005
"""
Convertit le fichier Excel de transport historique en données d'entraînement RL.

Fonctionnalités :
- Lecture du fichier Excel
- Géocodage des adresses (Nominatim API)
- Mapping des initiales chauffeurs → IDs
- Calcul des distances GPS
- Export au format JSON pour entraînement RL

Auteur: ATMR Project
Date: 21 octobre 2025
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from models import Driver
from shared.geo_utils import haversine_distance

# Mapping des initiales vers les noms (confirmé avec la DB)
DRIVER_INITIALS_MAP = {
    "Y.L": "Yannis Labrot",
    "YL": "Yannis Labrot",
    "D.D": "Dris Daoudi",
    "DD": "Dris Daoudi",
    "G.B": "Giuseppe Bekasy",
    "GB": "Giuseppe Bekasy",
    "K.A": "Khalid Alaoui",
    "KA": "Khalid Alaoui",
}


class AddressGeocoder:
    """Géocodeur d'adresses utilisant Nominatim (gratuit)."""

    def __init__(self, cache_file: str = "data/rl/geocode_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {"User-Agent": "ATMR-Dispatch-App/1.0"}

    def _load_cache(self) -> dict:
        """Charge le cache des adresses déjà géocodées."""
        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        """Sauvegarde le cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def geocode(self, address: str) -> tuple[float, float] | None:
        """
        Géocode une adresse en coordonnées GPS.

        Args:
            address: Adresse textuelle

        Returns:
            (latitude, longitude) ou None si échec
        """
        # Nettoyer l'adresse
        address_clean = re.sub(r"\s+", " ", str(address).strip())

        # Vérifier le cache
        if address_clean in self.cache:
            cached = self.cache[address_clean]
            return (cached["lat"], cached["lon"])

        # Appeler l'API Nominatim (limite: 1 req/sec)
        try:
            params = {
                "q": address_clean,
                "format": "json",
                "limit": 1,
                "countrycodes": "ch",  # Suisse uniquement
            }

            response = requests.get(
                self.base_url, params=params, headers=self.headers, timeout=10
            )

            if response.status_code == 200:
                results = response.json()
                if results:
                    result = results[0]
                    lat = float(result["lat"])
                    lon = float(result["lon"])

                    # Sauvegarder dans le cache
                    self.cache[address_clean] = {"lat": lat, "lon": lon}
                    self._save_cache()

                    # Respecter la limite de 1 req/sec
                    time.sleep(1.1)

                    return (lat, lon)

        except Exception as e:
            print(f"⚠️  Erreur géocodage '{address_clean[:50]}...': {e}")

        return None


def convert_excel_to_rl_data(
    excel_file: str = "transport.xlsx",
    output_file: str = "data/rl/historical_dispatches_from_excel.json",
    min_courses_per_day: int = 3,
) -> None:
    """
    Convertit le fichier Excel en données d'entraînement RL.

    Args:
        excel_file: Chemin du fichier Excel
        output_file: Chemin du fichier JSON de sortie
        min_courses_per_day: Nombre minimum de courses par jour
    """
    print("=" * 80)
    print("🔄 CONVERSION EXCEL → DONNÉES D'ENTRAÎNEMENT RL")
    print("=" * 80)
    print(f"📂 Fichier source : {excel_file}")
    print(f"📂 Fichier sortie : {output_file}")
    print()

    # Lire le fichier Excel
    df = pd.read_excel(excel_file, sheet_name="Feuil1")

    # Nettoyer les noms de colonnes (trim espaces)
    df.columns = df.columns.str.strip()

    print(f"📊 {len(df)} courses chargées depuis Excel")
    print(f"📋 Colonnes : {list(df.columns)}")
    print()

    # Créer le géocodeur
    geocoder = AddressGeocoder()
    print("🗺️  Géocodeur initialisé (cache actif)")
    print()

    # Charger les drivers de la DB pour le mapping
    app = create_app()
    with app.app_context():
        drivers = Driver.query.all()
        driver_map = {}
        for d in drivers:
            if hasattr(d, "user") and d.user:
                full_name = f"{d.user.first_name} {d.user.last_name}"
                driver_map[full_name] = d.id
                print(f"  Chauffeur trouvé : {full_name} (ID: {d.id})")

    print()
    print("🔄 Début du traitement...")
    print()

    # Grouper par date
    dispatches_by_date = {}
    geocoding_success = 0
    geocoding_failed = 0

    for row_idx, row in df.iterrows():
        try:
            idx = int(row_idx) if isinstance(row_idx, (int, float)) else 0  # pyright: ignore

            # Extraire la date
            date_str = str(row["Date et Heure prévues"]).split()[0]  # "01.10.2025"
            date_obj = datetime.strptime(date_str, "%d.%m.%Y").date()

            # Extraire les heures (peut contenir plusieurs heures)
            time_str = str(row["Date et Heure prévues"])
            hours = re.findall(r"\b(\d{1,2}):(\d{2})\b", time_str)

            if not hours:
                continue

            # Utiliser la première heure trouvée
            hour, minute = hours[0]
            scheduled_time = datetime.combine(
                date_obj, datetime.strptime(f"{hour}:{minute}", "%H:%M").time()
            )

            # Extraire les adresses
            pickup_addr = str(row["Adresse de départ"]).strip()
            dropoff_addr = str(row["Adresse d'arrivée"]).strip()

            # Géocoder
            pickup_coords = geocoder.geocode(pickup_addr)
            dropoff_coords = geocoder.geocode(dropoff_addr)

            if pickup_coords and dropoff_coords:
                geocoding_success += 1
            else:
                geocoding_failed += 1
                # Utiliser coordonnées par défaut (Genève centre)
                if not pickup_coords:
                    pickup_coords = (46.2044, 6.1432)
                if not dropoff_coords:
                    dropoff_coords = (46.2044, 6.1432)

            # Calculer distance
            distance_km = haversine_distance(
                pickup_coords[0], pickup_coords[1],
                dropoff_coords[0], dropoff_coords[1]
            )

            # Mapper le chauffeur
            cft_initials = str(row["CFT"]).strip().split("/")[
                0
            ]  # Prendre le premier si plusieurs
            driver_name = DRIVER_INITIALS_MAP.get(cft_initials)
            driver_id = driver_map.get(driver_name) if driver_name else None

            if not driver_id:
                # Fallback : utiliser l'ID du premier chauffeur trouvé
                driver_id = list(driver_map.values())[0] if driver_map else 1

            # Créer le booking
            booking_data = {
                "id": idx + 1000,  # ID unique
                "scheduled_time": scheduled_time.isoformat(),
                "pickup_lat": pickup_coords[0],
                "pickup_lon": pickup_coords[1],
                "dropoff_lat": dropoff_coords[0],
                "dropoff_lon": dropoff_coords[1],
                "pickup_location": pickup_addr[:100],
                "dropoff_location": dropoff_addr[:100],
                "distance_km": round(distance_km, 2),
                "estimated_duration_minutes": round((distance_km / 30.0) * 60, 1),
                "assigned_driver_id": driver_id,
                "actual_delay_minutes": 0,
            }

            # Ajouter au dispatch de la journée
            date_key = date_obj.isoformat()
            if date_key not in dispatches_by_date:
                dispatches_by_date[date_key] = []

            dispatches_by_date[date_key].append(booking_data)

            # Progress
            if (idx + 1) % 20 == 0:
                print(f"  ⏳ Traité {idx + 1}/{len(df)} courses...")

        except Exception as e:
            print(f"  ⚠️  Erreur ligne {idx + 1}: {e}")
            continue

    print()
    print("=" * 80)
    print("📊 RÉSUMÉ DU TRAITEMENT")
    print("=" * 80)
    print(f"✅ Géocodage réussi  : {geocoding_success} adresses")
    print(f"⚠️  Géocodage échoué : {geocoding_failed} adresses (coord. par défaut)")
    print(f"📅 Jours uniques     : {len(dispatches_by_date)}")
    print()

    # Créer les dispatches au format RL
    dispatches = []
    total_bookings = 0

    for date_key, bookings in sorted(dispatches_by_date.items()):
        if len(bookings) < min_courses_per_day:
            continue

        # Calculer les métriques du dispatch
        driver_loads = {}
        total_distance = 0.0

        for b in bookings:
            driver_id = b["assigned_driver_id"]
            driver_loads[driver_id] = driver_loads.get(driver_id, 0) + 1
            total_distance += b["distance_km"]

        # Écart de charge
        if driver_loads:
            max_load = max(driver_loads.values())
            min_load = min(driver_loads.values())
            load_gap = max_load - min_load
        else:
            load_gap = 0

        # Score qualité
        quality_score = max(
            0,
            100 - (load_gap * 15) - (total_distance * 0.3),
        )

        # Créer le dispatch
        dispatch = {
            "id": f"excel_{date_key}",
            "date": date_key,
            "num_bookings": len(bookings),
            "num_drivers": len(driver_loads),
            "driver_loads": driver_loads,
            "load_gap": load_gap,
            "total_distance_km": round(total_distance, 2),
            "avg_distance_per_booking": round(total_distance / len(bookings), 2),
            "retards_count": 0,
            "quality_score": round(quality_score, 2),
            "bookings": bookings,
            "drivers": [
                {
                    "id": did,
                    "name": f"Driver {did}",
                    "is_emergency": False,
                    "num_assignments": driver_loads.get(did, 0),
                }
                for did in driver_loads
            ],
        }

        dispatches.append(dispatch)
        total_bookings += len(bookings)

    # Statistiques finales
    print("📊 DISPATCHES CRÉÉS :")
    print("-" * 80)
    for d in dispatches[:5]:
        print(
            f"  {d['date']} : {d['num_bookings']} courses, {d['num_drivers']} chauffeurs, gap={d['load_gap']}"
        )

    if len(dispatches) > 5:
        print(f"  ... et {len(dispatches) - 5} autres dispatches")
    print()

    if dispatches:
        avg_gap = sum(d["load_gap"] for d in dispatches) / len(dispatches)
        avg_distance = sum(d["total_distance_km"] for d in dispatches) / len(
            dispatches
        )

        print("📈 Statistiques globales :")
        print(f"  - Total dispatches   : {len(dispatches)}")
        print(f"  - Total bookings     : {total_bookings}")
        print(f"  - Écart moyen        : {avg_gap:.2f} courses")
        print(f"  - Distance moyenne   : {avg_distance:.1f} km/dispatch")
        print()

    # Sauvegarder
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "company_id": 1,
        "period": {
            "start": min(dispatches_by_date.keys()) if dispatches_by_date else None,
            "end": max(dispatches_by_date.keys()) if dispatches_by_date else None,
        },
        "exported_at": datetime.now().isoformat(),
        "source": "excel",
        "total_dispatches": len(dispatches),
        "total_bookings": total_bookings,
        "statistics": {
            "avg_load_gap": round(avg_gap, 2) if dispatches else 0,
            "avg_total_distance": round(avg_distance, 2) if dispatches else 0,
            "geocoding_success_rate": round(
                (geocoding_success / (geocoding_success + geocoding_failed)) * 100, 1
            )
            if (geocoding_success + geocoding_failed) > 0
            else 0,
        },
        "dispatches": dispatches,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Données exportées vers : {output_path.absolute()}")
    print(f"📦 Taille du fichier     : {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("🚀 PROCHAINE ÉTAPE : Réentraîner le modèle RL !")
    print()
    print("   docker exec atmr-api-1 python backend/scripts/rl_train_offline.py \\")
    print("     --data data/rl/historical_dispatches_from_excel.json \\")
    print("     --episodes 10000 \\")
    print("     --output data/rl/models/dispatch_optimized_v2.pth")
    print()


if __name__ == "__main__":
    convert_excel_to_rl_data(
        excel_file="transport.xlsx",
        output_file="data/rl/historical_dispatches_from_excel.json",
        min_courses_per_day=3,
    )

