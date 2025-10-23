#!/usr/bin/env python3
# ruff: noqa: T201, DTZ001, DTZ007, DTZ005
"""
Convertit le fichier XLSB complet (1 année, 12 mois) en données RL.

Auteur: ATMR Project
Date: 22 octobre 2025
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pyxlsb  # pyright: ignore[reportMissingImports]
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from models import Driver
from shared.geo_utils import haversine_distance

# Mapping des initiales (confirmé avec l'utilisateur)
DRIVER_INITIALS_MAP = {
    "Y.L": "Yannis Labrot",
    "YL": "Yannis Labrot",
    "D.D": "Dris Daoudi",
    "DD": "Dris Daoudi",
    "G.B": "Giuseppe Bekasy",
    "GB": "Giuseppe Bekasy",
    "K.A": "Khalid Alaoui",
    "KA": "Khalid Alaoui",
    "A.B": "PONCTUEL_AB",  # Chauffeur ponctuel
    "AB": "PONCTUEL_AB",
    "D.J": "PONCTUEL_DJ",  # Chauffeur ponctuel
    "DJ": "PONCTUEL_DJ",
    "J.B": "Giuseppe Bekasy",  # Alias possible
    "JB": "Giuseppe Bekasy",
}


class AddressGeocoder:
    """Géocodeur d'adresses utilisant Nominatim."""

    def __init__(self, cache_file: str = "data/rl/geocode_cache_full_year.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {"User-Agent": "ATMR-Dispatch-App/1.0"}

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def geocode(self, address: str) -> tuple[float, float] | None:
        address_clean = re.sub(r"\s+", " ", str(address).strip())

        if address_clean in self.cache:
            cached = self.cache[address_clean]
            return (cached["lat"], cached["lon"])

        try:
            params = {
                "q": address_clean,
                "format": "json",
                "limit": 1,
                "countrycodes": "ch",
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

                    self.cache[address_clean] = {"lat": lat, "lon": lon}
                    self._save_cache()

                    time.sleep(1.1)  # Limite API

                    return (lat, lon)

        except Exception:
            pass

        return None


def convert_xlsb_full_year(
    xlsb_file: str = "transport_annee_complete.xlsb",
    output_file: str = "data/rl/historical_dispatches_full_year.json",
    min_courses_per_day: int = 3,
) -> None:
    """Convertit le fichier XLSB complet (12 mois) en données RL."""

    print("=" * 80)
    print("🔄 CONVERSION XLSB 1 ANNÉE → DONNÉES RL")
    print("=" * 80)
    print(f"📂 Fichier source : {xlsb_file}")
    print(f"📂 Fichier sortie : {output_file}")
    print()

    # Créer le géocodeur
    geocoder = AddressGeocoder()

    # Charger les drivers
    app = create_app()
    with app.app_context():
        drivers = Driver.query.all()
        driver_map = {}
        for d in drivers:
            if hasattr(d, "user") and d.user:
                full_name = f"{d.user.first_name} {d.user.last_name}"
                driver_map[full_name] = d.id

    print(f"👥 {len(driver_map)} chauffeurs mappés")
    print()

    # Lire le fichier XLSB
    dispatches_by_date = {}
    geocoding_success = 0
    geocoding_failed = 0
    total_rows_processed = 0

    with pyxlsb.open_workbook(xlsb_file) as wb:
        for sheet_name in wb.sheets:
            print(f"📄 Traitement feuille : {sheet_name}")

            rows = []
            with wb.get_sheet(sheet_name) as sheet:
                for row in sheet.rows():
                    rows.append([cell.v if cell.v is not None else "" for cell in row])

            print(f"   📊 {len(rows)} lignes trouvées")

            # Trouver la ligne d'en-têtes (contient "Nom/Prénom")
            header_row_idx = None
            for i, row in enumerate(rows[:10]):
                if any("Nom/Prénom" in str(cell) for cell in row):
                    header_row_idx = i
                    break

            if header_row_idx is None:
                print("   ⚠️  En-têtes non trouvés, skip")
                continue

            print(f"   ✅ En-têtes trouvés ligne {header_row_idx + 1}")

            # Traiter les lignes de données
            for idx, row in enumerate(rows[header_row_idx + 1:]):
                try:
                    if not row or len(row) < 6:
                        continue

                    # Extraire les données
                    client = str(row[0]).strip()
                    date_time_str = str(row[1]).strip()
                    # course_type = str(row[2]).strip()  # A/R ou simple (non utilisé pour l'instant)
                    pickup_addr = str(row[3]).strip()
                    dropoff_addr = str(row[4]).strip()
                    cft = str(row[5]).strip()

                    if not client or not date_time_str or not cft:
                        continue

                    # Extraire la date
                    date_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", date_time_str)
                    if not date_match:
                        continue

                    day, month, year = date_match.groups()
                    date_obj = datetime(int(year), int(month), int(day)).date()

                    # Extraire les heures
                    hours = re.findall(r"\b(\d{1,2}):(\d{2})\b", date_time_str)
                    if not hours:
                        continue

                    hour, minute = hours[0]
                    scheduled_time = datetime.combine(
                        date_obj, datetime.strptime(f"{hour}:{minute}", "%H:%M").time()
                    )

                    # Géocoder avec logs
                    if total_rows_processed % 20 == 0:
                        print(f"   🗺️  Géocodage ligne {idx} ({total_rows_processed+1} traitées)...", flush=True)

                    pickup_coords = geocoder.geocode(pickup_addr)
                    dropoff_coords = geocoder.geocode(dropoff_addr)

                    if pickup_coords and dropoff_coords:
                        geocoding_success += 1
                        print(f"   ✅ Géocodage réussi ligne {idx}", flush=True)
                    else:
                        geocoding_failed += 1
                        print(f"   ⚠️  Géocodage échoué ligne {idx}", flush=True)
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
                    cft_initials = cft.split("/")[0].strip()
                    driver_name = DRIVER_INITIALS_MAP.get(cft_initials)
                    driver_id = driver_map.get(driver_name) if driver_name else None

                    if not driver_id:
                        driver_id = list(driver_map.values())[0] if driver_map else 1

                    # Créer le booking
                    booking_data = {
                        "id": total_rows_processed + 1000,
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

                    # Ajouter au dispatch
                    date_key = date_obj.isoformat()
                    if date_key not in dispatches_by_date:
                        dispatches_by_date[date_key] = []

                    dispatches_by_date[date_key].append(booking_data)
                    total_rows_processed += 1

                except Exception:
                    continue

            print(f"   ✅ {total_rows_processed} courses traitées au total")
            print()

    print("=" * 80)
    print("📊 RÉSUMÉ CONVERSION")
    print("=" * 80)
    print(f"✅ Géocodage réussi  : {geocoding_success}")
    print(f"⚠️  Géocodage échoué : {geocoding_failed}")
    print(f"📦 Courses totales   : {total_rows_processed}")
    print(f"📅 Jours uniques     : {len(dispatches_by_date)}")
    print()

    # Créer les dispatches
    dispatches = []
    total_bookings = 0

    for date_key, bookings in sorted(dispatches_by_date.items()):
        if len(bookings) < min_courses_per_day:
            continue

        driver_loads = {}
        total_distance = 0.0

        for b in bookings:
            driver_id = b["assigned_driver_id"]
            driver_loads[driver_id] = driver_loads.get(driver_id, 0) + 1
            total_distance += b["distance_km"]

        if driver_loads:
            max_load = max(driver_loads.values())
            min_load = min(driver_loads.values())
            load_gap = max_load - min_load
        else:
            load_gap = 0

        quality_score = max(0, 100 - (load_gap * 15) - (total_distance * 0.3))

        dispatch = {
            "id": f"xlsb_{date_key}",
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

    # Statistiques
    if dispatches:
        avg_gap = sum(d["load_gap"] for d in dispatches) / len(dispatches)
        avg_distance = sum(d["total_distance_km"] for d in dispatches) / len(dispatches)

        print("📈 STATISTIQUES FINALES :")
        print(f"  - Total dispatches   : {len(dispatches)}")
        print(f"  - Total courses      : {total_bookings}")
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
        "source": "xlsb_full_year",
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

    print(f"✅ Données exportées : {output_path.absolute()}")
    print(f"📦 Taille fichier    : {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("🚀 PROCHAINE ÉTAPE : Réentraîner avec 15,000 épisodes !")
    print()


if __name__ == "__main__":
    convert_xlsb_full_year(
        xlsb_file="transport_annee_complete.xlsb",
        output_file="data/rl/historical_dispatches_full_year.json",
        min_courses_per_day=3,
    )

