#!/usr/bin/env python3
# ruff: noqa: T201, DTZ001, DTZ007, W293
"""Conversion XLSB 1 année → Données RL (Version corrigée)"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pyxlsb  # pyright: ignore[reportMissingImports]
import requests

# Add parent to path
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
        # Normalisation basique
        address_norm = str(address or "").replace("·", " ")
        address_norm = re.sub(r"\s+", " ", address_norm).strip()
        if not address_norm:
            return None

        # Cache
        if address_norm in self.cache:
            cached = self.cache[address_norm]
            return (cached["lat"], cached["lon"])

        # Paramètres centrés sur Genève (viewbox) + bounded
        base_params = {
            "format": "json",
            "limit": 1,
            "countrycodes": "ch",
            # Genève approx viewbox (lon,lat)
            "viewbox": "5.96,46.33,6.33,46.30",  # Ouest/Est, Nord/Sud (approx)
            "bounded": 1,
        }

        # Stratégies de requête: 1) adresse complète, 2) CP + Ville
        queries: list[str] = [address_norm]

        # Extraire CP + Ville pour fallback
        m = re.search(r"(\b\d{4}\b)\s+([A-Za-zÀ-ÿ'\-\s]+)$", address_norm)
        if m:
            cp, ville = m.groups()
            queries.append(f"{cp} {ville}")

        # Retries simples avec backoff
        for qi, q in enumerate(queries):
            for attempt in range(2):
                try:
                    params = dict(base_params)
                    params["q"] = q
                    resp = requests.get(self.base_url, params=params, headers=self.headers, timeout=8)
                    if resp.status_code == 200:
                        results = resp.json()
                        if results:
                            result = results[0]
                            lat = float(result.get("lat"))
                            lon = float(result.get("lon"))
                            self.cache[address_norm] = {"lat": lat, "lon": lon}
                            self._save_cache()
                            time.sleep(1.0)
                            return (lat, lon)
                except Exception:
                    pass
                # petit backoff
                time.sleep(0.6 + 0.4 * attempt)

        return None


def extract_address(full_text: str) -> str:
    """Extrait l'adresse d'un texte avec nom de lieu + adresse."""
    if not full_text:
        return ""
    
    # Normaliser les séparateurs (beaucoup d'espaces → nouvelle ligne)
    text = str(full_text).replace("\u00A0", " ")  # espaces insécables
    text = re.sub(r"\s{3,}", "\n", text)  # blocs d'espaces → saut de ligne
    text = re.sub(r"\s*\n\s*", "\n", text)  # nettoyer

    # Découper en lignes non vides
    lines = [ln.strip() for ln in text.split("\n") if ln and ln.strip()]

    # Heuristique: garder les 2 dernières lignes (rue, puis CP Ville)
    if len(lines) >= 2:
        candidate = " ".join(lines[-2:])
    else:
        candidate = lines[0] if lines else ""

    # Nettoyage final
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def convert_xlsb_full_year_v2(
    xlsb_file: str = "transport_annee_complete.xlsb",
    output_file: str = "data/rl/historical_dispatches_full_year.json",
    min_courses_per_day: int = 3,
) -> None:
    """Convertit le fichier XLSB 1 année en données RL (version corrigée)."""

    print("=" * 80)
    print("🔄 CONVERSION XLSB 1 ANNÉE → DONNÉES RL (V2)")
    print("=" * 80)
    print(f"📂 Fichier source : {xlsb_file}")
    print(f"📂 Fichier sortie : {output_file}")
    print()

    # Init Flask pour accès DB
    app = create_app()

    with app.app_context():
        # Charger les chauffeurs
        drivers = Driver.query.filter_by(company_id=2).all()
        driver_map = {f"{d.user.first_name} {d.user.last_name}": d.id for d in drivers if d.user}
        print(f"👥 {len(driver_map)} chauffeurs mappés")
        print()

        # Géocodeur
        geocoder = AddressGeocoder()

        # Statistiques
        geocoding_success = 0
        geocoding_failed = 0
        total_rows_processed = 0

        # Résultats
        all_bookings = []

        # Lire le fichier XLSB
        wb = pyxlsb.open_workbook(xlsb_file)

        for sheet_name in wb.sheets:
            print(f"📄 Traitement feuille : {sheet_name}")

            # Lire les lignes
            with wb.get_sheet(sheet_name) as sheet:
                rows = list(sheet.rows())

            print(f"   📊 {len(rows)} lignes trouvées")

            if len(rows) < 3:
                print("   ⚠️  Feuille vide, ignorée")
                continue

            # ✨ CORRECTION : Les en-têtes sont à la ligne 2 (index 1)
            header_row_idx = 1
            data_start_idx = 2

            print(f"   ✅ En-têtes trouvés ligne {header_row_idx + 1}")
            print()

            # Traiter chaque ligne de données
            for row_idx in range(data_start_idx, len(rows)):
                row = rows[row_idx]
                cells = [cell.v for cell in row]

                if len(cells) < 6:
                    continue

                try:
                    # Colonnes (index 0-based):
                    # 0 = Nom/Prénom
                    # 1 = Date / Heure (format: "DD.MM.YYYY  HH:MM  HH:MM")
                    # 2 = Course (A/R ou simple)
                    # 3 = Départ
                    # 4 = Arrivée
                    # 5 = CFT (chauffeur)

                    client_name = str(cells[0]).strip() if cells[0] else ""
                    date_time_str = str(cells[1]).strip() if cells[1] else ""
                    course_type = str(cells[2]).strip() if cells[2] else ""
                    departure_text = str(cells[3]) if cells[3] else ""
                    arrival_text = str(cells[4]) if cells[4] else ""
                    cft = str(cells[5]).strip() if cells[5] else ""

                    # Ignorer les lignes vides ou d'en-tête
                    if not client_name or not date_time_str or client_name.lower() in ["nom", "prénom", "nom/prénom"]:
                        continue

                    # Extraire la date (format: DD.MM.YYYY)
                    date_match = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", date_time_str)
                    if not date_match:
                        continue

                    day, month, year = date_match.groups()
                    date_obj = datetime(int(year), int(month), int(day)).date()

                    # Extraire les heures (format: "DD.MM.YYYY  09:15  16:00")
                    hours = re.findall(r"\b(\d{1,2}):(\d{2})\b", date_time_str)
                    if not hours:
                        continue

                    # Géocoder (avec logs)
                    if total_rows_processed % 10 == 0:
                        print(f"   🗺️  Ligne {row_idx} ({total_rows_processed} traitées)...", flush=True)

                    pickup_addr = extract_address(departure_text)
                    dropoff_addr = extract_address(arrival_text)

                    pickup_coords = geocoder.geocode(pickup_addr)
                    dropoff_coords = geocoder.geocode(dropoff_addr)

                    if pickup_coords and dropoff_coords:
                        geocoding_success += 1
                    else:
                        geocoding_failed += 1
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

                    # Créer les bookings (1 ou 2 selon A/R)
                    if course_type.upper() == "A/R" and len(hours) >= 2:
                        # ALLER
                        hour_aller, min_aller = hours[0]
                        scheduled_aller = datetime.combine(
                            date_obj, datetime.strptime(f"{hour_aller}:{min_aller}", "%H:%M").time()
                        )

                        booking_aller = {
                            "id": total_rows_processed + 1000,
                            "client_name": client_name,
                            "pickup_address": pickup_addr,
                            "dropoff_address": dropoff_addr,
                            "pickup_lat": pickup_coords[0],
                            "pickup_lon": pickup_coords[1],
                            "dropoff_lat": dropoff_coords[0],
                            "dropoff_lon": dropoff_coords[1],
                            "scheduled_time": scheduled_aller.isoformat(),
                            "distance_km": distance_km,
                            "driver_id": driver_id,
                            "driver_name": driver_name or "Inconnu",
                        }
                        all_bookings.append(booking_aller)
                        total_rows_processed += 1

                        # RETOUR
                        hour_retour, min_retour = hours[1]
                        scheduled_retour = datetime.combine(
                            date_obj, datetime.strptime(f"{hour_retour}:{min_retour}", "%H:%M").time()
                        )

                        booking_retour = {
                            "id": total_rows_processed + 1000,
                            "client_name": client_name,
                            "pickup_address": dropoff_addr,  # Inversé
                            "dropoff_address": pickup_addr,  # Inversé
                            "pickup_lat": dropoff_coords[0],
                            "pickup_lon": dropoff_coords[1],
                            "dropoff_lat": pickup_coords[0],
                            "dropoff_lon": pickup_coords[1],
                            "scheduled_time": scheduled_retour.isoformat(),
                            "distance_km": distance_km,
                            "driver_id": driver_id,
                            "driver_name": driver_name or "Inconnu",
                        }
                        all_bookings.append(booking_retour)
                        total_rows_processed += 1

                    else:
                        # Course simple
                        hour, minute = hours[0]
                        scheduled_time = datetime.combine(
                            date_obj, datetime.strptime(f"{hour}:{minute}", "%H:%M").time()
                        )

                        booking = {
                            "id": total_rows_processed + 1000,
                            "client_name": client_name,
                            "pickup_address": pickup_addr,
                            "dropoff_address": dropoff_addr,
                            "pickup_lat": pickup_coords[0],
                            "pickup_lon": pickup_coords[1],
                            "dropoff_lat": dropoff_coords[0],
                            "dropoff_lon": dropoff_coords[1],
                            "scheduled_time": scheduled_time.isoformat(),
                            "distance_km": distance_km,
                            "driver_id": driver_id,
                            "driver_name": driver_name or "Inconnu",
                        }
                        all_bookings.append(booking)
                        total_rows_processed += 1

                except Exception as e:
                    print(f"   ⚠️  Erreur ligne {row_idx}: {e}", flush=True)
                    continue

            print(f"   ✅ {total_rows_processed} courses traitées au total")
            print()

        wb.close()

        # Grouper par date
        bookings_by_date = {}
        for booking in all_bookings:
            date = booking["scheduled_time"][:10]  # YYYY-MM-DD
            if date not in bookings_by_date:
                bookings_by_date[date] = []
            bookings_by_date[date].append(booking)

        # Créer dispatches
        dispatches = []
        for date, bookings in sorted(bookings_by_date.items()):
            if len(bookings) < min_courses_per_day:
                continue

            dispatch = {
                "date": date,
                "num_bookings": len(bookings),
                "bookings": bookings,
            }
            dispatches.append(dispatch)

        # Sauvegarder
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"dispatches": dispatches}, f, indent=2, ensure_ascii=False)

        # Résumé
        print("=" * 80)
        print("📊 RÉSUMÉ CONVERSION V2")
        print("=" * 80)
        print(f"✅ Géocodage réussi  : {geocoding_success}")
        print(f"⚠️  Géocodage échoué : {geocoding_failed}")
        print(f"📦 Courses totales   : {len(all_bookings)}")
        print(f"📅 Dispatches créés  : {len(dispatches)}")
        print()
        print(f"✅ Données exportées : {output_path}")
        print(f"📦 Taille fichier    : {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print()
        print("🚀 PROCHAINE ÉTAPE : Réentraîner avec 15,000 épisodes !")
        print()


if __name__ == "__main__":
    convert_xlsb_full_year_v2()

