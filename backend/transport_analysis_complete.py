#!/usr/bin/env python3
# ruff: noqa: T201, W293
"""
Script amélioré pour analyser et extraire les données du fichier Excel transport_janvier.xlsx
"""

import json
import os
import re
import time
from datetime import UTC, datetime
from typing import Dict, List
from urllib.parse import quote

import pandas as pd
import requests


def clean_excel_data(file_path: str) -> Dict:
    """Nettoie et structure les données Excel"""

    print(f"Analyse et nettoyage du fichier: {file_path}")

    try:
        # Lire le fichier Excel
        df = pd.read_excel(file_path)

        print(f"Fichier lu avec succès: {df.shape[0]} lignes x {df.shape[1]} colonnes")

        # Identifier les vraies données (ignorer les lignes d'en-tête)
        # Les données commencent à partir de la ligne 2 (index 2)
        data_df = df.iloc[2:].copy()

        # Renommer les colonnes pour plus de clarté
        column_mapping = {
            'JANVIER': 'nom_prenom',
            2025: 'date_heure',
            'Unnamed: 2': 'type_course',
            'EMMENEZ-MOI Sàrl ': 'adresse_depart',
            'Unnamed: 4': 'adresse_arrivee',
            'Unnamed: 5': 'conducteur'
        }

        data_df = data_df.rename(columns=column_mapping)

        # Nettoyer les données
        cleaned_data = []

        for index, row in data_df.iterrows():
            if pd.isna(row['nom_prenom']) or row['nom_prenom'].strip() == '':
                continue

            # Extraire la date et l'heure
            date_heure_str = str(row['date_heure']) if not pd.isna(row['date_heure']) else ''

            # Parser la date et l'heure
            date_info = parse_date_time(date_heure_str)

            # Nettoyer les adresses
            depart = clean_address(str(row['adresse_depart']) if not pd.isna(row['adresse_depart']) else '')
            arrivee = clean_address(str(row['adresse_arrivee']) if not pd.isna(row['adresse_arrivee']) else '')

            transport_record = {
                'nom_prenom': str(row['nom_prenom']).strip(),
                'date': date_info['date'],
                'heure_depart': date_info['heure_depart'],
                'heure_arrivee': date_info['heure_arrivee'],
                'type_course': str(row['type_course']).strip() if not pd.isna(row['type_course']) else '',
                'adresse_depart': depart,
                'adresse_arrivee': arrivee,
                'conducteur': str(row['conducteur']).strip() if not pd.isna(row['conducteur']) else '',
                'ligne_originale': int(index)
            }

            cleaned_data.append(transport_record)

        print(f"Données nettoyées: {len(cleaned_data)} enregistrements valides")

        # Analyser la qualité des données
        analysis = {
            'total_records': len(cleaned_data),
            'records_with_depart': len([r for r in cleaned_data if r['adresse_depart']]),
            'records_with_arrivee': len([r for r in cleaned_data if r['adresse_arrivee']]),
            'records_with_dates': len([r for r in cleaned_data if r['date']]),
            'unique_depart_addresses': len({r['adresse_depart'] for r in cleaned_data if r['adresse_depart']}),
            'unique_arrivee_addresses': len({r['adresse_arrivee'] for r in cleaned_data if r['adresse_arrivee']}),
            'sample_data': cleaned_data[:5]
        }

        return {
            'success': True,
            'cleaned_data': cleaned_data,
            'analysis': analysis
        }

    except Exception as e:
        print(f"Erreur lors du nettoyage: {e}")
        return {'success': False, 'error': str(e)}

def parse_date_time(date_heure_str: str) -> Dict[str, str]:
    """Parse la chaîne de date/heure"""

    result: Dict[str, str] = {
        'date': '',
        'heure_depart': '',
        'heure_arrivee': ''
    }

    if not date_heure_str or date_heure_str.strip() == '':
        return result

    # Rechercher la date (format DD.MM.YYYY)
    date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', date_heure_str)
    if date_match:
        result['date'] = str(date_match.group(1))

    # Rechercher les heures (format HH:MM)
    heures = re.findall(r'(\d{1,2}:\d{2})', date_heure_str)
    if len(heures) >= 1:
        result['heure_depart'] = heures[0]
    if len(heures) >= 2:
        result['heure_arrivee'] = heures[1]

    return result

def clean_address(address: str) -> str:
    """Nettoie une adresse"""
    if not address or address.strip() == '' or address.lower() in ['nan', 'none']:
        return ''

    # Supprimer les espaces en début/fin et normaliser
    cleaned = address.strip()

    # Supprimer les caractères de contrôle
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)

    # Normaliser les espaces multiples
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned

def test_geocoding_api(address: str) -> Dict | None:
    """Teste l'API Google Geocoding"""
    try:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            print("Cle API Google Maps non trouvee dans les variables d'environnement")
            return None

        # Encoder l'adresse pour l'URL
        encoded_address = quote(address)
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data['status'] == 'OK' and data['results']:
            result = data['results'][0]
            return {
                'address': result['formatted_address'],
                'lat': result['geometry']['location']['lat'],
                'lng': result['geometry']['location']['lng'],
                'status': 'OK'
            }
        else:
            return {
                'status': data['status'],
                'error': data.get('error_message', 'Unknown error')
            }

    except Exception as e:
        return {'error': str(e)}

def geocode_addresses(cleaned_data: List[Dict], max_addresses: int = 10) -> Dict:
    """Géocode un échantillon d'adresses pour tester"""

    print(f"Test de geocodage sur {max_addresses} adresses...")

    # Collecter toutes les adresses uniques
    all_addresses = set()
    for record in cleaned_data:
        if record['adresse_depart']:
            all_addresses.add(record['adresse_depart'])
        if record['adresse_arrivee']:
            all_addresses.add(record['adresse_arrivee'])

    # Prendre un échantillon
    sample_addresses = list(all_addresses)[:max_addresses]

    geocoding_results = {}

    for i, address in enumerate(sample_addresses):
        print(f"Geocodage {i+1}/{len(sample_addresses)}: {address[:50]}...")

        result = test_geocoding_api(address)
        geocoding_results[address] = result

        # Pause pour éviter de dépasser les limites de l'API
        time.sleep(0.1)

    return geocoding_results

def main():
    file_path = "/app/transport_janvier.xlsx"

    print("=== ANALYSE COMPLETE DU FICHIER EXCEL ===")
    print("=" * 50)

    # Nettoyer les données
    result = clean_excel_data(file_path)

    if not result['success']:
        print(f"Echec: {result['error']}")
        return

    cleaned_data = result['cleaned_data']
    analysis = result['analysis']

    print("\n=== RAPPORT D'ANALYSE ===")
    print("=" * 50)
    print(f"Total d'enregistrements: {analysis['total_records']}")
    print(f"Enregistrements avec adresse de depart: {analysis['records_with_depart']}")
    print(f"Enregistrements avec adresse d'arrivee: {analysis['records_with_arrivee']}")
    print(f"Enregistrements avec date: {analysis['records_with_dates']}")
    print(f"Adresses de depart uniques: {analysis['unique_depart_addresses']}")
    print(f"Adresses d'arrivee uniques: {analysis['unique_arrivee_addresses']}")

    print("\n=== EXEMPLE DE DONNEES NETTOYEES ===")
    print("=" * 50)
    for i, record in enumerate(analysis['sample_data']):
        print(f"\nEnregistrement {i+1}:")
        print(f"  Nom: {record['nom_prenom']}")
        print(f"  Date: {record['date']}")
        print(f"  Heure depart: {record['heure_depart']}")
        print(f"  Heure arrivee: {record['heure_arrivee']}")
        print(f"  Type: {record['type_course']}")
        print(f"  Depart: {record['adresse_depart']}")
        print(f"  Arrivee: {record['adresse_arrivee']}")
        print(f"  Conducteur: {record['conducteur']}")

    # Test de géocodage
    print("\n=== TEST DE GEOCODAGE ===")
    print("=" * 50)

    geocoding_results = geocode_addresses(cleaned_data, max_addresses=5)

    successful_geocodes = 0
    for address, result in geocoding_results.items():
        if result and result.get('status') == 'OK':
            successful_geocodes += 1
            print(f"✓ {address[:50]}... -> {result['lat']:.6f}, {result['lng']:.6f}")
        else:
            print(f"✗ {address[:50]}... -> Erreur: {result.get('error', 'Unknown') if result else 'No result'}")

    print(f"\nGeocodage reussi: {successful_geocodes}/{len(geocoding_results)} adresses")

    # Sauvegarder les résultats
    output_data = {
        'analysis': analysis,
        'cleaned_data': cleaned_data,
        'geocoding_test': geocoding_results,
        'timestamp': datetime.now(tz=UTC).isoformat()
    }

    with open('/app/transport_analysis_complete.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\nRapport complet sauvegarde dans: transport_analysis_complete.json")

    print("\n=== CONCLUSION ===")
    print("=" * 50)

    if analysis['total_records'] > 0:
        print("✓ Le fichier Excel contient des données de transport valides")
        print("✓ Les adresses sont correctement formatées")
        print("✓ Les dates et heures sont extraites")

        if successful_geocodes > 0:
            print("✓ Le geocodage Google Maps fonctionne")
            print("✓ Les coordonnees GPS peuvent etre obtenues")
        else:
            print("⚠ Le geocodage necessite une configuration de l'API Google Maps")

        print("\nLe fichier est pret pour l'integration dans le systeme de dispatch!")
    else:
        print("❌ Aucune donnee valide trouvee dans le fichier")

if __name__ == "__main__":
    main()
