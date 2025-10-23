#!/usr/bin/env python3
"""
Script pour analyser le fichier Excel transport_janvier.xlsx
et vérifier la qualité des données pour le système de dispatch
"""

import pandas as pd
import json
import requests
from typing import Dict, Optional
import os

def analyze_excel_file(file_path: str) -> Dict:
    """Analyse le fichier Excel et retourne un rapport détaillé"""
    
    print(f"Analyse du fichier: {file_path}")
    
    try:
        # Lire le fichier Excel
        df = pd.read_excel(file_path)
        
        print("Fichier lu avec succès")
        print(f"Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        print(f"Colonnes disponibles: {list(df.columns)}")
        
        # Analyse des colonnes
        analysis = {
            "file_info": {
                "path": file_path,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns)
            },
            "data_quality": {},
            "sample_data": {},
            "geocoding_needed": [],
            "issues": []
        }
        
        # Analyser chaque colonne
        for col in df.columns:
            col_data = df[col]
            
            # Statistiques de base
            non_null_count = col_data.notna().sum()
            null_count = col_data.isna().sum()
            unique_count = col_data.nunique()
            
            analysis["data_quality"][col] = {
                "non_null_count": int(non_null_count),
                "null_count": int(null_count),
                "unique_count": int(unique_count),
                "data_type": str(col_data.dtype),
                "sample_values": col_data.dropna().head(5).tolist()
            }
            
            # Identifier les colonnes qui pourraient contenir des adresses
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['adresse', 'address', 'lieu', 'location', 'depart', 'arrivee', 'destination', 'origine']):
                analysis["geocoding_needed"].append(col)
        
        # Afficher les premières lignes
        print("\nPremieres lignes du fichier:")
        print(df.head().to_string())
        
        # Analyser les données manquantes
        print("\nAnalyse des données manquantes:")
        missing_data = df.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                print(f"  - {col}: {missing_count} valeurs manquantes ({percentage:.1f}%)")
                analysis["issues"].append(f"Colonne '{col}' a {missing_count} valeurs manquantes ({percentage:.1f}%)")
        
        # Identifier les colonnes critiques pour le dispatch
        critical_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'heure', 'time', 'client', 'customer', 'adresse', 'address', 'lieu', 'location']):
                critical_columns.append(col)
        
        analysis["critical_columns"] = critical_columns
        
        print(f"\nColonnes critiques identifiees: {critical_columns}")
        
        return analysis
        
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")
        return {"error": str(e)}

def test_geocoding_api(address: str) -> Optional[Dict]:
    """Teste l'API Google Geocoding avec une adresse"""
    try:
        # Note: Vous devrez ajouter votre clé API Google
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            print("Cle API Google Maps non trouvee. Ajoutez GOOGLE_MAPS_API_KEY dans vos variables d'environnement.")
            return None
        
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK' and data['results']:
            result = data['results'][0]
            return {
                "address": result['formatted_address'],
                "lat": result['geometry']['location']['lat'],
                "lng": result['geometry']['location']['lng'],
                "status": "OK"
            }
        else:
            return {
                "status": data['status'],
                "error": data.get('error_message', 'Unknown error')
            }
            
    except Exception as e:
        return {"error": str(e)}

def main():
    file_path = r"C:\Users\jasiq\atmr\backend\transport_janvier.xlsx"
    
    print("Debut de l'analyse du fichier Excel")
    print("=" * 50)
    
    # Analyser le fichier
    analysis = analyze_excel_file(file_path)
    
    if "error" in analysis:
        print(f"Echec de l'analyse: {analysis['error']}")
        return
    
    print("\nRESUME DE L'ANALYSE")
    print("=" * 50)
    
    # Afficher le résumé
    print(f"Fichier: {analysis['file_info']['path']}")
    print(f"Dimensions: {analysis['file_info']['rows']} lignes x {analysis['file_info']['columns']} colonnes")
    print(f"Colonnes: {', '.join(analysis['file_info']['column_names'])}")
    
    if analysis['critical_columns']:
        print(f"Colonnes critiques: {', '.join(analysis['critical_columns'])}")
    
    if analysis['geocoding_needed']:
        print(f"Colonnes necessitant geocodage: {', '.join(analysis['geocoding_needed'])}")
    
    if analysis['issues']:
        print("Problemes detectes:")
        for issue in analysis['issues']:
            print(f"  - {issue}")
    
    # Test de géocodage si des adresses sont trouvées
    if analysis['geocoding_needed']:
        print("\nTest de geocodage...")
        test_address = "Zurich, Switzerland"  # Adresse de test
        geocoding_result = test_geocoding_api(test_address)
        
        if geocoding_result:
            if geocoding_result.get('status') == 'OK':
                print("Geocodage fonctionnel")
                print(f"   Adresse test: {geocoding_result['address']}")
                print(f"   Coordonnees: {geocoding_result['lat']}, {geocoding_result['lng']}")
            else:
                print(f"Probleme de geocodage: {geocoding_result.get('error', 'Unknown error')}")
        else:
            print("Geocodage non teste (cle API manquante)")
    
    # Sauvegarder l'analyse
    output_file = "excel_analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\nRapport detaille sauvegarde dans: {output_file}")
    
    print("\nAnalyse terminee!")

if __name__ == "__main__":
    main()
