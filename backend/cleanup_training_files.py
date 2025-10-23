#!/usr/bin/env python3
"""
Script de nettoyage des fichiers d'entraînement inutilisés
Garde seulement les modèles importants et supprime les fichiers temporaires
"""

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

def cleanup_training_files():
    """Nettoie les fichiers d'entraînement inutilisés"""
    
    print("=================================================================================")
    print("🧹 NETTOYAGE DES FICHIERS D'ENTRAÎNEMENT")
    print("=================================================================================")
    
    # Répertoires à nettoyer
    cleanup_dirs = {
        'data/rl/models': 'Modèles RL',
        'data/ml': 'Modèles ML',
        'backend': 'Scripts temporaires',
        '.': 'Fichiers racine'
    }
    
    total_freed = 0
    files_kept = []
    files_deleted = []
    
    # 1. Nettoyage des modèles RL
    print("\n📂 NETTOYAGE MODÈLES RL (data/rl/models/)")
    print("-" * 50)
    
    models_dir = Path('data/rl/models')
    if models_dir.exists():
        # Modèles à garder (les plus importants)
        keep_models = {
            'dispatch_optimized_v4_corrected.pth',  # Modèle actuel
            'dispatch_optimized_v3_cleaned.pth',   # Modèle précédent
            'dispatch_optimized_v2.pth',           # Modèle v2
            'dispatch_optimized_v1.pth',           # Modèle v1
            'dqn_best.pth',                        # Meilleur modèle
            'dqn_final.pth'                        # Modèle final
        }
        
        for model_file in models_dir.glob('*.pth'):
            if model_file.name in keep_models:
                files_kept.append(f"✅ Gardé: {model_file.name}")
            else:
                size = model_file.stat().st_size
                total_freed += size
                files_deleted.append(f"🗑️  Supprimé: {model_file.name} ({size/1024/1024:.1f} MB)")
                model_file.unlink()
    
    # 2. Nettoyage des scripts temporaires d'analyse
    print("\n📂 NETTOYAGE SCRIPTS TEMPORAIRES")
    print("-" * 50)
    
    temp_scripts = [
        'analyze_excel_simple.py',
        'analyze_excel_corrected.py', 
        'analyze_all_sheets.py',
        'analyze_all_sheets_improved.py',
        'analyze_all_sheets_corrected.py',
        'analyze_all_sheets_final.py',
        'prepare_training_data.py',
        'prepare_complete_training.py',
        'prepare_final_training.py',
        'generate_training_document.py',
        'generate_training_corrected.py',
        'generate_training_cleaned_final.py',
        'clean_data_final.py',
        'fix_drivers_categories.py',
        'create_corrected_rl_data.py',
        'train_rl_corrected.py',
        'monitor_corrected_training.py'
    ]
    
    for script in temp_scripts:
        script_path = Path(script)
        if script_path.exists():
            size = script_path.stat().st_size
            total_freed += size
            files_deleted.append(f"🗑️  Supprimé: {script} ({size/1024:.1f} KB)")
            script_path.unlink()
    
    # 3. Nettoyage des fichiers de données temporaires
    print("\n📂 NETTOYAGE DONNÉES TEMPORAIRES")
    print("-" * 50)
    
    temp_data_files = [
        'excel_analysis_corrected.json',
        'all_sheets_analysis.json',
        'all_sheets_analysis_improved.json', 
        'all_sheets_analysis_corrected.json',
        'all_sheets_analysis_final.json',
        'excel_data_cleaned_final.json',
        'driver_categories_corrected.json',
        'historical_dispatches_cleaned.json',
        'historical_dispatches_corrected.json',
        'training_data_complete.json',
        'training_data_complete_final.json',
        'training_data_corrected_final.json',
        'training_data_final_complete.json'
    ]
    
    for data_file in temp_data_files:
        data_path = Path(data_file)
        if data_path.exists():
            size = data_path.stat().st_size
            total_freed += size
            files_deleted.append(f"🗑️  Supprimé: {data_file} ({size/1024:.1f} KB)")
            data_path.unlink()
    
    # 4. Nettoyage des fichiers Excel/CSV temporaires
    print("\n📂 NETTOYAGE FICHIERS EXCEL/CSV TEMPORAIRES")
    print("-" * 50)
    
    temp_docs = [
        'training_document_complete_final.xlsx',
        'training_document_complete_final.csv',
        'training_document_corrected_final.xlsx', 
        'training_document_corrected_final.csv',
        'training_document_final_complete.xlsx',
        'training_document_final_complete.csv'
    ]
    
    for doc in temp_docs:
        doc_path = Path(doc)
        if doc_path.exists():
            size = doc_path.stat().st_size
            total_freed += size
            files_deleted.append(f"🗑️  Supprimé: {doc} ({size/1024:.1f} KB)")
            doc_path.unlink()
    
    # 5. Nettoyage des logs d'entraînement anciens
    print("\n📂 NETTOYAGE LOGS ANCIENS")
    print("-" * 50)
    
    logs_dir = Path('data/rl')
    if logs_dir.exists():
        for log_file in logs_dir.glob('training_*.log'):
            # Garder seulement le dernier log
            if 'output.log' not in log_file.name:
                size = log_file.stat().st_size
                total_freed += size
                files_deleted.append(f"🗑️  Supprimé: {log_file.name} ({size/1024:.1f} KB)")
                log_file.unlink()
    
    # Résumé du nettoyage
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DU NETTOYAGE")
    print("="*80)
    
    print(f"\n✅ FICHIERS CONSERVÉS ({len(files_kept)}):")
    for kept in files_kept:
        print(f"   {kept}")
    
    print(f"\n🗑️  FICHIERS SUPPRIMÉS ({len(files_deleted)}):")
    for deleted in files_deleted:
        print(f"   {deleted}")
    
    print(f"\n💾 ESPACE LIBÉRÉ: {total_freed/1024/1024:.1f} MB")
    
    # Fichiers finaux conservés
    print(f"\n📁 FICHIERS FINAUX CONSERVÉS:")
    final_files = [
        'training_data_cleaned_final.json',
        'training_document_cleaned_final.xlsx',
        'training_document_cleaned_final.csv',
        'data/rl/models/dispatch_optimized_v4_corrected.pth',
        'data/rl/models/dispatch_optimized_v3_cleaned.pth',
        'data/rl/models/dispatch_optimized_v2.pth',
        'data/rl/models/dispatch_optimized_v1.pth',
        'data/rl/models/dqn_best.pth',
        'data/rl/models/dqn_final.pth'
    ]
    
    for final_file in final_files:
        if Path(final_file).exists():
            print(f"   ✅ {final_file}")
        else:
            print(f"   ❌ {final_file} (manquant)")
    
    print(f"\n🎉 NETTOYAGE TERMINÉ!")
    print(f"   - {len(files_kept)} fichiers conservés")
    print(f"   - {len(files_deleted)} fichiers supprimés") 
    print(f"   - {total_freed/1024/1024:.1f} MB libérés")
    
    # Sauvegarder le rapport
    report = {
        'cleanup_date': datetime.now(tz=UTC).isoformat(),
        'files_kept': files_kept,
        'files_deleted': files_deleted,
        'space_freed_mb': total_freed/1024/1024,
        'final_files': final_files
    }
    
    import json
    with open('cleanup_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Rapport sauvegardé: cleanup_report.json")

if __name__ == "__main__":
    cleanup_training_files()

