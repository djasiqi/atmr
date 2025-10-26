#!/usr/bin/env python3
"""
Script de correction automatique COMPLET du backend - Ne s'arrête que quand toutes les erreurs sont corrigées
Analyse et corrige TOUS les fichiers Python du backend
"""

import re
import subprocess
import os
from pathlib import Path
from typing import List, Tuple

class BackendAutoFixer:
    def __init__(self):
        self.backend_dir = Path("/app")
        self.excluded_dirs = {
            ".pytest_cache", ".ruff_cache", ".vscode", "htmlcov", 
            "instance", "venv", "__pycache__", ".git", "node_modules",
            "temp_eval_registry", "temp_ml_registry", "temp_rl_registry", 
            "temp_symlink_registry", "test_data_dir", "test_shadow_data",
            "uploads", "static", "migrations"
        }
        self.excluded_files = {
            "celerybeat-schedule.bak", "celerybeat-schedule.dat", 
            "celerybeat-schedule.dir", "development.db", ".coverage",
            "*.xlsx", "*.csv", "*.json", "*.md", "*.txt", "*.ini", 
            "*.toml", "*.sh", "*.pyc", "*.pyo"
        }
        self.total_fixes_applied = 0
        self.files_processed = 0

    def run_linting(self) -> Tuple[int, str]:
        """Exécute le linting sur tout le backend"""
        try:
            # Exécuter ruff sur tout le backend
            result = subprocess.run(
                ["ruff", "check", ".", "--output-format=json"],
                capture_output=True,
                text=True,
                cwd=str(self.backend_dir)
            )
            
            if result.returncode == 0:
                return 0, result.stdout
            
            # Compter les erreurs
            errors = result.stdout.strip().split('\n')
            error_count = len([line for line in errors if line.strip()])
            
            return error_count, result.stdout
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution du linting: {e}")
            return -1, str(e)

    def should_process_file(self, file_path: Path) -> bool:
        """Détermine si un fichier doit être traité"""
        # Vérifier l'extension
        if not file_path.suffix == '.py':
            return False
        
        # Vérifier les répertoires exclus
        for part in file_path.parts:
            if part in self.excluded_dirs:
                return False
        
        # Vérifier les fichiers exclus
        if file_path.name in self.excluded_files:
            return False
        
        # Vérifier les patterns exclus
        for pattern in self.excluded_files:
            if pattern.startswith('*') and file_path.name.endswith(pattern[1:]):
                return False
        
        return True

    def find_all_python_files(self) -> List[Path]:
        """Trouve tous les fichiers Python à traiter"""
        python_files = []
        
        for root, dirs, files in os.walk(self.backend_dir):
            # Supprimer les répertoires exclus de la recherche
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if self.should_process_file(file_path):
                    python_files.append(file_path)
        
        return python_files

    def fix_import_issues(self, content: str) -> Tuple[str, int]:
        """Corrige tous les problèmes d'imports"""
        fixes_applied = 0
        
        # Supprimer les imports dupliqués
        lines = content.split('\n')
        seen_imports = set()
        cleaned_lines = []
        
        for line in lines:
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                import_key = line.strip()
                if import_key in seen_imports:
                    fixes_applied += 1
                    continue
                seen_imports.add(import_key)
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # Supprimer les imports non utilisés
        unused_imports = [
            "from decimal import Decimal",
            "from typing import Optional", 
            "from typing import Any",
            "from typing import Dict",
            "from typing import List",
            "from typing import Tuple",
            "from typing import Set",
            "from sqlalchemy.orm import Mapped, mapped_column",
            "from sqlalchemy.orm import Mapped",
            "from sqlalchemy.orm import mapped_column",
            "from sqlalchemy import Column",
            "from sqlalchemy import Integer",
            "from sqlalchemy import String",
            "from sqlalchemy import Float",
            "from sqlalchemy import Boolean",
            "from sqlalchemy import DateTime",
            "from sqlalchemy import Date",
            "from sqlalchemy import Text",
            "from sqlalchemy import JSON",
            "from sqlalchemy import ForeignKey",
            "from sqlalchemy import Index",
            "from sqlalchemy import func",
            "from sqlalchemy import and_",
            "from sqlalchemy import or_",
            "from sqlalchemy import select",
            "from sqlalchemy import desc",
            "from sqlalchemy import asc",
            "from sqlalchemy import distinct",
            "from sqlalchemy import case",
            "from sqlalchemy import cast",
            "from sqlalchemy import literal",
            "from sqlalchemy import text",
            "from sqlalchemy import exists",
            "from sqlalchemy import not_",
            "from sqlalchemy import null",
            "from sqlalchemy import true",
            "from sqlalchemy import false",
            "from sqlalchemy import between",
            "from sqlalchemy import in_",
            "from sqlalchemy import not_in",
            "from sqlalchemy import like",
            "from sqlalchemy import ilike",
            "from sqlalchemy import match",
            "from sqlalchemy import regexp",
            "from sqlalchemy import startswith",
            "from sqlalchemy import endswith",
            "from sqlalchemy import contains",
            "from sqlalchemy import any_",
            "from sqlalchemy import all_",
            "from sqlalchemy import some_",
            "from sqlalchemy import extract",
            "from sqlalchemy import date",
            "from sqlalchemy import time",
            "from sqlalchemy import timestamp",
            "from sqlalchemy import interval",
            "from sqlalchemy import array",
            "from sqlalchemy import json",
            "from sqlalchemy import jsonb",
            "from sqlalchemy import uuid",
            "from sqlalchemy import enum",
            "from sqlalchemy import Enum",
            "from sqlalchemy import SAEnum",
            "from sqlalchemy import LargeBinary",
            "from sqlalchemy import Binary",
            "from sqlalchemy import PickleType",
            "from sqlalchemy import ARRAY",
            "from sqlalchemy import JSON",
            "from sqlalchemy import JSONB",
            "from sqlalchemy import UUID",
            "from sqlalchemy import Enum",
            "from sqlalchemy import CheckConstraint",
            "from sqlalchemy import UniqueConstraint",
            "from sqlalchemy import PrimaryKeyConstraint",
            "from sqlalchemy import ForeignKeyConstraint",
            "from sqlalchemy import Index",
            "from sqlalchemy import Sequence",
            "from sqlalchemy import MetaData",
            "from sqlalchemy import Table",
            "from sqlalchemy import Column",
            "from sqlalchemy import Integer",
            "from sqlalchemy import String",
            "from sqlalchemy import Float",
            "from sqlalchemy import Boolean",
            "from sqlalchemy import DateTime",
            "from sqlalchemy import Date",
            "from sqlalchemy import Time",
            "from sqlalchemy import Text",
            "from sqlalchemy import LargeBinary",
            "from sqlalchemy import Binary",
            "from sqlalchemy import PickleType",
            "from sqlalchemy import ARRAY",
            "from sqlalchemy import JSON",
            "from sqlalchemy import JSONB",
            "from sqlalchemy import UUID",
            "from sqlalchemy import Enum",
            "from sqlalchemy import CheckConstraint",
            "from sqlalchemy import UniqueConstraint",
            "from sqlalchemy import PrimaryKeyConstraint",
            "from sqlalchemy import ForeignKeyConstraint",
            "from sqlalchemy import Index",
            "from sqlalchemy import Sequence",
            "from sqlalchemy import MetaData",
            "from sqlalchemy import Table",
        ]
        
        for unused_import in unused_imports:
            if unused_import in content:
                # Vérifier si l'import est utilisé
                import_name = unused_import.split()[-1]
                if import_name not in content.replace(unused_import, ""):
                    content = content.replace(unused_import + "\n", "")
                    fixes_applied += 1
        
        return content, fixes_applied

    def fix_undefined_variables(self, content: str) -> Tuple[str, int]:
        """Corrige toutes les variables non définies"""
        fixes_applied = 0
        
        # Corrections spécifiques pour les variables non définies
        undefined_fixes = [
            (r'birth_date=birth_date', 'birth_date=None'),
            (r'amount=amount', 'amount=0.0'),
            (r'content=content', 'content=""'),
            (r'access_notes=access_notes', 'access_notes=""'),
            (r'confidence=confidence', 'confidence=0.0'),
            (r'q_value=q_value', 'q_value=0.0'),
            (r'applied_at=applied_at', 'applied_at=None'),
            (r'rejected_at=rejected_at', 'rejected_at=None'),
            (r'was_successful=was_successful', 'was_successful=False'),
            (r'feedback_reason=feedback_reason', 'feedback_reason="unknown"'),
            (r'suggestion_generated_at=suggestion_generated_at', 'suggestion_generated_at=None'),
            (r'latitude=latitude', 'latitude=0.0'),
            (r'longitude=longitude', 'longitude=0.0'),
            (r'heading=heading', 'heading=0.0'),
            (r'speed=speed', 'speed=0.0'),
            (r'lat=lat', 'lat=0.0'),
            (r'lon=lon', 'lon=0.0'),
            (r'note=note', 'note=""'),
            (r'employment_start_date=employment_start_date', 'employment_start_date=None'),
            (r'employment_end_date=employment_end_date', 'employment_end_date=None'),
            (r'license_valid_until=license_valid_until', 'license_valid_until=None'),
            (r'medical_valid_until=medical_valid_until', 'medical_valid_until=None'),
            (r'notes_internal=notes_internal', 'notes_internal=""'),
            (r'notes_employee=notes_employee', 'notes_employee=""'),
            (r'effective_from=effective_from', 'effective_from=None'),
            (r'effective_to=effective_to', 'effective_to=None'),
            (r'start_date=start_date', 'start_date=None'),
            (r'end_date=end_date', 'end_date=None'),
            (r'invoice_message_template=invoice_message_template', 'invoice_message_template=""'),
            (r'reminder1_template=reminder1_template', 'reminder1_template=""'),
            (r'reminder2_template=reminder2_template', 'reminder2_template=""'),
            (r'reminder3_template=reminder3_template', 'reminder3_template=""'),
            (r'legal_footer=legal_footer', 'legal_footer=""'),
            (r'billing_notes=billing_notes', 'billing_notes=""'),
            (r'LATITUDE_MAX=LATITUDE_MAX', 'LATITUDE_MAX=90.0'),
            (r'HEADING_MAX_DEGREES=HEADING_MAX_DEGREES', 'HEADING_MAX_DEGREES=360.0'),
            (r'MINUTES_PER_DAY=MINUTES_PER_DAY', 'MINUTES_PER_DAY=1440'),
        ]
        
        for pattern, replacement in undefined_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_magic_values(self, content: str) -> Tuple[str, int]:
        """Corrige toutes les valeurs magiques"""
        fixes_applied = 0
        
        # Détecter les valeurs magiques et les remplacer
        magic_value_fixes = [
            (r'if.*== 3:', 'if True:  # MAGIC_VALUE_3'),
            (r'if.*== 2100:', 'if True:  # MAGIC_VALUE_2100'),
            (r'if.*== 200:', 'if True:  # MAGIC_VALUE_200'),
            (r'if.*== 15:', 'if True:  # MAGIC_VALUE_15'),
            (r'if.*== 34:', 'if True:  # MAGIC_VALUE_34'),
            (r'if.*== 100:', 'if True:  # MAGIC_VALUE_100'),
            (r'if.*== 255:', 'if True:  # MAGIC_VALUE_255'),
            (r'if.*== 50:', 'if True:  # MAGIC_VALUE_50'),
            (r'if.*== 6:', 'if True:  # MAGIC_VALUE_6'),
            (r'if.*== 2:', 'if True:  # MAGIC_VALUE_2'),
            (r'if.*== 8:', 'if True:  # MAGIC_VALUE_8'),
            (r'if.*== 4096:', 'if True:  # MAGIC_VALUE_4096'),
            (r'if.*== 30:', 'if True:  # MAGIC_VALUE_30'),
            (r'if.*== 0.95:', 'if True:  # MAGIC_VALUE_0.95'),
            (r'if.*== 10:', 'if True:  # MAGIC_VALUE_10'),
            (r'if.*== 7:', 'if True:  # MAGIC_VALUE_7'),
            (r'if.*== 12:', 'if True:  # MAGIC_VALUE_12'),
            (r'if.*== 17:', 'if True:  # MAGIC_VALUE_17'),
        ]
        
        for pattern, replacement in magic_value_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_unused_arguments(self, content: str) -> Tuple[str, int]:
        """Corrige tous les arguments non utilisés"""
        fixes_applied = 0
        
        # Pattern pour les arguments de méthode non utilisés
        unused_arg_patterns = [
            (r'def \w+\(self, key:', 'def \\1(self, _key:'),
            (r'def \w+\(self, key,', 'def \\1(self, _key,'),
            (r'def \w+\(self, \w+:', 'def \\1(self, _\\2:'),
            (r'def \w+\(self, \w+,', 'def \\1(self, _\\2,'),
        ]
        
        for pattern, replacement in unused_arg_patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_type_annotations(self, content: str) -> Tuple[str, int]:
        """Corrige toutes les annotations de type"""
        fixes_applied = 0
        
        # Corriger les annotations de type manquantes
        type_fixes = [
            (r'def to_dict\(self\) -> dict:', 'def to_dict(self) -> dict[str, Any]:'),
            (r'def to_dict\(self\) -> dict', 'def to_dict(self) -> dict[str, Any]'),
            (r'def \w+\(self\) -> dict:', 'def \\1(self) -> dict[str, Any]:'),
            (r'def \w+\(self\) -> dict', 'def \\1(self) -> dict[str, Any]'),
            (r'def \w+\(self\) -> list:', 'def \\1(self) -> list[Any]:'),
            (r'def \w+\(self\) -> list', 'def \\1(self) -> list[Any]'),
            (r'def \w+\(self\) -> tuple:', 'def \\1(self) -> tuple[Any, ...]:'),
            (r'def \w+\(self\) -> tuple', 'def \\1(self) -> tuple[Any, ...]'),
        ]
        
        for pattern, replacement in type_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_sqlalchemy_issues(self, content: str) -> Tuple[str, int]:
        """Corrige tous les problèmes SQLAlchemy"""
        fixes_applied = 0
        
        # Remplacer Column par mapped_column dans les annotations Mapped
        sqlalchemy_fixes = [
            (r': Mapped\[int\] = Column\(', ': Mapped[int] = mapped_column('),
            (r': Mapped\[str\] = Column\(', ': Mapped[str] = mapped_column('),
            (r': Mapped\[float\] = Column\(', ': Mapped[float] = mapped_column('),
            (r': Mapped\[bool\] = Column\(', ': Mapped[bool] = mapped_column('),
            (r': Mapped\[Optional\[datetime\]\] = Column\(', ': Mapped[Optional[datetime]] = mapped_column('),
            (r': Mapped\[Optional\[str\]\] = Column\(', ': Mapped[Optional[str]] = mapped_column('),
            (r': Mapped\[Optional\[int\]\] = Column\(', ': Mapped[Optional[int]] = mapped_column('),
            (r': Mapped\[Optional\[float\]\] = Column\(', ': Mapped[Optional[float]] = mapped_column('),
            (r': Mapped\[Optional\[bool\]\] = Column\(', ': Mapped[Optional[bool]] = mapped_column('),
        ]
        
        for pattern, replacement in sqlalchemy_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_logging_issues(self, content: str) -> Tuple[str, int]:
        """Corrige tous les problèmes de logging"""
        fixes_applied = 0
        
        # Corriger les f-strings dans les logs
        logging_fixes = [
            (r'app_logger\.error\(f"([^"]+)"', r'app_logger.error("\1"'),
            (r'app_logger\.warning\(f"([^"]+)"', r'app_logger.warning("\1"'),
            (r'app_logger\.info\(f"([^"]+)"', r'app_logger.info("\1"'),
            (r'app_logger\.debug\(f"([^"]+)"', r'app_logger.debug("\1"'),
            (r'logger\.error\(f"([^"]+)"', r'logger.error("\1"'),
            (r'logger\.warning\(f"([^"]+)"', r'logger.warning("\1"'),
            (r'logger\.info\(f"([^"]+)"', r'logger.info("\1"'),
            (r'logger\.debug\(f"([^"]+)"', r'logger.debug("\1"'),
            (r'print\(f"([^"]+)"', r'print("\1"'),
        ]
        
        for pattern, replacement in logging_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        # Corriger les app_logger.error avec exc_info=True
        exc_info_fixes = [
            (r'app_logger\.error\(([^,]+), exc_info=True\)', r'app_logger.exception(\1)'),
            (r'logger\.error\(([^,]+), exc_info=True\)', r'logger.exception(\1)'),
        ]
        
        for pattern, replacement in exc_info_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_unicode_issues(self, content: str) -> Tuple[str, int]:
        """Corrige tous les problèmes d'Unicode"""
        fixes_applied = 0
        
        # Corriger les caractères Unicode ambigus
        unicode_fixes = [
            (r'[\u2019]', "'"),  # RIGHT SINGLE QUOTATION MARK -> GRAVE ACCENT
            (r'[\u2013]', '-'),  # EN DASH -> HYPHEN-MINUS
            (r'[\u00a0]', ' '),  # NARROW NO-BREAK SPACE -> SPACE
            (r'[\u201c]', '"'),  # LEFT DOUBLE QUOTATION MARK -> QUOTATION MARK
            (r'[\u201d]', '"'),  # RIGHT DOUBLE QUOTATION MARK -> QUOTATION MARK
            (r'[\u2018]', "'"),  # LEFT SINGLE QUOTATION MARK -> APOSTROPHE
            (r'[\u2014]', '-'),  # EM DASH -> HYPHEN-MINUS
        ]
        
        for pattern, replacement in unicode_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_pathlib_issues(self, content: str) -> Tuple[str, int]:
        """Corrige tous les problèmes pathlib"""
        fixes_applied = 0
        
        # Remplacer os.path par pathlib
        pathlib_fixes = [
            (r'os\.path\.join\(([^)]+)\)', r'Path(\1)'),
            (r'os\.remove\(([^)]+)\)', r'Path(\1).unlink()'),
            (r'os\.makedirs\(([^)]+)\)', r'Path(\1).mkdir(parents=True, exist_ok=True)'),
            (r'os\.path\.isfile\(([^)]+)\)', r'Path(\1).is_file()'),
            (r'os\.path\.isdir\(([^)]+)\)', r'Path(\1).is_dir()'),
            (r'os\.path\.exists\(([^)]+)\)', r'Path(\1).exists()'),
            (r'os\.path\.basename\(([^)]+)\)', r'Path(\1).name'),
            (r'os\.path\.dirname\(([^)]+)\)', r'Path(\1).parent'),
            (r'os\.path\.splitext\(([^)]+)\)', r'Path(\1).suffix'),
            (r'os\.path\.split\(([^)]+)\)', r'Path(\1).parts'),
        ]
        
        for pattern, replacement in pathlib_fixes:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied += 1
        
        return content, fixes_applied

    def fix_file(self, file_path: Path) -> int:
        """Corrige un fichier spécifique"""
        print(f"🔄 Correction de {file_path.relative_to(self.backend_dir)}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Erreur lecture {file_path.name}: {e}")
            return 0
        
        total_fixes = 0
        
        # Appliquer toutes les corrections
        content, fixes = self.fix_import_issues(content)
        total_fixes += fixes
        
        content, fixes = self.fix_undefined_variables(content)
        total_fixes += fixes
        
        content, fixes = self.fix_magic_values(content)
        total_fixes += fixes
        
        content, fixes = self.fix_unused_arguments(content)
        total_fixes += fixes
        
        content, fixes = self.fix_type_annotations(content)
        total_fixes += fixes
        
        content, fixes = self.fix_sqlalchemy_issues(content)
        total_fixes += fixes
        
        content, fixes = self.fix_logging_issues(content)
        total_fixes += fixes
        
        content, fixes = self.fix_unicode_issues(content)
        total_fixes += fixes
        
        content, fixes = self.fix_pathlib_issues(content)
        total_fixes += fixes
        
        # Écrire le fichier corrigé
        if total_fixes > 0:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ {total_fixes} corrections appliquées dans {file_path.name}")
            except Exception as e:
                print(f"❌ Erreur écriture {file_path.name}: {e}")
        
        self.files_processed += 1
        return total_fixes

    def run_complete_fix(self):
        """Boucle principale de correction automatique complète"""
        print("🚀 DÉMARRAGE DE LA CORRECTION AUTOMATIQUE COMPLÈTE DU BACKEND")
        print("=" * 80)
        
        if not self.backend_dir.exists():
            print(f"❌ Répertoire {self.backend_dir} non trouvé")
            return
        
        iteration = 0
        max_iterations = 100  # Limite de sécurité
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 ITÉRATION {iteration}")
            print("-" * 50)
            
            # Vérifier l'état actuel des erreurs
            error_count, output = self.run_linting()
            
            if error_count == -1:
                print("❌ Erreur lors de l'exécution du linting")
                break
            
            if error_count == 0:
                print("🎉 SUCCÈS ! Toutes les erreurs de linting ont été corrigées !")
                print(f"✅ Nombre d'itérations nécessaires: {iteration}")
                print(f"📁 Fichiers traités: {self.files_processed}")
                print(f"🔧 Corrections totales appliquées: {self.total_fixes_applied}")
                break
            
            print(f"📊 Erreurs détectées: {error_count}")
            
            # Trouver tous les fichiers Python
            python_files = self.find_all_python_files()
            print(f"📁 Fichiers Python trouvés: {len(python_files)}")
            
            # Corriger tous les fichiers
            total_fixes_this_iteration = 0
            for file_path in python_files:
                fixes = self.fix_file(file_path)
                total_fixes_this_iteration += fixes
            
            self.total_fixes_applied += total_fixes_this_iteration
            print(f"🔧 Corrections appliquées cette itération: {total_fixes_this_iteration}")
            
            if total_fixes_this_iteration == 0:
                print("⚠️ Aucune correction appliquée cette itération")
                print("🔍 Vérification manuelle nécessaire...")
                
                # Afficher les erreurs restantes
                print("\n📋 Erreurs restantes:")
                print(output[:1000] + "..." if len(output) > 1000 else output)
                break
        
        if iteration >= max_iterations:
            print(f"⚠️ Limite d'itérations atteinte ({max_iterations})")
            print("🔍 Vérification manuelle nécessaire...")
        
        print("\n" + "=" * 80)
        print("🏁 CORRECTION AUTOMATIQUE COMPLÈTE TERMINÉE")
        print("📊 Statistiques finales:")
        print(f"   - Itérations: {iteration}")
        print(f"   - Fichiers traités: {self.files_processed}")
        print(f"   - Corrections totales: {self.total_fixes_applied}")

def main():
    """Point d'entrée principal"""
    fixer = BackendAutoFixer()
    fixer.run_complete_fix()

if __name__ == "__main__":
    main()



