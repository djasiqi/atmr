#!/usr/bin/env python3
"""Script de déploiement de l'Étape 9 - Hardening Docker/Prod.

Orchestre le déploiement complet du hardening Docker
avec validation et tests de production.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class DockerHardeningDeployer:
    """Déployeur pour le hardening Docker."""

    def __init__(self):
        """Initialise le déployeur."""
        self.backend_dir = Path("backend")
        self.deployment_log = []
        self.results = {}

    def log(self, ____________________________________________________________________________________________________message: str, level: str = "INFO") -> None:
        """Ajoute un message au log de déploiement."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)

    def run_command(self, ____________________________________________________________________________________________________command: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Exécute une commande et retourne le résultat."""
        self.log(f"Exécution: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=False, capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.backend_dir
            )
            
            success = result.returncode == 0
            self.log(f"Commande {'réussie' if success else 'échouée'}: {result.returncode}")
            
            if result.stdout:
                self.log(f"STDOUT: {result.stdout[:500]}...")
            if result.stderr:
                self.log(f"STDERR: {result.stderr[:500]}...", "WARNING")
            
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            self.log(f"Timeout après {timeout}s", "ERROR")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            self.log(f"Erreur: {e}", "ERROR")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def step1_validate_files(self) -> bool:
        """Étape 1: Validation des fichiers Docker."""
        self.log("🔍 Étape 1: Validation des fichiers Docker")
        
        required_files = [
            "Dockerfile.production",
            "docker-entrypoint.sh",
            "scripts/warmup_models.py",
            "scripts/docker_smoke_tests.py",
            "scripts/build-docker.sh"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.backend_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log(f"❌ Fichiers manquants: {missing_files}", "ERROR")
            return False
        
        self.log("✅ Tous les fichiers Docker présents")
        return True

    def step2_run_validation(self) -> bool:
        """Étape 2: Exécution de la validation complète."""
        self.log("🧪 Étape 2: Exécution de la validation complète")
        
        validation_script = self.backend_dir / "scripts" / "validate_step9_docker_hardening.py"
        
        if not validation_script.exists():
            self.log("❌ Script de validation non trouvé", "ERROR")
            return False
        
        result = self.run_command(["python", str(validation_script)])
        
        if result["success"]:
            self.log("✅ Validation complète réussie")
            return True
        self.log("❌ Validation échouée", "ERROR")
        return False

    def step3_build_docker_image(self) -> bool:
        """Étape 3: Build de l'image Docker."""
        self.log("🔨 Étape 3: Build de l'image Docker")
        
        build_script = self.backend_dir / "scripts" / "build-docker.sh"
        
        if not build_script.exists():
            self.log("❌ Script de build non trouvé", "ERROR")
            return False
        
        # Rendre le script exécutable
        os.chmod(build_script, 0o755)
        
        # Exécuter le build avec les options de test
        result = self.run_command([
            "bash", str(build_script),
            "test-build",
            "1.00",
            "--no-push"
        ], timeout=0.600)  # 10 minutes pour le build
        
        if result["success"]:
            self.log("✅ Build Docker réussi")
            return True
        self.log("❌ Build Docker échoué", "ERROR")
        return False

    def step4_run_smoke_tests(self) -> bool:
        """Étape 4: Exécution des tests de smoke."""
        self.log("🧪 Étape 4: Exécution des tests de smoke")
        
        smoke_tests_script = self.backend_dir / "scripts" / "docker_smoke_tests.py"
        
        if not smoke_tests_script.exists():
            self.log("❌ Script de tests de smoke non trouvé", "ERROR")
            return False
        
        result = self.run_command([
            "python", str(smoke_tests_script),
            "--image", "atmr-backend",
            "--tag", "test-build"
        ], timeout=0.300)  # 5 minutes pour les tests
        
        if result["success"]:
            self.log("✅ Tests de smoke réussis")
            return True
        self.log("❌ Tests de smoke échoués", "ERROR")
        return False

    def step5_test_warmup_script(self) -> bool:
        """Étape 5: Test du script de warmup."""
        self.log("🔥 Étape 5: Test du script de warmup")
        
        warmup_script = self.backend_dir / "scripts" / "warmup_models.py"
        
        if not warmup_script.exists():
            self.log("❌ Script de warmup non trouvé", "ERROR")
            return False
        
        # Test du script de warmup avec --help
        result = self.run_command([
            "python", str(warmup_script), "--help"
        ])
        
        if result["success"]:
            self.log("✅ Script de warmup fonctionnel")
            return True
        self.log("❌ Script de warmup défaillant", "ERROR")
        return False

    def step6_validate_docker_compose(self) -> bool:
        """Étape 6: Validation du docker-compose."""
        self.log("🐳 Étape 6: Validation du docker-compose")
        
        compose_file = Path("docker-compose.production.yml")
        
        if not compose_file.exists():
            self.log("❌ docker-compose.production.yml non trouvé", "ERROR")
            return False
        
        # Validation de la syntaxe docker-compose
        result = self.run_command([
            "docker-compose", "-f", str(compose_file), "config"
        ])
        
        if result["success"]:
            self.log("✅ Docker Compose syntaxe valide")
            return True
        self.log("❌ Docker Compose syntaxe invalide", "ERROR")
        return False

    def step7_generate_deployment_report(self) -> bool:
        """Étape 7: Génération du rapport de déploiement."""
        self.log("📋 Étape 7: Génération du rapport de déploiement")
        
        report = {
            "deployment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.00",
                "environment": "production"
            },
            "steps_completed": list(self.results.keys()),
            "success_rate": sum(1 for r in self.results.values() if r) / len(self.results) if self.results else 0,
            "results": self.results,
            "deployment_log": self.deployment_log
        }
        
        report_file = f"docker-hardening-deployment-report-{int(time.time())}.json"
        
        try:
            with Path(report_file, "w", encoding="utf-8").open() as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log(f"✅ Rapport de déploiement généré: {report_file}")
            return True
        except Exception as e:
            self.log(f"❌ Erreur lors de la génération du rapport: {e}", "ERROR")
            return False

    def deploy_all_steps(self) -> Dict[str, Any]:
        """Exécute toutes les étapes de déploiement."""
        self.log("🚀 Démarrage du déploiement Docker Hardening")
        self.log("=" * 60)
        
        steps = [
            ("Validation des fichiers", self.step1_validate_files),
            ("Validation complète", self.step2_run_validation),
            ("Build Docker", self.step3_build_docker_image),
            ("Tests de smoke", self.step4_run_smoke_tests),
            ("Test warmup", self.step5_test_warmup_script),
            ("Validation docker-compose", self.step6_validate_docker_compose),
            ("Génération rapport", self.step7_generate_deployment_report),
        ]
        
        successful_steps = 0
        total_steps = len(steps)
        
        for step_name, step_func in steps:
            self.log(f"\n🔧 Exécution: {step_name}")
            try:
                if step_func():
                    self.results[step_name] = True
                    successful_steps += 1
                    self.log(f"✅ {step_name} réussi")
                else:
                    self.results[step_name] = False
                    self.log(f"❌ {step_name} échoué", "ERROR")
            except Exception as e:
                self.log(f"❌ Erreur lors de {step_name}: {e}", "ERROR")
                self.results[step_name] = False
        
        # Résumé du déploiement
        self.log("\n" + "=" * 60)
        self.log("📊 RÉSUMÉ DU DÉPLOIEMENT DOCKER HARDENING")
        self.log("=" * 60)
        
        self.log(f"Étapes réussies: {successful_steps}/{total_steps}")
        
        for step_name, _ in steps:
            status = "✅ RÉUSSI" if self.results.get(step_name, False) else "❌ ÉCHOUÉ"
            self.log(f"  {step_name}: {status}")
        
        success_rate = successful_steps / total_steps
        
        if success_rate >= 0.8:
            self.log("\n🎉 DÉPLOIEMENT RÉUSSI!")
            self.log("✅ Le hardening Docker/Prod est déployé avec succès")
        elif success_rate >= 0.6:
            self.log("\n⚠️  DÉPLOIEMENT PARTIELLEMENT RÉUSSI")
            self.log("⚠️  Certaines étapes nécessitent une attention")
        else:
            self.log("\n❌ DÉPLOIEMENT ÉCHOUÉ")
            self.log("❌ Le hardening Docker/Prod nécessite des corrections")
        
        return {
            "successful_steps": successful_steps,
            "total_steps": total_steps,
            "success_rate": success_rate,
            "results": self.results,
            "deployment_log": self.deployment_log
        }


def main():
    """Fonction principale."""
    print("🚀 Déploiement de l'Étape 9 - Hardening Docker/Prod")
    
    deployer = DockerHardeningDeployer()
    
    try:
        results = deployer.deploy_all_steps()
        
        # Code de sortie basé sur le succès
        if results["success_rate"] >= 0.8:
            print("\n🎉 Déploiement terminé avec succès!")
            sys.exit(0)
        else:
            print("\n❌ Déploiement échoué")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Déploiement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception:
        print("\n❌ Erreur lors du déploiement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
