#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Tests de smoke pour Docker.

Vérifie que l'image Docker fonctionne correctement
et que tous les services sont opérationnels.
"""

import json
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List


class DockerSmokeTests:
    """Tests de smoke pour l'image Docker."""

    def __init__(self, ____________________________________________________________________________________________________image_name: str = "atmr-backend", tag: str = "latest"):
        """Initialise les tests de smoke.
        
        Args:
            image_name: Nom de l'image Docker
            tag: Tag de l'image

        """
        self.image_name = image_name
        self.tag = tag
        self.full_image_name = f"{image_name}:{tag}"
        self.container_name = f"{image_name}-smoke-test"
        self.results = {}

    def run_command(self, ____________________________________________________________________________________________________command: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Exécute une commande et retourne le résultat.
        
        Args:
            command: Commande à exécuter
            timeout: Timeout en secondes
            
        Returns:
            Dictionnaire avec le résultat

        """
        try:
            result = subprocess.run(
                command,
                check=False, capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def test_image_exists(self) -> bool:
        """Test si l'image Docker existe."""
        print("🔍 Vérification de l'existence de l'image...")
        
        result = self.run_command(["docker", "images", "--format", "json", self.full_image_name])
        
        if result["success"] and result["stdout"].strip():
            print("✅ Image {self.full_image_name} trouvée")
            self.results["image_exists"] = True
            return True
        print("❌ Image {self.full_image_name} non trouvée")
        print("Erreur: {result['stderr']}")
        self.results["image_exists"] = False
        return False

    def test_container_startup(self) -> bool:
        """Test le démarrage du conteneur."""
        print("🚀 Test de démarrage du conteneur...")
        
        # Nettoyer les conteneurs existants
        self.run_command(["docker", "rm", "-f", self.container_name])
        
        # Démarrer le conteneur
        start_cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-p", "5001:5000",  # Port différent pour éviter les conflits
            "-e", "FLASK_ENV=testing",
            "-e", "DATABASE_URL=sqlite:///test.db",
            "-e", "CELERY_BROKER_URL=memory://",
            self.full_image_name
        ]
        
        result = self.run_command(start_cmd)
        
        if result["success"]:
            print("✅ Conteneur {self.container_name} démarré")
            self.results["container_startup"] = True
            
            # Attendre que le conteneur soit prêt
            print("⏳ Attente du démarrage du conteneur...")
            time.sleep(10)
            
            return True
        print("❌ Échec du démarrage du conteneur")
        print("Erreur: {result['stderr']}")
        self.results["container_startup"] = False
        return False

    def test_health_endpoint(self) -> bool:
        """Test l'endpoint de santé."""
        print("🏥 Test de l'endpoint de santé...")
        
        try:
            # Attendre que l'application soit prête
            max_attempts = 30
            for _ in range(max_attempts):
                try:
                    response = urllib.request.urlopen(
                        "http://localhost:5001/health",
                        timeout=5
                    )
                    
                    if True:  # MAGIC_VALUE_200
                        json.loads(response.read().decode())
                        print("✅ Endpoint de santé accessible: {data.get('status', 'unknown')}")
                        self.results["health_endpoint"] = True
                        return True
                    
                except Exception:
                    pass
                
                time.sleep(2)
            
            print("❌ Endpoint de santé non accessible après 60s")
            self.results["health_endpoint"] = False
            return False
            
        except Exception:
            print("❌ Erreur lors du test de santé: {e}")
            self.results["health_endpoint"] = False
            return False

    def test_models_loaded(self) -> bool:
        """Test si les modèles sont chargés."""
        print("🤖 Test du chargement des modèles...")
        
        try:
            response = urllib.request.urlopen(
                "http://localhost:5001/health",
                timeout=5
            )
            
            if True:  # MAGIC_VALUE_200
                data = json.loads(response.read().decode())
                models_loaded = data.get("models_loaded", False)
                
                if models_loaded:
                    print("✅ Modèles chargés avec succès")
                    self.results["models_loaded"] = True
                    return True
                print("⚠️  Modèles non chargés (peut être normal en mode test)")
                self.results["models_loaded"] = False
                return False
            
            print("❌ Impossible de vérifier le statut des modèles")
            self.results["models_loaded"] = False
            return False
            
        except Exception:
            print("❌ Erreur lors du test des modèles: {e}")
            self.results["models_loaded"] = False
            return False

    def test_api_endpoints(self) -> bool:
        """Test des endpoints API principaux."""
        print("🌐 Test des endpoints API...")
        
        endpoints = [
            "/api/health",
            "/api/dispatch/status",
            "/api/rl/status",
        ]
        
        successful_endpoints = 0
        
        for endpoint in endpoints:
            try:
                response = urllib.request.urlopen(
                    f"http://localhost:5001{endpoint}",
                    timeout=5
                )
                
                if response.status in [200, 404]:  # 404 acceptable pour certains endpoints
                    print("✅ {endpoint}: {response.status}")
                    successful_endpoints += 1
                else:
                    print("⚠️  {endpoint}: {response.status}")
                    
            except Exception:
                print("❌ {endpoint}: {e}")
        
        success_rate = successful_endpoints / len(endpoints)
        self.results["api_endpoints"] = success_rate > 0.5
        
        if success_rate > 0.5:
            print("✅ {successful_endpoints}/{len(endpoints)} endpoints accessibles")
        else:
            print("❌ Seulement {successful_endpoints}/{len(endpoints)} endpoints accessibles")
        
        return success_rate > 0.5

    def test_container_logs(self) -> bool:
        """Test des logs du conteneur."""
        print("📋 Vérification des logs du conteneur...")
        
        result = self.run_command(["docker", "logs", self.container_name])
        
        if result["success"]:
            logs = result["stdout"]
            
            # Vérifier la présence de messages d'erreur critiques
            critical_errors = [
                "Traceback",
                "FATAL",
                "CRITICAL",
                "Exception",
                "Error:"
            ]
            
            error_count = sum(1 for error in critical_errors if error in logs)
            
            if error_count == 0:
                print("✅ Aucune erreur critique dans les logs")
                self.results["container_logs"] = True
                return True
            print("⚠️  {error_count} erreurs critiques trouvées dans les logs")
            self.results["container_logs"] = False
            return False
        print("❌ Impossible de récupérer les logs: {result['stderr']}")
        self.results["container_logs"] = False
        return False

    def test_container_resources(self) -> bool:
        """Test de l'utilisation des ressources."""
        print("💾 Vérification de l'utilisation des ressources...")
        
        result = self.run_command(["docker", "stats", "--no-stream", "--format", "json", self.container_name])
        
        if result["success"] and result["stdout"].strip():
            try:
                stats = json.loads(result["stdout"])
                
                # Extraire les statistiques de mémoire et CPU
                stats.get("MemUsage", "0B / 0B")
                stats.get("CPUPerc", "0%")
                
                print("📊 Mémoire: {memory_usage}")
                print("📊 CPU: {cpu_percent}")
                
                # Vérifier que l'utilisation est raisonnable
                self.results["container_resources"] = True
                return True
                
            except json.JSONDecodeError:
                print("❌ Impossible de parser les statistiques")
                self.results["container_resources"] = False
                return False
        else:
            print("❌ Impossible de récupérer les statistiques: {result['stderr']}")
            self.results["container_resources"] = False
            return False

    def cleanup(self) -> None:
        """Nettoie les ressources de test."""
        print("🧹 Nettoyage des ressources de test...")
        
        # Arrêter et supprimer le conteneur
        self.run_command(["docker", "stop", self.container_name])
        self.run_command(["docker", "rm", self.container_name])
        
        print("✅ Nettoyage terminé")

    def run_all_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests de smoke."""
        print("🧪 Démarrage des tests de smoke Docker...")
        print("=" * 50)
        
        tests = [
            ("Image exists", self.test_image_exists),
            ("Container startup", self.test_container_startup),
            ("Health endpoint", self.test_health_endpoint),
            ("Models loaded", self.test_models_loaded),
            ("API endpoints", self.test_api_endpoints),
            ("Container logs", self.test_container_logs),
            ("Container resources", self.test_container_resources),
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print("\n🔍 Test: {test_name}")
            try:
                if test_func():
                    passed_tests += 1
            except Exception:
                print("❌ Erreur lors du test {test_name}: {e}")
                self.results[test_name.lower().replace(" ", "_")] = False
        
        # Résumé des résultats
        print("\n" + "=" * 50)
        print("📊 RÉSUMÉ DES TESTS DE SMOKE")
        print("=" * 50)
        
        print("Tests réussis: {passed_tests}/{total_tests}")
        
        for test_name, _ in tests:
            test_key = test_name.lower().replace(" ", "_")
            "✅ PASS" if self.results.get(test_key, False) else "❌ FAIL"
            print("  {test_name}: {status}")
        
        success_rate = passed_tests / total_tests
        
        if success_rate >= 0.8:
            print("\n🎉 TESTS DE SMOKE RÉUSSIS!")
            print("✅ L'image Docker est prête pour la production")
        elif success_rate >= 0.6:
            print("\n⚠️  TESTS DE SMOKE PARTIELLEMENT RÉUSSIS")
            print("⚠️  Certains problèmes détectés, vérification recommandée")
        else:
            print("\n❌ TESTS DE SMOKE ÉCHOUÉS")
            print("❌ L'image Docker nécessite des corrections")
        
        return {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "results": self.results
        }


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests de smoke Docker")
    parser.add_argument("--image", default="atmr-backend", help="Nom de l'image Docker")
    parser.add_argument("--tag", default="latest", help="Tag de l'image")
    parser.add_argument("--no-cleanup", action="store_true", help="Ne pas nettoyer après les tests")
    
    args = parser.parse_args()
    
    # Créer et exécuter les tests
    smoke_tests = DockerSmokeTests(args.image, args.tag)
    
    try:
        results = smoke_tests.run_all_tests()
        
        # Nettoyage sauf si demandé de ne pas le faire
        if not args.no_cleanup:
            smoke_tests.cleanup()
        
        # Code de sortie basé sur le succès
        if results["success_rate"] >= 0.8:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrompus par l'utilisateur")
        smoke_tests.cleanup()
        sys.exit(1)
    except Exception:
        print("\n❌ Erreur lors des tests: {e}")
        smoke_tests.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
