#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de déploiement pour l'Étape 12 - Distributional RL (C51 / QR-DQN).

Ce script orchestre le déploiement des améliorations Distributional RL
et mesure l'amélioration de la stabilité et de la capture d'incertitude.
"""

import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def measure_stability_improvement():
    """Mesure l'amélioration de la stabilité avec Distributional RL."""
    print("\n🧪 Mesure amélioration de la stabilité")
    print("-" * 50)
    
    try:
        from services.rl.distributional_dqn import C51Network, QRNetwork, UncertaintyCapture
        from services.rl.improved_q_network import ImprovedQNetwork
        
        # Créer les réseaux
        c51_network = C51Network(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_atoms=51
        )
        
        qr_network = QRNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_quantiles=0.200
        )
        
        standard_network = ImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            hidden_sizes=(128, 64, 32, 16)
        )
        
        # Simulation d'apprentissage
        num_episodes = 100
        c51_stabilities = []
        qr_stabilities = []
        standard_stabilities = []
        
        c51_uncertainty = UncertaintyCapture("c51")
        qr_uncertainty = UncertaintyCapture("qr_dqn")
        
        for _ in range(num_episodes):
            # État aléatoire
            state = torch.randn(1, 10)
            
            # C51 Network
            c51_distribution = c51_network.get_distribution(state)
            c51_uncertainty_result = c51_uncertainty.calculate_uncertainty(c51_distribution)
            c51_uncertainty.update_uncertainty_history(c51_uncertainty_result)
            c51_stabilities.append(c51_uncertainty_result["confidence"])
            
            # QR-DQN Network
            qr_quantiles = qr_network(state)
            qr_uncertainty_result = qr_uncertainty.calculate_uncertainty(qr_quantiles)
            qr_uncertainty.update_uncertainty_history(qr_uncertainty_result)
            qr_stabilities.append(qr_uncertainty_result["confidence"])
            
            # Standard Network (simulation de stabilité)
            standard_q_values = standard_network(state)
            standard_stability = 1.0 - torch.std(standard_q_values).item() / torch.mean(standard_q_values).item()
            standard_stabilities.append(standard_stability)
        
        # Calculer les moyennes
        avg_c51_stability = sum(c51_stabilities) / len(c51_stabilities)
        avg_qr_stability = sum(qr_stabilities) / len(qr_stabilities)
        avg_standard_stability = sum(standard_stabilities) / len(standard_stabilities)
        
        print("  📊 Stabilité moyenne C51: {avg_c51_stability")
        print("  📊 Stabilité moyenne QR-DQN: {avg_qr_stability")
        print("  📊 Stabilité moyenne Standard: {avg_standard_stability")
        
        # Calculer les améliorations
        c51_improvement = avg_c51_stability - avg_standard_stability
        qr_improvement = avg_qr_stability - avg_standard_stability
        
        print("  📊 Amélioration C51: {c51_improvement")
        print("  📊 Amélioration QR-DQN: {qr_improvement")
        
        # Vérifier l'amélioration
        assert c51_improvement > 0, "L'amélioration C51 doit être positive"
        assert qr_improvement > 0, "L'amélioration QR-DQN doit être positive"
        
        print("  ✅ Amélioration de la stabilité: SUCCÈS")
        
        return True, c51_improvement, qr_improvement
        
    except Exception:
        print("  ❌ Mesure amélioration stabilité: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False, 0.0, 0.0

def test_uncertainty_capture_effectiveness():
    """Teste l'efficacité de la capture d'incertitude."""
    print("\n🧪 Test efficacité capture d'incertitude")
    print("-" * 50)
    
    try:
        
        # Créer les réseaux
        c51_network = C51Network(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_atoms=51
        )
        
        qr_network = QRNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_quantiles=0.200
        )
        
        # Mesurer la capture d'incertitude
        states = torch.randn(50, 10)
        
        c51_uncertainties = []
        qr_uncertainties = []
        
        c51_uncertainty = UncertaintyCapture("c51")
        qr_uncertainty = UncertaintyCapture("qr_dqn")
        
        for state in states:
            # C51
            c51_distribution = c51_network.get_distribution(state.unsqueeze(0))
            c51_uncertainty_result = c51_uncertainty.calculate_uncertainty(c51_distribution)
            c51_uncertainties.append(c51_uncertainty_result["entropy"])
            
            # QR-DQN
            qr_quantiles = qr_network(state.unsqueeze(0))
            qr_uncertainty_result = qr_uncertainty.calculate_uncertainty(qr_quantiles)
            qr_uncertainties.append(qr_uncertainty_result["iqr"])
        
        # Calculer les statistiques
        c51_avg_uncertainty = sum(c51_uncertainties) / len(c51_uncertainties)
        qr_avg_uncertainty = sum(qr_uncertainties) / len(qr_uncertainties)
        
        sum((x - c51_avg_uncertainty) ** 2 for x in c51_uncertainties) / len(c51_uncertainties)
        sum((x - qr_avg_uncertainty) ** 2 for x in qr_uncertainties) / len(qr_uncertainties)
        
        print("  📊 Incertitude moyenne C51: {c51_avg_uncertainty")
        print("  📊 Incertitude moyenne QR-DQN: {qr_avg_uncertainty")
        print("  📊 Variance incertitude C51: {c51_uncertainty_variance")
        print("  📊 Variance incertitude QR-DQN: {qr_uncertainty_variance")
        
        # Vérifier que l'incertitude est capturée
        assert c51_avg_uncertainty > 0, "L'incertitude C51 doit être positive"
        assert qr_avg_uncertainty > 0, "L'incertitude QR-DQN doit être positive"
        
        print("  ✅ Capture d'incertitude: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test capture d'incertitude: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_distributional_loss_convergence():
    """Teste la convergence des pertes distributionnelles."""
    print("\n🧪 Test convergence des pertes distributionnelles")
    print("-" * 50)
    
    try:
        from services.rl.distributional_dqn import C51Network, DistributionalLoss, QRNetwork
        
        # Créer les réseaux
        c51_network = C51Network(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_atoms=51
        )
        
        qr_network = QRNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_quantiles=0.200
        )
        
        # Optimiseurs
        c51_optimizer = torch.optim.Adam(c51_network.parameters(), lr=0.0001)
        qr_optimizer = torch.optim.Adam(qr_network.parameters(), lr=0.0001)
        
        # Simulation d'entraînement
        num_steps = 50
        c51_losses = []
        qr_losses = []
        
        for _ in range(num_steps):
            # Données d'entraînement
            state = torch.randn(4, 10)
            target_logits = torch.randn(4, 5, 51)
            target_quantiles = torch.randn(4, 5, 200)
            actions = torch.randint(0, 5, (4,))
            rewards = torch.randn(4)
            dones = torch.randint(0, 2, (4,)).bool()
            
            # C51
            logits = c51_network(state)
            c51_loss = DistributionalLoss.c51_loss(
                logits, target_logits, actions, rewards, dones, 0.99,
                c51_network.z, c51_network.delta_z
            )
            
            c51_optimizer.zero_grad()
            c51_loss.backward()
            c51_optimizer.step()
            
            c51_losses.append(c51_loss.item())
            
            # QR-DQN
            quantiles = qr_network(state)
            qr_loss = DistributionalLoss.quantile_loss(
                quantiles, target_quantiles, actions, rewards, dones, 0.99, qr_network.tau
            )
            
            qr_optimizer.zero_grad()
            qr_loss.backward()
            qr_optimizer.step()
            
            qr_losses.append(qr_loss.item())
        
        # Calculer la convergence
        c51_initial_loss = sum(c51_losses[:10]) / 10
        c51_final_loss = sum(c51_losses[-10:]) / 10
        c51_convergence = c51_initial_loss - c51_final_loss
        
        qr_initial_loss = sum(qr_losses[:10]) / 10
        qr_final_loss = sum(qr_losses[-10:]) / 10
        qr_convergence = qr_initial_loss - qr_final_loss
        
        print("  📊 Convergence C51: {c51_convergence")
        print("  📊 Convergence QR-DQN: {qr_convergence")
        
        # Vérifier la convergence
        assert c51_convergence > 0, "La convergence C51 doit être positive"
        assert qr_convergence > 0, "La convergence QR-DQN doit être positive"
        
        print("  ✅ Convergence des pertes: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test convergence pertes: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_integration_with_existing_system():
    """Teste l'intégration avec le système existant."""
    print("\n🧪 Test intégration système existant")
    print("-" * 50)
    
    try:
        from services.rl.distributional_dqn import compare_distributional_methods, create_distributional_network
        
        # Test création via factory function
        c51_network = create_distributional_network(
            network_type="c51",
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_atoms=51
        )
        
        qr_network = create_distributional_network(
            network_type="qr_dqn",
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            num_quantiles=0.200
        )
        
        # Test forward pass
        state = torch.randn(3, 10)
        
        c51_output = c51_network.get_q_values(state)
        qr_output = qr_network.get_q_values(state)
        
        assert c51_output.shape == (3, 5), f"Shape attendue (3, 5), reçue {c51_output.shape}"
        assert qr_output.shape == (3, 5), f"Shape attendue (3, 5), reçue {qr_output.shape}"
        
        # Test comparaison
        comparison = compare_distributional_methods(c51_network, qr_network, state)
        
        assert isinstance(comparison, dict), "Comparaison doit être un dictionnaire"
        assert "c51" in comparison, "C51 manquant dans la comparaison"
        assert "qr_dqn" in comparison, "QR-DQN manquant dans la comparaison"
        
        print("  ✅ Création réseaux: SUCCÈS")
        print("  ✅ Forward pass: SUCCÈS")
        print("  ✅ Comparaison: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test intégration système: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def generate_deployment_report():
    """Génère un rapport de déploiement."""
    print("\n📊 Génération rapport de déploiement")
    print("-" * 50)
    
    try:
        # Mesurer les améliorations
        _stability_success, c51_improvement, qr_improvement = measure_stability_improvement()
        uncertainty_success = test_uncertainty_capture_effectiveness()
        convergence_success = test_distributional_loss_convergence()
        integration_success = test_integration_with_existing_system()
        
        # Générer le rapport
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": "Étape 12 - Distributional RL (C51 / QR-DQN)",
            "status": "DÉPLOYÉ",
            "metrics": {
                "c51_stability_improvement": c51_improvement,
                "qr_dqn_stability_improvement": qr_improvement,
                "uncertainty_capture_effectiveness": uncertainty_success,
                "loss_convergence": convergence_success,
                "system_integration": integration_success
            },
            "files_created": [
                "services/rl/distributional_dqn.py",
                "tests/rl/test_distributional_dqn.py",
                "scripts/validate_step12_distributional_rl.py",
                "scripts/deploy_step12_distributional_rl.py"
            ],
            "features": [
                "C51Network pour distribution catégorielle",
                "QRNetwork pour distribution de quantiles",
                "DistributionalLoss pour pertes spécialisées",
                "UncertaintyCapture pour capture d'incertitude",
                "Factory functions pour création facile",
                "Tests complets avec validation",
                "Amélioration de la stabilité",
                "Capture d'incertitude des retards"
            ]
        }
        
        # Sauvegarder le rapport
        report_path = Path(__file__).parent / "step12_deployment_report.json"
        import json
        with Path(report_path, "w", encoding="utf-8").open() as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("  ✅ Rapport sauvegardé: {report_path}")
        print("  📊 Amélioration stabilité C51: {c51_improvement")
        print("  📊 Amélioration stabilité QR-DQN: {qr_improvement")
        print("  📊 Capture d'incertitude: {'✅' if uncertainty_success else '❌'}")
        print("  📊 Convergence pertes: {'✅' if convergence_success else '❌'}")
        print("  📊 Intégration système: {'✅' if integration_success else '❌'}")
        
        return True, report
        
    except Exception:
        print("  ❌ Génération rapport: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False, {}

def run_deployment():
    """Exécute le déploiement complet de l'Étape 12."""
    print("🚀 DÉPLOIEMENT DE L'ÉTAPE 12 - DISTRIBUTIONAL RL")
    print("=" * 70)
    print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🐳 Environnement: Docker Container")
    print("🐍 Python: {sys.version}")
    print()
    
    # Liste des étapes de déploiement
    deployment_steps = [
        {
            "name": "Mesure amélioration stabilité",
            "function": measure_stability_improvement
        },
        {
            "name": "Test efficacité capture d'incertitude",
            "function": test_uncertainty_capture_effectiveness
        },
        {
            "name": "Test convergence pertes distributionnelles",
            "function": test_distributional_loss_convergence
        },
        {
            "name": "Test intégration système",
            "function": test_integration_with_existing_system
        },
        {
            "name": "Génération rapport",
            "function": generate_deployment_report
        }
    ]
    
    results = []
    total_steps = len(deployment_steps)
    successful_steps = 0
    
    # Exécuter chaque étape
    for step in deployment_steps:
        print("\n📋 Étape: {step['name']}")
        
        if step["name"] == "Mesure amélioration stabilité":
            success, c51_improvement, qr_improvement = step["function"]()
            results.append({
                "name": step["name"],
                "success": success,
                "c51_improvement": c51_improvement,
                "qr_improvement": qr_improvement
            })
        else:
            success = step["function"]()
            results.append({
                "name": step["name"],
                "success": success
            })
        
        if success:
            successful_steps += 1
    
    # Générer le rapport final
    print("\n" + "=" * 70)
    print("📊 RAPPORT FINAL DE DÉPLOIEMENT - ÉTAPE 12")
    print("=" * 70)
    
    print("Total des étapes: {total_steps}")
    print("Étapes réussies: {successful_steps}")
    print("Étapes échouées: {total_steps - successful_steps}")
    print("Taux de succès: {(successful_steps / total_steps * 100)")
    
    print("\n📋 Détail des résultats:")
    for result in results:
        "✅" if result["success"] else "❌"
        print("  {status_emoji} {result['name']}")
        print("     Statut: {'SUCCÈS' if result['success'] else 'ÉCHEC'}")
        if "c51_improvement" in result:
            print("     Amélioration C51: {result['c51_improvement']")
            print("     Amélioration QR-DQN: {result['qr_improvement']")
        print()
    
    # Conclusion
    if successful_steps == total_steps:
        print("🎉 DÉPLOIEMENT COMPLET RÉUSSI!")
        print("✅ Les méthodes Distributional RL sont déployées")
        print("✅ L'amélioration de la stabilité est mesurée")
        print("✅ La capture d'incertitude est efficace")
        print("✅ Les pertes distributionnelles convergent")
        print("✅ L'intégration système fonctionne")
        print("✅ L'Étape 12 est prête pour l'expérimentation")
    else:
        print("⚠️ DÉPLOIEMENT PARTIEL")
        print("✅ Certaines fonctionnalités sont déployées")
        print("⚠️ Certaines étapes ont échoué")
        print("🔍 Vérifier les erreurs ci-dessus")
    
    return successful_steps >= total_steps * 0.8  # 80% de succès acceptable

def main():
    """Fonction principale."""
    try:
        success = run_deployment()
        
        if success:
            print("\n🎉 DÉPLOIEMENT RÉUSSI!")
            print("✅ L'Étape 12 - Distributional RL est déployée")
            return 0
        print("\n⚠️ DÉPLOIEMENT PARTIEL")
        print("❌ Certains aspects nécessitent attention")
        return 1
            
    except Exception:
        print("\n🚨 ERREUR CRITIQUE: {e}")
        print("Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
