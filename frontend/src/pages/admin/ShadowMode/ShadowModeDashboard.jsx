import React from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaChartLine, FaRobot } from 'react-icons/fa';
import useShadowMode from '../../../hooks/useShadowMode';
import styles from './ShadowModeDashboard.module.css';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Sidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';

/**
 * Dashboard Admin pour monitorer le Shadow Mode MDI.
 *
 * Affiche en temps réel :
 * - KPIs (taux d'accord, comparaisons, désaccords)
 * - Dernières prédictions et comparaisons
 * - Recommandations pour Phase 2 (GO/NO-GO)
 * - Métriques de performance MDI
 */
const ShadowModeDashboard = () => {
  const {
    status: _status,
    stats,
    predictions,
    comparisons,
    disagreements,
    highConfidenceDisagreements,
    loading,
    error,
    reload,
    isActive,
    agreementRate,
    totalComparisons,
    totalPredictions,
    isReadyForPhase2,
  } = useShadowMode({ autoRefresh: true, refreshInterval: 30000 });

  if (loading) {
    return (
      <div className={styles.container}>
        <HeaderDashboard />
        <div className={styles.layout}>
          <Sidebar />
          <main className={styles.main}>
            <div className={styles.loadingContainer}>
              <div className={styles.spinner}></div>
              <p>Chargement des données Shadow Mode...</p>
            </div>
          </main>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <HeaderDashboard />
        <div className={styles.layout}>
          <Sidebar />
          <main className={styles.main}>
            <div className={styles.errorContainer}>
              <FaExclamationTriangle className={styles.errorIcon} />
              <h2>Erreur de chargement</h2>
              <p>{error}</p>
              <button onClick={reload} className={styles.retryButton}>
                🔄 Réessayer
              </button>
            </div>
          </main>
        </div>
      </div>
    );
  }

  // Si Shadow Mode pas actif
  if (!isActive) {
    return (
      <div className={styles.container}>
        <HeaderDashboard />
        <div className={styles.layout}>
          <Sidebar />
          <main className={styles.main}>
            <div className={styles.header}>
              <h1>
                <FaRobot /> Shadow Mode MDI
              </h1>
              <p className={styles.subtitle}>
                Monitoring et validation du système de Reinforcement Learning
              </p>
            </div>

            <div className={styles.inactiveWarning}>
              <FaExclamationTriangle className={styles.warningIcon} />
              <div>
                <h3>🔍 Shadow Mode Inactif</h3>
                <p>
                  Le Shadow Mode n'est pas actuellement actif. Le système MDI doit être activé en
                  mode surveillance pour commencer la validation.
                </p>
                <div className={styles.inactiveActions}>
                  <h4>Actions recommandées :</h4>
                  <ol>
                    <li>Vérifier que le backend MDI est déployé</li>
                    <li>Activer les routes Shadow Mode (/api/shadow-mode/*)</li>
                    <li>Effectuer des assignations réelles pour générer des comparaisons</li>
                    <li>Attendre 1-2 semaines de données (objectif: &gt;1000 comparaisons)</li>
                  </ol>
                </div>
              </div>
            </div>
          </main>
        </div>
      </div>
    );
  }

  // Calcul de métriques supplémentaires
  const highConfidenceRate =
    predictions.length > 0
      ? predictions.filter((p) => (p.confidence || 0) > 0.8).length / predictions.length
      : 0;

  const dqnAssignRate =
    predictions.length > 0
      ? predictions.filter((p) => p.action_type === 'assign').length / predictions.length
      : 0;

  const actualAssignRate =
    comparisons.length > 0
      ? comparisons.filter((c) => c.actual_driver_id !== null).length / comparisons.length
      : 0;

  return (
    <div className={styles.container}>
      <HeaderDashboard />
      <div className={styles.layout}>
        <Sidebar />
        <main className={styles.main}>
          {/* Header */}
          <div className={styles.header}>
            <div>
              <h1>
                <FaRobot /> Shadow Mode MDI
              </h1>
              <p className={styles.subtitle}>
                Monitoring en temps réel • Actualisation automatique toutes les 30 secondes
              </p>
            </div>
            <button onClick={reload} className={styles.refreshButton}>
              🔄 Actualiser
            </button>
          </div>

          {/* Recommandation Phase 2 */}
          {isReadyForPhase2 ? (
            <div className={styles.recommendationSuccess}>
              <FaCheckCircle className={styles.successIcon} />
              <div>
                <h3>✅ PRÊT POUR PHASE 2 (A/B Testing)!</h3>
                <p>
                  Le système MDI a atteint <strong>{(agreementRate * 100).toFixed(1)}%</strong> de
                  taux d'accord sur <strong>{totalComparisons}+</strong> comparaisons. Tous les
                  critères de validation sont remplis.
                </p>
                <div className={styles.phase2Actions}>
                  <strong>Prochaines étapes :</strong>
                  <ol>
                    <li>
                      Analyser les désaccords haute confiance ({highConfidenceDisagreements.length})
                    </li>
                    <li>Exporter le rapport de validation</li>
                    <li>Obtenir l'approbation pour Phase 2</li>
                    <li>Configurer A/B Testing (50/50 MDI vs Système actuel)</li>
                  </ol>
                </div>
              </div>
            </div>
          ) : (
            <div className={styles.recommendationInfo}>
              <FaChartLine className={styles.infoIcon} />
              <div>
                <h3>⏳ Shadow Mode en cours de validation</h3>
                <p>
                  Le système MDI est en phase de monitoring. Taux d'accord actuel:{' '}
                  <strong>{(agreementRate * 100).toFixed(1)}%</strong> (objectif: &gt;75%).
                  Comparaisons: <strong>{totalComparisons}</strong> (objectif: &gt;1000).
                </p>
                <div className={styles.progressBars}>
                  <div className={styles.progressItem}>
                    <label>
                      Taux d'accord : {(agreementRate * 100).toFixed(1)}% / 75%{' '}
                      {agreementRate >= 0.75 ? '✅' : '⏳'}
                    </label>
                    <div className={styles.progressBar}>
                      <div
                        className={styles.progressFill}
                        style={{
                          width: `${Math.min((agreementRate / 0.75) * 100, 100)}%`,
                          backgroundColor: agreementRate >= 0.75 ? '#4caf50' : '#ff9800',
                        }}
                      ></div>
                    </div>
                  </div>
                  <div className={styles.progressItem}>
                    <label>
                      Comparaisons : {totalComparisons} / 1000{' '}
                      {totalComparisons >= 1000 ? '✅' : '⏳'}
                    </label>
                    <div className={styles.progressBar}>
                      <div
                        className={styles.progressFill}
                        style={{
                          width: `${Math.min((totalComparisons / 1000) * 100, 100)}%`,
                          backgroundColor: totalComparisons >= 1000 ? '#4caf50' : '#ff9800',
                        }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* KPIs */}
          <div className={styles.kpisGrid}>
            <div className={styles.kpiCard}>
              <div className={styles.kpiHeader}>
                <span className={styles.kpiIcon}>📊</span>
                <h3>Taux d'Accord</h3>
              </div>
              <div
                className={`${styles.kpiValue} ${
                  agreementRate >= 0.75 ? styles.success : styles.warning
                }`}
              >
                {(agreementRate * 100).toFixed(1)}%
              </div>
              <p className={styles.kpiSubtext}>
                {stats?.agreements_count || 0} accords / {totalComparisons} comparaisons
              </p>
              <div className={styles.kpiFooter}>Objectif: &gt;75%</div>
            </div>

            <div className={styles.kpiCard}>
              <div className={styles.kpiHeader}>
                <span className={styles.kpiIcon}>🔢</span>
                <h3>Comparaisons</h3>
              </div>
              <div
                className={`${styles.kpiValue} ${
                  totalComparisons >= 1000 ? styles.success : styles.warning
                }`}
              >
                {totalComparisons}
              </div>
              <p className={styles.kpiSubtext}>{totalPredictions} prédictions MDI</p>
              <div className={styles.kpiFooter}>Objectif: &gt;1000</div>
            </div>

            <div className={styles.kpiCard}>
              <div className={styles.kpiHeader}>
                <span className={styles.kpiIcon}>⚠️</span>
                <h3>Désaccords</h3>
              </div>
              <div className={styles.kpiValue}>{disagreements.length}</div>
              <p className={styles.kpiSubtext}>
                {highConfidenceDisagreements.length} haute confiance (&gt;80%)
              </p>
              <div className={styles.kpiFooter}>À analyser</div>
            </div>

            <div className={styles.kpiCard}>
              <div className={styles.kpiHeader}>
                <span className={styles.kpiIcon}>🎯</span>
                <h3>Phase 2</h3>
              </div>
              <div
                className={`${styles.kpiValue} ${isReadyForPhase2 ? styles.success : styles.info}`}
              >
                {isReadyForPhase2 ? '✅ Prêt' : '⏳ En cours'}
              </div>
              <p className={styles.kpiSubtext}>
                {isReadyForPhase2 ? 'Validation complète' : 'Monitoring actif'}
              </p>
              <div className={styles.kpiFooter}>{isReadyForPhase2 ? 'GO' : 'NO-GO'}</div>
            </div>
          </div>

          {/* Métriques Supplémentaires */}
          <div className={styles.section}>
            <h2>📈 Métriques Détaillées</h2>
            <div className={styles.metricsGrid}>
              <div className={styles.metricItem}>
                <label>Confiance Haute (&gt;80%)</label>
                <div className={styles.metricBar}>
                  <div
                    className={styles.metricFill}
                    style={{ width: `${(highConfidenceRate * 100).toFixed(0)}%` }}
                  ></div>
                  <span>{(highConfidenceRate * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div className={styles.metricItem}>
                <label>MDI Taux Assignation</label>
                <div className={styles.metricBar}>
                  <div
                    className={styles.metricFill}
                    style={{ width: `${(dqnAssignRate * 100).toFixed(0)}%` }}
                  ></div>
                  <span>{(dqnAssignRate * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div className={styles.metricItem}>
                <label>Système Réel Taux Assignation</label>
                <div className={styles.metricBar}>
                  <div
                    className={styles.metricFill}
                    style={{ width: `${(actualAssignRate * 100).toFixed(0)}%` }}
                  ></div>
                  <span>{(actualAssignRate * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Tables */}
          <div className={styles.tablesContainer}>
            {/* Table Comparaisons */}
            <div className={styles.section}>
              <h2>🔍 Dernières Comparaisons (MDI vs Réel)</h2>
              <div className={styles.tableWrapper}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Booking</th>
                      <th>MDI Prédit</th>
                      <th>Réel</th>
                      <th>Accord</th>
                      <th>Confiance</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisons.length === 0 ? (
                      <tr>
                        <td colSpan={6} className={styles.emptyMessage}>
                          Aucune comparaison disponible pour le moment.
                        </td>
                      </tr>
                    ) : (
                      comparisons.slice(0, 20).map((comp, idx) => (
                        <tr
                          key={idx}
                          className={comp.agreement ? styles.rowSuccess : styles.rowWarning}
                        >
                          <td>#{comp.booking_id}</td>
                          <td>Driver #{comp.predicted_driver_id || 'wait'}</td>
                          <td>Driver #{comp.actual_driver_id || 'wait'}</td>
                          <td>
                            {comp.agreement ? (
                              <span className={styles.badgeSuccess}>✅ Accord</span>
                            ) : (
                              <span className={styles.badgeWarning}>⚠️ Désaccord</span>
                            )}
                          </td>
                          <td>
                            <span
                              className={
                                (comp.confidence || 0) > 0.8
                                  ? styles.badgeSuccess
                                  : (comp.confidence || 0) > 0.5
                                    ? styles.badgeInfo
                                    : styles.badgeWarning
                              }
                            >
                              {((comp.confidence || 0) * 100).toFixed(0)}%
                            </span>
                          </td>
                          <td>{new Date(comp.timestamp).toLocaleString('fr-FR')}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Table Désaccords Haute Confiance */}
            {highConfidenceDisagreements.length > 0 && (
              <div className={styles.section}>
                <h2>⚠️ Désaccords Haute Confiance (À Investiguer)</h2>
                <p className={styles.sectionSubtitle}>
                  Ces cas montrent un désaccord entre MDI et système réel malgré une confiance
                  élevée (&gt;80%).
                </p>
                <div className={styles.tableWrapper}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Booking</th>
                        <th>MDI Prédit</th>
                        <th>Réel</th>
                        <th>Confiance</th>
                        <th>Q-Value</th>
                        <th>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {highConfidenceDisagreements.slice(0, 10).map((comp, idx) => (
                        <tr key={idx} className={styles.rowWarning}>
                          <td>#{comp.booking_id}</td>
                          <td>Driver #{comp.predicted_driver_id || 'wait'}</td>
                          <td>Driver #{comp.actual_driver_id || 'wait'}</td>
                          <td>
                            <span className={styles.badgeWarning}>
                              {((comp.confidence || 0) * 100).toFixed(0)}%
                            </span>
                          </td>
                          <td>{comp.q_value ? comp.q_value.toFixed(1) : 'N/A'}</td>
                          <td>{new Date(comp.timestamp).toLocaleString('fr-FR')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Footer Actions */}
          <div className={styles.footer}>
            <p className={styles.footerText}>
              💡 <strong>Conseil:</strong> Continuez à utiliser le système normalement. Le Shadow
              Mode enregistre toutes les décisions en arrière-plan sans impacter les opérations.
            </p>
            <div className={styles.footerActions}>
              <button className={styles.secondaryButton} onClick={() => window.print()}>
                📄 Exporter Rapport
              </button>
              {isReadyForPhase2 && (
                <button className={styles.primaryButton}>🚀 Passer en Phase 2 (A/B Testing)</button>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default ShadowModeDashboard;
