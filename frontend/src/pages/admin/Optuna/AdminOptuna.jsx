import React, { useState, useEffect } from 'react';
import { FaRobot, FaPlay, FaSpinner, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';
import { runOptunaOptimization, fetchCompanies } from '../../../services/adminService';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import AdminSidebar from '../../../components/layout/Sidebar/AdminSidebar/AdminSidebar';
import styles from './AdminOptuna.module.css';

const AdminOptuna = () => {
  const [loading, setLoading] = useState(false);
  const [companies, setCompanies] = useState([]);
  const [status, setStatus] = useState(null); // 'idle', 'running', 'success', 'error'
  const [statusMessage, setStatusMessage] = useState('');
  
  // Configuration de l'optimisation
  const [config, setConfig] = useState({
    company_id: '',
    data_period: 'week',
    n_trials: 30,
    training_episodes: 150,
    eval_episodes: 15,
    custom_days: 7,
  });

  useEffect(() => {
    loadCompanies();
  }, []);

  const loadCompanies = async () => {
    try {
      const data = await fetchCompanies();
      setCompanies(data || []);
    } catch (error) {
      console.error('❌ Erreur chargement entreprises :', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setConfig((prev) => ({
      ...prev,
      [name]: name === 'company_id' || name === 'n_trials' || name === 'training_episodes' || name === 'eval_episodes' || name === 'custom_days'
        ? value === '' ? '' : parseInt(value, 10) || ''
        : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (config.n_trials < 1 || config.n_trials > 500) {
      alert('⚠️ Le nombre de trials doit être entre 1 et 500');
      return;
    }

    setLoading(true);
    setStatus('running');
    setStatusMessage('Démarrage de l\'optimisation Optuna...');

    try {
      // Préparer les données (enlever les champs vides)
      const payload = {
        data_period: config.data_period,
        n_trials: config.n_trials,
        training_episodes: config.training_episodes,
        eval_episodes: config.eval_episodes,
      };

      if (config.company_id) {
        payload.company_id = config.company_id;
      }

      if (config.data_period === 'custom') {
        payload.custom_days = config.custom_days;
      }

      await runOptunaOptimization(payload);
      
      setStatus('success');
      setStatusMessage(
        `✅ Optimisation démarrée avec succès ! ` +
        `Consultez https://optuna.lirie.ch pour suivre la progression.`
      );
      
      // Réinitialiser après 5 secondes
      setTimeout(() => {
        setStatus('idle');
        setStatusMessage('');
      }, 5000);

    } catch (error) {
      setStatus('error');
      setStatusMessage(
        error.response?.data?.message || 
        error.message || 
        'Erreur lors du démarrage de l\'optimisation'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <HeaderDashboard />
      <div className={styles.layout}>
        <AdminSidebar />
        <main className={styles.main}>
          <div className={styles.header}>
            <h1>
              <FaRobot /> Optimisation Optuna
            </h1>
            <p>
              Configurez et lancez l'optimisation des hyperparamètres DQN pour améliorer les performances
              du système de dispatch.
            </p>
          </div>

          {status && (
            <div className={`${styles.alert} ${styles[status]}`}>
              {status === 'running' && <FaSpinner className={styles.spinner} />}
              {status === 'success' && <FaCheckCircle />}
              {status === 'error' && <FaExclamationTriangle />}
              <span>{statusMessage}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="company_id">
                Entreprise (optionnel)
                <span className={styles.hint}>
                  Laisser vide pour optimiser toutes les entreprises
                </span>
              </label>
              <select
                id="company_id"
                name="company_id"
                value={config.company_id}
                onChange={handleInputChange}
              >
                <option value="">Toutes les entreprises</option>
                {companies.map((company) => (
                  <option key={company.id} value={company.id}>
                    {company.name} (ID: {company.id})
                  </option>
                ))}
              </select>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="data_period">
                Période de données
                <span className={styles.hint}>
                  Période des bookings à utiliser pour l'entraînement
                </span>
              </label>
              <select
                id="data_period"
                name="data_period"
                value={config.data_period}
                onChange={handleInputChange}
              >
                <option value="day">Jour (données du jour actuel)</option>
                <option value="week">Semaine (7 derniers jours) - Recommandé</option>
                <option value="month">Mois (30 derniers jours)</option>
                <option value="custom">Personnalisé</option>
              </select>
            </div>

            {config.data_period === 'custom' && (
              <div className={styles.formGroup}>
                <label htmlFor="custom_days">
                  Nombre de jours
                </label>
                <input
                  type="number"
                  id="custom_days"
                  name="custom_days"
                  min="1"
                  max="365"
                  value={config.custom_days}
                  onChange={handleInputChange}
                />
              </div>
            )}

            <div className={styles.formGroup}>
              <label htmlFor="n_trials">
                Nombre de trials
                <span className={styles.hint}>
                  Nombre d'expériences Optuna à exécuter (recommandé: 30-100)
                </span>
              </label>
              <input
                type="number"
                id="n_trials"
                name="n_trials"
                min="1"
                max="500"
                value={config.n_trials}
                onChange={handleInputChange}
                required
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="training_episodes">
                Épisodes d'entraînement par trial
                <span className={styles.hint}>
                  Nombre d'épisodes d'entraînement pour chaque trial (recommandé: 150-200)
                </span>
              </label>
              <input
                type="number"
                id="training_episodes"
                name="training_episodes"
                min="10"
                max="1000"
                value={config.training_episodes}
                onChange={handleInputChange}
                required
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="eval_episodes">
                Épisodes d'évaluation par trial
                <span className={styles.hint}>
                  Nombre d'épisodes d'évaluation pour chaque trial (recommandé: 15-20)
                </span>
              </label>
              <input
                type="number"
                id="eval_episodes"
                name="eval_episodes"
                min="1"
                max="100"
                value={config.eval_episodes}
                onChange={handleInputChange}
                required
              />
            </div>

            <div className={styles.formActions}>
              <button
                type="submit"
                disabled={loading}
                className={styles.submitButton}
              >
                {loading ? (
                  <>
                    <FaSpinner className={styles.spinner} /> Démarrage...
                  </>
                ) : (
                  <>
                    <FaPlay /> Lancer l'optimisation
                  </>
                )}
              </button>
            </div>
          </form>

          <div className={styles.infoBox}>
            <h3>ℹ️ Informations</h3>
            <ul>
              <li>
                L'optimisation s'exécute en arrière-plan et peut prendre plusieurs heures selon le nombre de trials.
              </li>
              <li>
                Suivez la progression en temps réel sur{' '}
                <a href="https://optuna.lirie.ch" target="_blank" rel="noopener noreferrer">
                  https://optuna.lirie.ch
                </a>
              </li>
              <li>
                Chaque entreprise aura sa propre étude Optuna pour une optimisation personnalisée.
              </li>
              <li>
                <strong>Recommandation :</strong> Utilisez "Semaine" pour les mises à jour régulières et "Mois" pour l'optimisation initiale.
              </li>
            </ul>
          </div>
        </main>
      </div>
    </div>
  );
};

export default AdminOptuna;

