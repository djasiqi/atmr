import React, { useState, useEffect } from 'react';
import { FaRobot, FaPlay, FaSpinner, FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';
import { runOptunaOptimization, fetchCompanies, trainModelWithOptimalParams } from '../../../services/adminService';
import styles from './AdminOptuna.module.css';
import shell from '../adminShell.module.css';

const AdminOptuna = () => {
  const [loading, setLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [companies, setCompanies] = useState([]);
  const [status, setStatus] = useState(null); // 'idle', 'running', 'success', 'error'
  const [statusMessage, setStatusMessage] = useState('');
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingStatusMessage, setTrainingStatusMessage] = useState('');
  
  // Configuration de l'optimisation
  const [config, setConfig] = useState({
    company_id: '',
    data_period: 'week',
    n_trials: 30,
    training_episodes: 150,
    eval_episodes: 15,
    custom_days: 7,
  });

  // Configuration de l'entraînement
  const [trainingConfig, setTrainingConfig] = useState({
    config_path: '',
    study_name: '',
    model_output_path: 'data/rl/models/dqn_optimized.pth',
    training_episodes: 1000,
    eval_episodes: 50,
    company_id: '',
  });

  useEffect(() => {
    loadCompanies();
  }, []);

  const loadCompanies = async () => {
    try {
      const data = await fetchCompanies();
      setCompanies(data || []);
    } catch (error) {
      console.error('Erreur chargement entreprises :', error);
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
      alert('Le nombre de trials doit être entre 1 et 500');
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
        `Optimisation demarree avec succes. ` +
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
        'Erreur lors du demarrage de l optimisation'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleTrainingSubmit = async (e) => {
    if (e) e.preventDefault();
    
    if (trainingConfig.training_episodes < 100 || trainingConfig.training_episodes > 10000) {
      alert('Le nombre d episodes d entrainement doit etre entre 100 et 10000');
      return;
    }

    setTrainingLoading(true);
    setTrainingStatus('running');
    setTrainingStatusMessage('Démarrage de l\'entraînement du modèle...');

    try {
      // Préparer les données (enlever les champs vides)
      const payload = {
        model_output_path: trainingConfig.model_output_path,
        training_episodes: trainingConfig.training_episodes,
        eval_episodes: trainingConfig.eval_episodes,
      };

      if (trainingConfig.config_path) {
        payload.config_path = trainingConfig.config_path;
      }

      if (trainingConfig.study_name) {
        payload.study_name = trainingConfig.study_name;
      }

      if (trainingConfig.company_id) {
        payload.company_id = trainingConfig.company_id;
      }

      await trainModelWithOptimalParams(payload);
      
      setTrainingStatus('success');
      setTrainingStatusMessage(
        `Entrainement demarre avec succes. ` +
        `Le modele sera sauvegarde dans ${trainingConfig.model_output_path} une fois termine.`
      );
      
      // Réinitialiser après 5 secondes
      setTimeout(() => {
        setTrainingStatus('idle');
        setTrainingStatusMessage('');
      }, 5000);

    } catch (error) {
      setTrainingStatus('error');
      setTrainingStatusMessage(
        error.response?.data?.message || 
        error.message || 
        'Erreur lors du demarrage de l entrainement'
      );
    } finally {
      setTrainingLoading(false);
    }
  };

  return (
    <main className={shell.content}>
          <div className={styles.header}>
            <h1>
              <FaRobot /> Optimisation Optuna
            </h1>
            <p>
              Configurez et lancez l optimisation des hyperparametres DQN pour ameliorer les performances
              du systeme de dispatch.
            </p>
          </div>

          {status && (
            <div className={`${styles.alert} ${styles[status]}`} role="status" aria-live="polite">
              {status === 'running' && <FaSpinner className={styles.spinner} />}
              {status === 'success' && <FaCheckCircle />}
              {status === 'error' && <FaExclamationTriangle />}
              <span>{statusMessage}</span>
            </div>
          )}

          <section className={styles.surfaceCard}>
            <div className={styles.sectionHeader}>
              <h2>Configuration de l optimisation</h2>
              <p>Selectionnez le perimetre de donnees et les parametres d exploration.</p>
            </div>

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
                className={styles.input}
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
                className={styles.input}
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
                  className={styles.input}
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
                className={styles.input}
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
                className={styles.input}
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
                className={styles.input}
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
          </section>

          {/* Section Entraînement du modèle */}
          <section className={styles.surfaceCard}>
            <div className={styles.sectionHeader}>
              <h2>
                <FaRobot /> Entrainement du modele avec hyperparametres optimaux
              </h2>
              <p>
                Lancez ensuite un entrainement complet avec la meilleure configuration retenue.
              </p>
            </div>

            {trainingStatus && (
              <div className={`${styles.alert} ${styles[trainingStatus]}`} role="status" aria-live="polite">
                {trainingStatus === 'running' && <FaSpinner className={styles.spinner} />}
                {trainingStatus === 'success' && <FaCheckCircle />}
                {trainingStatus === 'error' && <FaExclamationTriangle />}
                <span>{trainingStatusMessage}</span>
              </div>
            )}

            <form onSubmit={(e) => {
              e.preventDefault();
              handleTrainingSubmit();
            }} className={styles.form}>
              <div className={styles.formGroup}>
                <label htmlFor="training_config_path">
                  Chemin vers optimal_config.json
                  <span className={styles.hint}>
                    Ex: data/rl/optimal_config_dqn_optimization_all_companies.json
                  </span>
                </label>
                <input
                  type="text"
                  id="training_config_path"
                  name="config_path"
                  value={trainingConfig.config_path}
                  onChange={(e) => setTrainingConfig({...trainingConfig, config_path: e.target.value})}
                  placeholder="data/rl/optimal_config_*.json"
                  className={styles.input}
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="training_study_name">
                  OU Nom de l'étude Optuna
                  <span className={styles.hint}>
                    Ex: dqn_optimization_all_companies
                  </span>
                </label>
                <input
                  type="text"
                  id="training_study_name"
                  name="study_name"
                  value={trainingConfig.study_name}
                  onChange={(e) => setTrainingConfig({...trainingConfig, study_name: e.target.value})}
                  placeholder="dqn_optimization_all_companies"
                  className={styles.input}
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="training_model_output_path">
                  Chemin de sortie du modèle
                  <span className={styles.hint}>
                    Où sauvegarder le modèle entraîné
                  </span>
                </label>
                <input
                  type="text"
                  id="training_model_output_path"
                  name="model_output_path"
                  value={trainingConfig.model_output_path}
                  onChange={(e) => setTrainingConfig({...trainingConfig, model_output_path: e.target.value})}
                  className={styles.input}
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="full_training_episodes">
                  Episodes d entrainement
                  <span className={styles.hint}>
                    Nombre d episodes d entrainement complet (recommande: 1000+)
                  </span>
                </label>
                <input
                  type="number"
                  id="full_training_episodes"
                  name="training_episodes"
                  min="100"
                  max="10000"
                  value={trainingConfig.training_episodes}
                  onChange={(e) => setTrainingConfig({...trainingConfig, training_episodes: parseInt(e.target.value, 10) || 1000})}
                  required
                  className={styles.input}
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="training_eval_episodes">
                  Épisodes d'évaluation
                  <span className={styles.hint}>
                    Nombre d'épisodes d'évaluation finale (recommandé: 50)
                  </span>
                </label>
                <input
                  type="number"
                  id="training_eval_episodes"
                  name="eval_episodes"
                  min="10"
                  max="200"
                  value={trainingConfig.eval_episodes}
                  onChange={(e) => setTrainingConfig({...trainingConfig, eval_episodes: parseInt(e.target.value, 10) || 50})}
                  required
                  className={styles.input}
                />
              </div>

              <div className={styles.formActions}>
                <button
                  type="submit"
                  disabled={trainingLoading}
                  className={styles.submitButton}
                >
                  {trainingLoading ? (
                    <>
                      <FaSpinner className={styles.spinner} /> Démarrage...
                    </>
                  ) : (
                    <>
                      <FaPlay /> Lancer l entrainement
                    </>
                  )}
                </button>
              </div>
            </form>
          </section>

          <div className={styles.infoBox}>
            <h3>Informations utiles</h3>
            <ul>
              <li>
                L optimisation s execute en arriere-plan et peut prendre plusieurs heures selon le nombre de trials.
              </li>
              <li>
                Suivez la progression en temps reel sur{' '}
                <a href="https://optuna.lirie.ch" target="_blank" rel="noopener noreferrer">
                  https://optuna.lirie.ch
                </a>
              </li>
              <li>
                Chaque entreprise aura sa propre etude Optuna pour une optimisation personnalisee.
              </li>
              <li>
                <strong>Recommandation :</strong> utilisez "Semaine" pour les mises a jour regulieres et "Mois" pour l optimisation initiale.
              </li>
              <li>
                <strong>Apres l optimisation :</strong> utilisez la section "Entrainement du modele" pour lancer un entrainement complet.
              </li>
            </ul>
          </div>
        </main>
  );
};

export default AdminOptuna;

