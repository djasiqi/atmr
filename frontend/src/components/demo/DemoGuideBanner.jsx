import React, { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { trackDemoEvent } from '../../services/demoAnalyticsService';
import styles from './DemoGuideBanner.module.css';

const GUIDE = {
  transporteur: {
    title: 'Mission transporteur (5-7 min)',
    subtitle: 'Parcours recommandé: dashboard -> demande -> dispatch -> suivi -> facture.',
    steps: [
      'Ouvrir le tableau de bord et lire les KPI du jour.',
      'Créer ou ouvrir une demande de transport.',
      'Assigner un chauffeur à une course.',
      'Passer la course au statut terminé.',
      'Consulter une facture générée.',
    ],
  },
  institution: {
    title: 'Mission institution (3-5 min)',
    subtitle: 'Parcours recommandé: création demande -> suivi -> historique.',
    steps: [
      'Créer une demande avec les informations patient.',
      'Suivre le statut de prise en charge.',
      'Consulter l’historique des demandes.',
    ],
  },
};

const DemoGuideBanner = ({ role = 'transporteur' }) => {
  const navigate = useNavigate();
  const guide = useMemo(() => GUIDE[role] || GUIDE.transporteur, [role]);

  const onMarkStep = (stepIndex) => {
    trackDemoEvent('demo_step_reached', { role, stepIndex: stepIndex + 1 });
  };

  const onComplete = () => {
    trackDemoEvent('demo_completed', { role });
    navigate('/contact/demo');
  };

  return (
    <section className={styles.banner} data-tour-id={`demo-guide-${role}`}>
      <h2 className={styles.title}>{guide.title}</h2>
      <p className={styles.meta}>{guide.subtitle}</p>
      <ol className={styles.missions}>
        {guide.steps.map((step, index) => (
          <li key={step}>
            {step}{' '}
            <button type="button" className={styles.cta} onClick={() => onMarkStep(index)}>
              Étape faite
            </button>
          </li>
        ))}
      </ol>
      <div className={styles.footer}>
        <span className={styles.hint}>Guidage léger: les ancres `data-tour-id` restent stables.</span>
        <button type="button" className={styles.cta} onClick={onComplete}>
          Terminer et contacter LIRIE
        </button>
      </div>
    </section>
  );
};

export default DemoGuideBanner;

