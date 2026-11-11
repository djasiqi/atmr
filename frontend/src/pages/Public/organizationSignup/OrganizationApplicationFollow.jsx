import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  fetchMyOrganizationApplication,
  TRACKING_STORAGE_KEY,
} from '../../../services/organizationApplicationService';
import ApplicationStatusCard from './ApplicationStatusCard';
import { WORST_CASE_APPLICATION } from './worstCaseApplication';
import styles from './OrganizationSignup.module.css';

export default function OrganizationApplicationFollow() {
  const [params] = useSearchParams();
  const worstCase = process.env.NODE_ENV === 'development' && params.get('data') === 'worst';
  const [application, setApplication] = useState(worstCase ? WORST_CASE_APPLICATION : null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (worstCase) {
      return undefined;
    }
    const token = sessionStorage.getItem(TRACKING_STORAGE_KEY) || '';
    let cancelled = false;
    fetchMyOrganizationApplication(token)
      .then((data) => {
        if (!cancelled) {
          setApplication(data);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError('Ouvrez le lien de confirmation reçu par e-mail pour consulter votre dossier.');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [worstCase]);

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <p className={styles.kicker}>Suivi</p>
        <h1 className={styles.title}>Votre dossier</h1>
        {error ? (
          <p className={styles.alert} role="alert">
            {error}
          </p>
        ) : null}
        {!application && !error ? <p className={styles.lead}>Chargement du dossier…</p> : null}
        {application ? <ApplicationStatusCard application={application} /> : null}
        <Link className={`${styles.choice} ${styles.choiceSecondary}`} to="/inscription/organisation">
          Retour au choix
        </Link>
      </div>
    </div>
  );
}
