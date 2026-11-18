import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  SLA_NOTICE,
  TRACKING_STORAGE_KEY,
  verifyOrganizationEmail,
} from '../../../services/organizationApplicationService';
import ApplicationStatusCard from './ApplicationStatusCard';
import styles from './OrganizationSignup.module.css';

export default function OrganizationEmailVerify() {
  const [params] = useSearchParams();
  const token = params.get('token') || '';
  const [view, setView] = useState(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(Boolean(token));

  useEffect(() => {
    if (!token) {
      return undefined;
    }
    let cancelled = false;
    window.history.replaceState({}, '', '/inscription/organisation/verifier');
    verifyOrganizationEmail(token)
      .then((data) => {
        if (cancelled) {
          return;
        }
        if (data.tracking_token) {
          sessionStorage.setItem(TRACKING_STORAGE_KEY, data.tracking_token);
        }
        setView(data);
      })
      .catch((requestError) => {
        if (!cancelled) {
          setError(
            requestError?.response?.data?.message
              || 'Ce lien n’est plus valable. Demandez un nouvel envoi depuis le formulaire.'
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setPending(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  const confirmed = view?.email_verified === true;
  let title = 'Vérification demandée';
  if (pending) {
    title = 'Vérification du lien';
  } else if (confirmed) {
    title = 'E-mail confirmé';
  } else if (error) {
    title = 'Lien inutilisable';
  } else if (!token) {
    title = 'Lien de confirmation';
  }

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <p className={styles.kicker}>Confirmation</p>
        <h1 className={styles.title}>{title}</h1>
        {pending ? <p className={styles.lead}>Vérification du lien…</p> : null}
        {!token && !view && !pending ? (
          <p className={styles.lead}>
            Ouvrez le dernier lien reçu par e-mail. Un lien plus ancien ne confirme plus le dossier.
          </p>
        ) : null}
        {error ? (
          <>
            <p className={styles.alert} role="alert">
              {error}
            </p>
            <p className={styles.hint}>
              Le lien est expiré, déjà utilisé, ou remplacé. Déposez à nouveau la demande :
              seul le nouveau message reste valable.
            </p>
          </>
        ) : null}
        {confirmed ? (
          <p className={styles.lead}>Dossier reçu. Aucun accès professionnel n’est ouvert.</p>
        ) : null}
        {view ? <ApplicationStatusCard application={view} /> : null}
        {!view ? <p className={styles.notice}>{SLA_NOTICE}</p> : null}
        <Link className={`${styles.choice} ${styles.choiceSecondary}`} to="/inscription/organisation/suivi">
          Suivre le dossier
        </Link>
      </div>
    </div>
  );
}
