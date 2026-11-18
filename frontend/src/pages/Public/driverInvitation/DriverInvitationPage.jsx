import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  ACCEPT_NETWORK_MESSAGE,
  acceptDriverInvitation,
  openDriverInvitation,
} from '../../../services/driverInvitationService';
import styles from '../organizationSignup/OrganizationSignup.module.css';

const CLOSED_NOTICE =
  'L’activation professionnelle n’est pas ouverte. Vérifiez l’entreprise : accepter ce lien ne crée pas d’accès chauffeur.';

function viewTitle(view, error) {
  if (!view && error) {
    return 'Lien inutilisable';
  }
  if (!view) {
    return 'Invitation chauffeur';
  }
  if (view.status === 'expired') {
    return 'Lien expiré';
  }
  if (view.status === 'cancelled') {
    return 'Invitation annulée';
  }
  if (view.status === 'accepted') {
    return 'Invitation déjà utilisée';
  }
  if (view.status === 'revoked') {
    return 'Rattachement retiré';
  }
  return 'Invitation reçue';
}

export default function DriverInvitationPage() {
  const [params] = useSearchParams();
  const [token] = useState(params.get('token') || '');
  const [view, setView] = useState(null);
  const [error, setError] = useState('');
  const [pending, setPending] = useState(Boolean(token));
  const [password, setPassword] = useState('');
  const [accepted, setAccepted] = useState(false);
  const [done, setDone] = useState('');

  useEffect(() => {
    if (!token) {
      return undefined;
    }
    window.history.replaceState({}, '', '/invitation/chauffeur');
    let cancelled = false;
    openDriverInvitation(token)
      .then((data) => {
        if (!cancelled) {
          setView(data);
        }
      })
      .catch((requestError) => {
        if (!cancelled) {
          setError(
            requestError?.response
              ? requestError.response.data?.message || 'Ce lien n’est pas valable.'
              : ACCEPT_NETWORK_MESSAGE
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

  const onAccept = async (event) => {
    event.preventDefault();
    setError('');
    setPending(true);
    try {
      const result = await acceptDriverInvitation({
        token,
        accept: true,
        password: view?.identity === 'new' ? password : undefined,
      });
      setDone(
        result.already_accepted
          ? 'Cette invitation était déjà acceptée. Aucun second accès n’a été créé.'
          : result.portal_open
            ? 'Acceptation enregistrée. Le portail chauffeur de cette entreprise est ouvert.'
            : 'Acceptation enregistrée. Le portail chauffeur n’est pas ouvert.'
      );
    } catch (requestError) {
      if (!requestError?.response) {
        setError(ACCEPT_NETWORK_MESSAGE);
      } else if (requestError.response.status === 401) {
        setError(
          'Connectez-vous avec le compte destinataire. L’adresse e-mail seule ne rattache pas le compte.'
        );
      } else {
        setError(
          requestError.response.data?.message
            || 'L’invitation n’a pas pu être acceptée.'
        );
      }
    } finally {
      setPending(false);
    }
  };

  const title = token ? viewTitle(view, error) : 'Invitation chauffeur';
  const canAnswer = view?.status === 'pending' && !done;

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <p className={styles.kicker}>Chauffeur</p>
        <h1 className={styles.title}>{title}</h1>
        {!token ? (
          <p className={styles.lead}>
            Ouvrez le lien reçu de votre entreprise. Il n’existe pas d’inscription chauffeur publique.
          </p>
        ) : null}
        {pending && !view ? <p className={styles.lead}>Vérification du lien…</p> : null}
        {view ? (
          <section className={styles.panel} aria-label="Entreprise invitante">
            <strong>{view.company_name}</strong>
            <p className={styles.hint}>
              {view.first_name}, cette invitation concerne uniquement cette entreprise.
              Adresse indiquée : {view.email_hint}.
            </p>
            {view.status === 'pending' && view.activation?.enabled === false ? (
              <p className={styles.notice}>{CLOSED_NOTICE}</p>
            ) : null}
          </section>
        ) : null}
        {error ? (
          <p className={styles.alert} role="alert">
            {error}
          </p>
        ) : null}
        {done ? (
          <p className={styles.notice} role="status">
            {done}
          </p>
        ) : null}
        {canAnswer ? (
          <form className={styles.form} onSubmit={onAccept}>
            {view.identity === 'account_present' ? (
              <p className={styles.lead}>
                Un compte existe déjà pour cette invitation. Connectez-vous, puis acceptez.
                L’adresse seule ne fusionne pas deux identités.
              </p>
            ) : (
              <label className={styles.field}>
                <span>Choisissez votre mot de passe</span>
                <input
                  className={styles.input}
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="new-password"
                  required
                />
              </label>
            )}
            <label className={styles.check}>
              <input
                type="checkbox"
                checked={accepted}
                onChange={(event) => setAccepted(event.target.checked)}
                required
              />
              <span>J’accepte de rejoindre uniquement cette entreprise.</span>
            </label>
            {view.identity === 'account_present' ? (
              <Link className={styles.choiceSecondary} to="/login">
                Se connecter
              </Link>
            ) : null}
            <button className={styles.button} type="submit" disabled={pending || !accepted}>
              Accepter l’invitation
            </button>
          </form>
        ) : null}
      </div>
    </div>
  );
}
