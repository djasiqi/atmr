import { useCallback, useEffect, useState } from 'react';
import {
  cancelDriverInvitation,
  createDriverInvitation,
  invitationErrorMessage,
  listDriverInvitations,
  resendDriverInvitation,
  revokeDriverInvitation,
} from '../../../services/driverInvitationService';
import styles from './DriverInvitations.module.css';

const EMPTY_FORM = { first_name: '', last_name: '', email: '' };

function deliveryText(notification) {
  if (!notification) {
    return 'Invitation enregistrée.';
  }
  if (notification.state === 'withheld') {
    return 'Invitation enregistrée. Le message n’a pas été remis : le destinataire est hors du périmètre de test.';
  }
  if (notification.delivered === true) {
    return 'Invitation enregistrée. La livraison est confirmée.';
  }
  return 'Invitation enregistrée. Le message est mis en file : la livraison n’est pas confirmée.';
}

export default function DriverInvitationsPanel({ formOpen, onCloseForm }) {
  const [rows, setRows] = useState([]);
  const [form, setForm] = useState(EMPTY_FORM);
  const [feedback, setFeedback] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);

  const load = useCallback(async () => {
    const data = await listDriverInvitations();
    setRows(Array.isArray(data?.invitations) ? data.invitations : []);
  }, []);

  useEffect(() => {
    let cancelled = false;
    listDriverInvitations()
      .then((data) => {
        if (!cancelled) {
          setRows(Array.isArray(data?.invitations) ? data.invitations : []);
        }
      })
      .catch((requestError) => {
        if (!cancelled) {
          setError(
            invitationErrorMessage(
              requestError,
              'Les invitations n’ont pas pu être chargées.'
            )
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const onChange = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  };

  const onCreate = async (event) => {
    event.preventDefault();
    setError('');
    setFeedback('');
    setPending(true);
    try {
      const created = await createDriverInvitation(form);
      setFeedback(deliveryText(created.notification));
      setForm(EMPTY_FORM);
      await load();
      onCloseForm();
    } catch (requestError) {
      setError(
        invitationErrorMessage(
          requestError,
          'L’invitation n’a pas pu être envoyée. Vos réponses sont conservées.'
        )
      );
      try {
        await load();
      } catch {
        // La liste sera rechargée à la prochaine ouverture.
      }
    } finally {
      setPending(false);
    }
  };

  const run = async (action, publicId, { confirmMessage } = {}) => {
    if (confirmMessage && !window.confirm(confirmMessage)) {
      return;
    }
    setError('');
    setFeedback('');
    setPending(true);
    try {
      const result = await action(publicId);
      setFeedback(
        result?.notification
          ? deliveryText(result.notification)
          : 'La liste des invitations est à jour.'
      );
      await load();
    } catch (requestError) {
      setError(invitationErrorMessage(requestError, 'L’action n’a pas abouti.'));
    } finally {
      setPending(false);
    }
  };

  return (
    <section className={styles.panel} aria-labelledby="driver-invitations-title">
      <h2 id="driver-invitations-title" className={styles.title}>
        Invitations chauffeur
      </h2>
      <p className={styles.note}>
        L’entreprise ne choisit pas le mot de passe. Le chauffeur rejoint uniquement
        cette entreprise, après avoir accepté le lien. L’activation du portail
        chauffeur reste fermée.
      </p>
      {formOpen ? (
        <form className={styles.form} onSubmit={onCreate}>
          <label className={styles.field}>
            <span>Prénom</span>
            <input
              className={styles.input}
              name="first_name"
              value={form.first_name}
              onChange={onChange}
              maxLength={80}
              autoComplete="given-name"
              required
            />
          </label>
          <label className={styles.field}>
            <span>Nom</span>
            <input
              className={styles.input}
              name="last_name"
              value={form.last_name}
              onChange={onChange}
              maxLength={80}
              autoComplete="family-name"
              required
            />
          </label>
          <label className={styles.field}>
            <span>E-mail</span>
            <input
              className={styles.input}
              name="email"
              type="email"
              value={form.email}
              onChange={onChange}
              maxLength={255}
              autoComplete="email"
              inputMode="email"
              required
            />
          </label>
          <div className={styles.actions}>
            <button className={styles.button} type="submit" disabled={pending}>
              Envoyer l’invitation
            </button>
            <button className={styles.secondary} type="button" onClick={onCloseForm}>
              Fermer
            </button>
          </div>
        </form>
      ) : null}
      {feedback ? (
        <p className={styles.feedback} role="status">
          {feedback}
        </p>
      ) : null}
      {error ? (
        <p className={styles.alert} role="alert">
          {error}
        </p>
      ) : null}
      {rows.length === 0 ? (
        <p className={styles.empty}>Aucune invitation pour le moment.</p>
      ) : (
        <ul className={styles.list}>
          {rows.map((row) => (
            <li key={row.public_id} className={styles.row}>
              <p className={styles.name}>
                {row.first_name} {row.last_name}
              </p>
              <p className={styles.meta}>
                {row.email} — {row.status_label}
              </p>
              <div className={styles.rowActions}>
                {row.status === 'pending' ? (
                  <>
                    <button
                      type="button"
                      className={styles.secondary}
                      disabled={pending}
                      onClick={() => run(resendDriverInvitation, row.public_id)}
                    >
                      Renvoyer
                    </button>
                    <button
                      type="button"
                      className={styles.danger}
                      disabled={pending}
                      onClick={() =>
                        run(cancelDriverInvitation, row.public_id, {
                          confirmMessage:
                            'Annuler cette invitation ? Le lien déjà envoyé ne fonctionnera plus.',
                        })
                      }
                    >
                      Annuler
                    </button>
                  </>
                ) : null}
                {row.status === 'accepted' ? (
                  <button
                    type="button"
                    className={styles.danger}
                    disabled={pending}
                    onClick={() =>
                      run(revokeDriverInvitation, row.public_id, {
                        confirmMessage:
                          'Retirer le rattachement de ce chauffeur à votre entreprise ? '
                          + 'Son compte personnel LIRIE n’est pas supprimé.',
                      })
                    }
                  >
                    Retirer le rattachement
                  </button>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
