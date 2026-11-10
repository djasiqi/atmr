import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  SLA_NOTICE,
  submissionErrorMessage,
  submitOrganizationApplication,
} from '../../../services/organizationApplicationService';
import styles from './OrganizationSignup.module.css';

const CATEGORIES = [
  ['hopital', 'Hôpital'],
  ['clinique', 'Clinique'],
  ['ems', 'EMS'],
  ['imad', 'IMAD'],
  ['curatelle', 'Curatelle'],
  ['autre', 'Autre'],
];

const EMPTY = {
  legal_name: '',
  ide_number: '',
  establishment_name: '',
  establishment_category: '',
  street: '',
  postal_code: '',
  city: '',
  applicant_first_name: '',
  applicant_last_name: '',
  applicant_function: '',
  applicant_email: '',
  applicant_phone: '',
  review_note: '',
  privacy_consent: false,
  website: '',
};

function firstInvalidField(kind, values) {
  const companyName = kind === 'company' ? 'legal_name' : 'establishment_name';
  const order = [
    companyName,
    kind === 'institution' ? 'establishment_category' : null,
    'street',
    'postal_code',
    'city',
    'applicant_first_name',
    'applicant_last_name',
    'applicant_function',
    'applicant_email',
    'applicant_phone',
    'privacy_consent',
  ].filter(Boolean);
  const messages = {
    legal_name: 'Indiquez la raison sociale.',
    establishment_name: 'Indiquez le nom de l’établissement.',
    establishment_category: 'Choisissez la catégorie de l’établissement.',
    street: 'Indiquez la rue.',
    postal_code: 'Indiquez un code postal suisse à 4 chiffres.',
    city: 'Indiquez la localité.',
    applicant_first_name: 'Indiquez le prénom.',
    applicant_last_name: 'Indiquez le nom.',
    applicant_function: 'Indiquez la fonction.',
    applicant_email: 'Indiquez une adresse e-mail professionnelle valide.',
    applicant_phone: 'Indiquez un téléphone professionnel valide.',
    privacy_consent: 'Le consentement au traitement du dossier est requis.',
  };
  for (const name of order) {
    const value = values[name];
    if (name === 'privacy_consent' && value !== true) {
      return { name, message: messages[name] };
    }
    if (name === 'postal_code' && !/^\d{4}$/.test(String(value || '').trim())) {
      return { name, message: messages[name] };
    }
    if (name === 'applicant_email' && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(value || '').trim())) {
      return { name, message: messages[name] };
    }
    if (name === 'applicant_phone' && String(value || '').replace(/[^\d+]/g, '').length < 8) {
      return { name, message: messages[name] };
    }
    if (name !== 'privacy_consent' && !String(value || '').trim()) {
      return { name, message: messages[name] };
    }
  }
  return null;
}

export default function OrganizationSignupForm({ kind }) {
  const isCompany = kind === 'company';
  const [values, setValues] = useState(EMPTY);
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);
  const [pending, setPending] = useState(false);
  const [invalidName, setInvalidName] = useState('');
  const lock = useRef(false);

  function update(event) {
    const { name, value, type, checked } = event.target;
    setValues((current) => ({
      ...current,
      [name]: type === 'checkbox' ? checked : value,
    }));
    setError('');
    setInvalidName('');
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (lock.current) {
      return;
    }
    const invalid = firstInvalidField(kind, values);
    if (invalid) {
      setError(invalid.message);
      setInvalidName(invalid.name);
      const field = event.currentTarget.elements.namedItem(invalid.name);
      if (field && 'focus' in field) {
        field.focus();
      }
      return;
    }
    lock.current = true;
    setPending(true);
    setError('');
    try {
      await submitOrganizationApplication({
        kind,
        ...values,
        privacy_consent: values.privacy_consent === true,
      });
      setDone(true);
    } catch (requestError) {
      setError(submissionErrorMessage(requestError));
    } finally {
      lock.current = false;
      setPending(false);
    }
  }

  if (done) {
    return (
      <div className={styles.page}>
        <div className={styles.shell}>
          <p className={styles.kicker}>Demande</p>
          <h1 className={styles.title}>Vérification demandée</h1>
          <p className={styles.lead}>
            Si cette adresse peut recevoir un message, un lien de confirmation vient d’être envoyé.
            Il reste valable 24 heures. Utilisez seulement ce dernier lien : un nouvel envoi remplace le précédent.
          </p>
          <p className={styles.notice}>
            Le dossier n’est reçu qu’après confirmation de l’e-mail. Aucun accès professionnel n’est ouvert.
          </p>
          <p className={styles.notice}>{SLA_NOTICE}</p>
          <Link className={styles.choice} to="/inscription/organisation/suivi">
            Suivre le dossier
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <p className={styles.kicker}>Préinscription</p>
        <h1 className={styles.title}>
          {isCompany ? 'Entreprise de transport' : 'Institution'}
        </h1>
        <p className={styles.lead}>
          {SLA_NOTICE} Aucun accès professionnel n’est ouvert avant une décision ultérieure.
        </p>
        <form className={styles.form} onSubmit={onSubmit} noValidate>
          {isCompany ? (
            <>
              <label className={styles.field}>
                <span>Raison sociale</span>
                <input
                  className={styles.input}
                  name="legal_name"
                  maxLength={200}
                  autoComplete="organization"
                  value={values.legal_name}
                  onChange={update}
                  aria-invalid={invalidName === 'legal_name'}
                  required
                />
              </label>
              <label className={styles.field}>
                <span>IDE / UID, si vous en avez un</span>
                <input
                  className={styles.input}
                  name="ide_number"
                  maxLength={20}
                  autoComplete="off"
                  placeholder="CHE-123.456.789"
                  value={values.ide_number}
                  onChange={update}
                />
              </label>
            </>
          ) : (
            <>
              <label className={styles.field}>
                <span>Nom de l’établissement</span>
                <input
                  className={styles.input}
                  name="establishment_name"
                  maxLength={200}
                  autoComplete="organization"
                  value={values.establishment_name}
                  onChange={update}
                  aria-invalid={invalidName === 'establishment_name'}
                  required
                />
              </label>
              <label className={styles.field}>
                <span>Catégorie</span>
                <select
                  className={styles.select}
                  name="establishment_category"
                  value={values.establishment_category}
                  onChange={update}
                  aria-invalid={invalidName === 'establishment_category'}
                  required
                >
                  <option value="">Choisir</option>
                  {CATEGORIES.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          <label className={styles.field}>
            <span>Rue et numéro</span>
            <input
              className={styles.input}
              name="street"
              maxLength={255}
              autoComplete="street-address"
              value={values.street}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Code postal</span>
            <input
              className={styles.input}
              name="postal_code"
              inputMode="numeric"
              autoComplete="postal-code"
              maxLength={4}
              value={values.postal_code}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Localité</span>
            <input
              className={styles.input}
              name="city"
              maxLength={100}
              autoComplete="address-level2"
              value={values.city}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Prénom du représentant</span>
            <input
              className={styles.input}
              name="applicant_first_name"
              maxLength={80}
              autoComplete="given-name"
              value={values.applicant_first_name}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Nom du représentant</span>
            <input
              className={styles.input}
              name="applicant_last_name"
              maxLength={80}
              autoComplete="family-name"
              value={values.applicant_last_name}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Fonction</span>
            <input
              className={styles.input}
              name="applicant_function"
              maxLength={80}
              value={values.applicant_function}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>E-mail professionnel</span>
            <input
              className={styles.input}
              name="applicant_email"
              type="email"
              maxLength={255}
              autoComplete="email"
              inputMode="email"
              value={values.applicant_email}
              onChange={update}
              aria-invalid={invalidName === 'applicant_email'}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Téléphone professionnel</span>
            <input
              className={styles.input}
              name="applicant_phone"
              type="tel"
              maxLength={30}
              autoComplete="tel"
              inputMode="tel"
              value={values.applicant_phone}
              onChange={update}
              required
            />
          </label>
          <label className={styles.field}>
            <span>Précision utile au contrôle, facultative</span>
            <textarea
              className={styles.textarea}
              name="review_note"
              maxLength={500}
              value={values.review_note}
              onChange={update}
            />
          </label>
          <label className={styles.check}>
            <input
              name="privacy_consent"
              type="checkbox"
              checked={values.privacy_consent}
              onChange={update}
              aria-invalid={invalidName === 'privacy_consent'}
              required
            />
            <span>J’accepte que LIRIE traite ce dossier pour examiner la demande.</span>
          </label>
          <label className={styles.honeypot} aria-hidden="true">
            Site web
            <input name="website" tabIndex={-1} autoComplete="off" value={values.website} onChange={update} />
          </label>
          {error ? (
            <p className={styles.alert} id="signup-error" role="alert">
              {error}
            </p>
          ) : null}
          <p className={styles.hint}>
            Un nouvel envoi remplace le lien précédent. Conservez seulement le dernier message.
          </p>
          <button className={styles.button} type="submit" disabled={pending} aria-busy={pending}>
            {pending ? 'Envoi en cours' : 'Envoyer la demande'}
          </button>
        </form>
      </div>
    </div>
  );
}
