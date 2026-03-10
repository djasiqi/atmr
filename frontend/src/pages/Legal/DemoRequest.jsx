import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  DEMO_ORGANIZATION_TYPES,
  DEMO_USE_CASES,
  submitDemoRequest,
} from '../../services/demoRequestService';
import styles from './DemoRequest.module.css';

const ORG_TYPE_LABELS = {
  transport_company: 'Entreprise de transport',
  ems: 'EMS',
  clinic: 'Clinique',
  hospital: 'Hopital',
  curatorship: 'Curatelle / Mandataire',
  other: 'Autre',
};

const USE_CASE_LABELS = {
  planning_dispatch: 'Planification / dispatch',
  billing: 'Facturation',
  transport_tracking: 'Suivi des transports',
  multi_company_coordination: 'Coordination multi-entreprises',
  reporting: 'Pilotage / reporting',
  si_integration: 'Integration SI',
  other: 'Autre',
};

const INITIAL_DATA = {
  name: '',
  email: '',
  phone: '',
  organization: '',
  organizationType: '',
  useCase: '',
  volumeRange: '',
  integrationRequired: '',
  integrationSystem: '',
  timing: '',
  preferredSlot: '',
  preferredPeriod: '',
  comment: '',
  privacyConsent: false,
  website: '',
  formStartedAtMs: Date.now(),
};

const trackDemoEvent = (eventName, payload = {}) => {
  window.dispatchEvent(
    new CustomEvent('demo_request_event', {
      detail: { eventName, ...payload },
    })
  );
};

const DemoRequest = () => {
  const [step, setStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitState, setSubmitState] = useState('idle');
  const [data, setData] = useState(INITIAL_DATA);
  const [errors, setErrors] = useState({});

  const stepLabel = useMemo(() => `Etape ${step} / 3`, [step]);

  const updateField = (event) => {
    const { name, value, type, checked } = event.target;
    setData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const validateStep = () => {
    const nextErrors = {};
    if (step === 1) {
      if (!data.name.trim()) nextErrors.name = 'Le nom est requis.';
      if (!data.email.trim()) nextErrors.email = "L'email professionnel est requis.";
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email.trim())) {
        nextErrors.email = "Le format de l'email est invalide.";
      }
      if (!data.organization.trim()) nextErrors.organization = "L'organisation est requise.";
      if (!data.organizationType) nextErrors.organizationType = "Le type d'organisation est requis.";
    }
    if (step === 2) {
      if (!data.useCase) nextErrors.useCase = "Le cas d'usage est requis.";
      if (!data.timing) nextErrors.timing = 'Le timing du projet est requis.';
      if (!data.integrationRequired) nextErrors.integrationRequired = "Le besoin d'integration est requis.";
      if (data.integrationRequired === 'yes' && !data.integrationSystem.trim()) {
        nextErrors.integrationSystem = 'Le systeme principal est requis.';
      }
    }
    if (step === 3) {
      if (!data.preferredSlot) nextErrors.preferredSlot = 'Le creneau souhaite est requis.';
      if (!data.preferredPeriod) nextErrors.preferredPeriod = 'La plage horaire preferee est requise.';
      if (!data.privacyConsent) nextErrors.privacyConsent = 'Le consentement est obligatoire.';
    }
    return nextErrors;
  };

  const goNext = () => {
    const nextErrors = validateStep();
    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors);
      trackDemoEvent('step_error', { step, errors: Object.keys(nextErrors) });
      return;
    }
    setErrors({});
    trackDemoEvent('step_next', { step });
    setStep((prev) => Math.min(prev + 1, 3));
  };

  const goBack = () => {
    setErrors({});
    setStep((prev) => Math.max(prev - 1, 1));
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    const nextErrors = validateStep();
    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors);
      trackDemoEvent('submit_error', { reason: 'client_validation' });
      return;
    }

    setErrors({});
    setSubmitState('idle');
    setIsSubmitting(true);
    try {
      await submitDemoRequest(
        {
          name: data.name.trim(),
          email: data.email.trim(),
          phone: data.phone.trim() || undefined,
          organization: data.organization.trim(),
          organization_type: data.organizationType,
          use_case: data.useCase,
          volume_range: data.volumeRange || undefined,
          integration_required: data.integrationRequired,
          integration_system:
            data.integrationRequired === 'yes' ? data.integrationSystem.trim() || undefined : undefined,
          timing: data.timing,
          preferred_slot: data.preferredSlot,
          preferred_period: data.preferredPeriod,
          comment: data.comment.trim() || undefined,
          privacy_consent: data.privacyConsent,
          honeypot: data.website,
          form_started_at_ms: data.formStartedAtMs,
        },
        { publicRequest: true }
      );
      setSubmitState('success');
      setData({ ...INITIAL_DATA, formStartedAtMs: Date.now() });
      setStep(1);
      trackDemoEvent('submit_success');
    } catch (error) {
      setSubmitState('error');
      trackDemoEvent('submit_error', { reason: error?.response?.status || 'network' });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <article className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <span className={styles.badge}>Demonstration</span>
          <h1>Demande de demonstration</h1>
          <p>Quelques informations pour preparer un echange utile, structure et adapte a votre contexte.</p>
        </div>
      </section>

      <section className={styles.container}>
        <div className={styles.stepHeader}>
          <span>{stepLabel}</span>
          <div className={styles.progress}>
            <div className={styles.progressBar} style={{ width: `${(step / 3) * 100}%` }} />
          </div>
        </div>

        <form className={styles.formCard} onSubmit={onSubmit} noValidate>
          {step === 1 && (
            <>
              <h2>Identification</h2>
              <p className={styles.stepIntro}>
                Ces informations nous permettent d&apos;adapter la demonstration a votre organisation.
              </p>
              <div className={styles.grid}>
                <label className={styles.field}>
                  <span>Nom et prenom *</span>
                  <input name="name" value={data.name} onChange={updateField} />
                  {errors.name && <small>{errors.name}</small>}
                </label>
                <label className={styles.field}>
                  <span>Email professionnel *</span>
                  <input type="email" name="email" value={data.email} onChange={updateField} />
                  {errors.email && <small>{errors.email}</small>}
                </label>
                <label className={styles.field}>
                  <span>Telephone</span>
                  <input name="phone" value={data.phone} onChange={updateField} />
                </label>
                <label className={styles.field}>
                  <span>Organisation / Institution *</span>
                  <input name="organization" value={data.organization} onChange={updateField} />
                  {errors.organization && <small>{errors.organization}</small>}
                </label>
              </div>
              <div className={styles.radioGroup}>
                <p>Type d&apos;organisation *</p>
                <div className={styles.radioGrid}>
                  {DEMO_ORGANIZATION_TYPES.map((value) => (
                    <label key={value} className={styles.radioItem}>
                      <input
                        type="radio"
                        name="organizationType"
                        value={value}
                        checked={data.organizationType === value}
                        onChange={updateField}
                      />
                      <span>{ORG_TYPE_LABELS[value]}</span>
                    </label>
                  ))}
                </div>
                {errors.organizationType && <small>{errors.organizationType}</small>}
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <h2>Votre contexte operationnel</h2>
              <div className={styles.grid}>
                <label className={styles.field}>
                  <span>Cas d&apos;usage principal *</span>
                  <select name="useCase" value={data.useCase} onChange={updateField}>
                    <option value="">Selectionner</option>
                    {DEMO_USE_CASES.map((value) => (
                      <option key={value} value={value}>
                        {USE_CASE_LABELS[value]}
                      </option>
                    ))}
                  </select>
                  {errors.useCase && <small>{errors.useCase}</small>}
                </label>
                <label className={styles.field}>
                  <span>Volumes indicatifs</span>
                  <select name="volumeRange" value={data.volumeRange} onChange={updateField}>
                    <option value="">Selectionner</option>
                    <option value="1_5">1-5 utilisateurs</option>
                    <option value="5_20">5-20 utilisateurs</option>
                    <option value="20_100">20-100 utilisateurs</option>
                    <option value="100_plus">{'>'} 100 utilisateurs</option>
                  </select>
                </label>
              </div>

              <div className={styles.radioGroup}>
                <p>Besoin d&apos;integration *</p>
                <div className={styles.radioInline}>
                  {[
                    { value: 'yes', label: 'Oui' },
                    { value: 'no', label: 'Non' },
                    { value: 'evaluate', label: 'A evaluer' },
                  ].map((item) => (
                    <label key={item.value} className={styles.radioItem}>
                      <input
                        type="radio"
                        name="integrationRequired"
                        value={item.value}
                        checked={data.integrationRequired === item.value}
                        onChange={updateField}
                      />
                      <span>{item.label}</span>
                    </label>
                  ))}
                </div>
                {errors.integrationRequired && <small>{errors.integrationRequired}</small>}
              </div>

              {data.integrationRequired === 'yes' && (
                <label className={styles.field}>
                  <span>Avec quel systeme principal ? *</span>
                  <input name="integrationSystem" value={data.integrationSystem} onChange={updateField} />
                  {errors.integrationSystem && <small>{errors.integrationSystem}</small>}
                </label>
              )}

              <div className={styles.radioGroup}>
                <p>Timing du projet *</p>
                <div className={styles.radioInline}>
                  {[
                    { value: 'immediate', label: 'Immediat' },
                    { value: 'one_three_months', label: '1-3 mois' },
                    { value: 'three_plus_months', label: '> 3 mois' },
                    { value: 'exploration', label: 'Exploration uniquement' },
                  ].map((item) => (
                    <label key={item.value} className={styles.radioItem}>
                      <input type="radio" name="timing" value={item.value} checked={data.timing === item.value} onChange={updateField} />
                      <span>{item.label}</span>
                    </label>
                  ))}
                </div>
                {errors.timing && <small>{errors.timing}</small>}
              </div>
            </>
          )}

          {step === 3 && (
            <>
              <h2>Organisation de la demonstration</h2>
              <div className={styles.grid}>
                <label className={styles.field}>
                  <span>Creneau souhaite *</span>
                  <select name="preferredSlot" value={data.preferredSlot} onChange={updateField}>
                    <option value="">Selectionner</option>
                    <option value="this_week">Cette semaine</option>
                    <option value="next_week">La semaine prochaine</option>
                    <option value="to_define">A convenir</option>
                  </select>
                  {errors.preferredSlot && <small>{errors.preferredSlot}</small>}
                </label>
                <label className={styles.field}>
                  <span>Plage horaire preferee *</span>
                  <select name="preferredPeriod" value={data.preferredPeriod} onChange={updateField}>
                    <option value="">Selectionner</option>
                    <option value="morning">Matin</option>
                    <option value="afternoon">Apres-midi</option>
                    <option value="flexible">Flexible</option>
                  </select>
                  {errors.preferredPeriod && <small>{errors.preferredPeriod}</small>}
                </label>
              </div>

              <label className={styles.field}>
                <span>Commentaire libre</span>
                <textarea rows={4} name="comment" value={data.comment} onChange={updateField} />
              </label>

              <label className={styles.hiddenHoneypot} aria-hidden="true">
                <span>Site web</span>
                <input tabIndex={-1} autoComplete="off" name="website" value={data.website} onChange={updateField} />
              </label>

              <label className={styles.checkbox}>
                <input type="checkbox" name="privacyConsent" checked={data.privacyConsent} onChange={updateField} />
                <span>
                  J&apos;accepte que LIRIE traite mes donnees afin d&apos;organiser la demonstration, conformement a la{' '}
                  <Link to="/privacy">politique de confidentialite</Link>.
                </span>
              </label>
              {errors.privacyConsent && <small>{errors.privacyConsent}</small>}
            </>
          )}

          <div className={styles.actions}>
            {step > 1 ? (
              <button type="button" className={styles.secondary} onClick={goBack}>
                Retour
              </button>
            ) : (
              <span />
            )}
            {step < 3 ? (
              <button type="button" onClick={goNext}>
                Continuer
              </button>
            ) : (
              <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? 'Envoi en cours...' : 'Envoyer la demande'}
              </button>
            )}
          </div>
        </form>

        {submitState === 'success' && (
          <section className={styles.successCard}>
            <h3>Merci.</h3>
            <p>Un membre de l&apos;equipe LIRIE vous contacte sous 24h ouvrees pour organiser la demonstration.</p>
            <p className={styles.nextTitle}>Que se passe-t-il ensuite ?</p>
            <ol>
              <li>Analyse de votre contexte</li>
              <li>Preparation personnalisee</li>
              <li>Demonstration adaptee</li>
              <li>Proposition de deploiement</li>
            </ol>
          </section>
        )}

        {submitState === 'error' && (
          <p className={styles.error}>
            Une erreur est survenue. Merci de reessayer ou d&apos;ecrire a <a href="mailto:info@lirie.ch">info@lirie.ch</a>.
          </p>
        )}
      </section>
    </article>
  );
};

export default DemoRequest;
