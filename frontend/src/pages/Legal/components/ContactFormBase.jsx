import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { submitContactRequest } from '../../../services/contactService';
import { submitDemoRequest } from '../../../services/demoRequestService';
import SuccessCard from './SuccessCard';
import styles from '../ContactSubpages.module.css';

const buildInitialForm = (config) => {
  const base = {
    website: '',
    privacy_consent: false,
  };
  config.fields.forEach((field) => {
    if (field.type === 'select') {
      base[field.name] = '';
    } else {
      base[field.name] = '';
    }
  });
  return base;
};

const ContactFormBase = ({ category, config }) => {
  const initialForm = useMemo(() => buildInitialForm(config), [config]);
  const [form, setForm] = useState(initialForm);
  const [formStartedAtMs, setFormStartedAtMs] = useState(() => Date.now());
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const [traceId, setTraceId] = useState('');

  const mapDemoOrganizationType = (value) => {
    if (value === 'transport') return 'transport_company';
    if (value === 'institution') return 'institution';
    if (value === 'curatorship') return 'curatorship';
    return 'other';
  };

  const mapDemoUseCase = (value) => {
    if (value === 'transport') return 'planning_dispatch';
    if (value === 'institution') return 'reporting';
    if (value === 'curatorship') return 'multi_company_coordination';
    return 'other';
  };

  const mapPreferredSlot = (value) => {
    if (value === 'to_schedule') return 'to_define';
    return value || '';
  };

  const onChange = (event) => {
    const { name, type, checked, value } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const validate = () => {
    const nextErrors = {};
    config.fields.forEach((field) => {
      const value = String(form[field.name] || '').trim();
      if (field.required && !value) {
        nextErrors[field.name] = 'Ce champ est requis.';
      }
      if (field.type === 'email' && value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
        nextErrors[field.name] = "Format d'email invalide.";
      }
    });
    if (!form.privacy_consent) {
      nextErrors.privacy_consent = 'Le consentement est requis.';
    }
    return nextErrors;
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setSubmitError('');
    const nextErrors = validate();
    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors);
      return;
    }
    setErrors({});
    setIsSubmitting(true);
    try {
      const payload = {
        category,
        website: form.website || '',
        privacy_consent: Boolean(form.privacy_consent),
      };
      config.fields.forEach((field) => {
        const raw = form[field.name];
        if (raw === undefined || raw === null) {
          return;
        }
        const value = typeof raw === 'string' ? raw.trim() : raw;
        if (value !== '') {
          payload[field.name] = value;
        }
      });
      let response;
      if (category === 'demo') {
        // Flux demo-request unique : une seule soumission, un seul email, une seule entrée admin.
        response = await submitDemoRequest(
          {
            name: payload.name,
            email: payload.email,
            phone: payload.phone || null,
            organization: payload.organization,
            organization_type: mapDemoOrganizationType(payload.organization_type),
            use_case: mapDemoUseCase(payload.organization_type),
            volume_range: payload.volume_range || null,
            integration_required: 'evaluate',
            integration_system: null,
            timing: payload.timing,
            preferred_slot: mapPreferredSlot(payload.preferred_slot),
            preferred_period: 'flexible',
            comment: payload.message,
            privacy_consent: Boolean(payload.privacy_consent),
            honeypot: payload.website || '',
            form_started_at_ms: formStartedAtMs,
            acknowledgement_already_sent: false,
            source: 'web_contact_demo',
          },
          { publicRequest: true }
        );
      } else {
        response = await submitContactRequest(payload);
      }
      setTraceId(response?.trace_id || '');
      setForm(initialForm);
      setFormStartedAtMs(Date.now());
    } catch (error) {
      const status = error?.response?.status;
      const apiMessage = error?.response?.data?.message || error?.response?.data?.error;
      if (status === 429) {
        setSubmitError(apiMessage || 'Trop de requetes. Merci de patienter puis de reessayer.');
      } else if (apiMessage) {
        setSubmitError(apiMessage);
      } else {
        setSubmitError("Une erreur est survenue. Merci de reessayer ou d'ecrire a info@lirie.ch.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  if (traceId) {
    return <SuccessCard traceId={traceId} />;
  }

  return (
    <form className={styles.formGrid} noValidate onSubmit={onSubmit}>
      {config.fields.map((field) => {
        const value = form[field.name] || '';
        const error = errors[field.name];
        const className = field.type === 'textarea' ? `${styles.field} ${styles.full}` : styles.field;

        if (field.type === 'select') {
          return (
            <label key={field.name} className={className}>
              <span>{field.label}</span>
              <select name={field.name} value={value} onChange={onChange} aria-invalid={Boolean(error)}>
                <option value="">{field.placeholder || 'Selectionner'}</option>
                {field.options?.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {error && <small>{error}</small>}
            </label>
          );
        }

        if (field.type === 'textarea') {
          return (
            <label key={field.name} className={className}>
              <span>{field.label}</span>
              <textarea
                name={field.name}
                rows={5}
                value={value}
                onChange={onChange}
                placeholder={field.placeholder || ''}
                aria-invalid={Boolean(error)}
              />
              {error && <small>{error}</small>}
            </label>
          );
        }

        return (
          <label key={field.name} className={className}>
            <span>{field.label}</span>
            <input
              type={field.type || 'text'}
              name={field.name}
              value={value}
              onChange={onChange}
              placeholder={field.placeholder || ''}
              aria-invalid={Boolean(error)}
            />
            {error && <small>{error}</small>}
          </label>
        );
      })}

      <label className={styles.honeypot} aria-hidden="true">
        <span>Site web</span>
        <input tabIndex={-1} autoComplete="off" name="website" value={form.website} onChange={onChange} />
      </label>

      <label className={`${styles.checkbox} ${styles.full}`}>
        <input
          type="checkbox"
          name="privacy_consent"
          checked={Boolean(form.privacy_consent)}
          onChange={onChange}
          aria-invalid={Boolean(errors.privacy_consent)}
        />
        <span>
          {config.consentText} <Link to="/privacy">Politique de confidentialite</Link>.
        </span>
      </label>
      {errors.privacy_consent && <small className={styles.full}>{errors.privacy_consent}</small>}

      <div className={`${styles.actions} ${styles.full}`}>
        <button className={styles.primaryButton} type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Envoi en cours...' : config.submitLabel || 'Envoyer'}
        </button>
      </div>

      {submitError && (
        <p className={`${styles.error} ${styles.full}`} role="alert">
          {submitError}
        </p>
      )}
    </form>
  );
};

export default ContactFormBase;
