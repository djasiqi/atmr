/**
 * Champs formulaire transporteur externe (création + détail).
 */

import React from 'react';
import s from './ExternalCarrierFields.module.css';

const EMPTY_FORM = {
  name: '',
  phone: '',
  email: '',
  reason: '',
};

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export const EMPTY_EXTERNAL_CARRIER_FORM = { ...EMPTY_FORM };

export function validateExternalCarrierForm(form) {
  const name = (form?.name || '').trim();
  if (!name) {
    return 'Le nom du transporteur externe est requis.';
  }
  if (name.length > 255) {
    return 'Le nom ne peut pas dépasser 255 caractères.';
  }
  const email = (form?.email || '').trim();
  if (email && !EMAIL_RE.test(email)) {
    return 'L\'adresse email du transporteur externe est invalide.';
  }
  if (email.length > 255) {
    return 'L\'email ne peut pas dépasser 255 caractères.';
  }
  const reason = (form?.reason || '').trim();
  if (reason.length > 120) {
    return 'La raison ne peut pas dépasser 120 caractères.';
  }
  return null;
}

export function buildExternalCarrierPayload(form) {
  const payload = {
    name: (form?.name || '').trim(),
  };
  const phone = (form?.phone || '').trim();
  const email = (form?.email || '').trim();
  const reason = (form?.reason || '').trim();
  if (phone) payload.phone = phone;
  if (email) payload.email = email;
  if (reason) payload.reason = reason;
  return payload;
}

export default function ExternalCarrierFields({
  value,
  onChange,
  showNotice = true,
  idPrefix = 'external-carrier',
}) {
  const form = value || EMPTY_FORM;

  const handleChange = (field) => (e) => {
    onChange({ ...form, [field]: e.target.value });
  };

  return (
    <div className={s.block}>
      <div className={s.header}>
        <h4 className={s.title}>Transporteur externe</h4>
        {showNotice && (
          <span className={s.notice} title="Ce transporteur n'est pas inscrit sur LIRIE. Aucune offre ni réservation LIRIE ne sera créée. La mission pourra être déclarée réalisée manuellement.">
            Non LIRIE — aucune offre ni réservation, déclaration manuelle
          </span>
        )}
      </div>
      <div className={s.grid}>
        <label className={s.field} htmlFor={`${idPrefix}-name`}>
          <span>Nom *</span>
          <input
            id={`${idPrefix}-name`}
            type="text"
            value={form.name}
            onChange={handleChange('name')}
            maxLength={255}
            required
            placeholder="Ex. Taxi Urgence SA"
          />
        </label>
        <label className={s.field} htmlFor={`${idPrefix}-phone`}>
          <span>Téléphone</span>
          <input
            id={`${idPrefix}-phone`}
            type="tel"
            value={form.phone}
            onChange={handleChange('phone')}
            maxLength={50}
            placeholder="+41 79 000 00 00"
          />
        </label>
        <label className={s.field} htmlFor={`${idPrefix}-email`}>
          <span>Email</span>
          <input
            id={`${idPrefix}-email`}
            type="email"
            value={form.email}
            onChange={handleChange('email')}
            maxLength={255}
            placeholder="ops@transporteur.ch"
          />
        </label>
        <label className={s.field} htmlFor={`${idPrefix}-reason`}>
          <span>Raison</span>
          <input
            id={`${idPrefix}-reason`}
            type="text"
            value={form.reason}
            onChange={handleChange('reason')}
            maxLength={120}
            placeholder="Ex. Dépannage"
          />
        </label>
      </div>
    </div>
  );
}
