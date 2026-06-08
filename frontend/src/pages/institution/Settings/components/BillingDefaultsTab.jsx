// pages/institution/Settings/components/BillingDefaultsTab.jsx
/**
 * Onglet Facturation pour les paramètres institution.
 *
 * Sections:
 * 1. Règles de facturation par défaut (intro + intent)
 * 2. Coordonnées de facturation (email, adresse, TVA)
 * 3. Conditions de paiement (délai, taux TVA, timezone)
 * 4. Délais de réponse transporteurs (timeouts opérationnels)
 *
 * Philosophie: guider les demandeurs, pas imposer. Les valeurs sont des
 * suggestions pré-remplies, modifiables au cas par cas lors de la demande.
 */

import React, { useState, useEffect } from 'react';
import { FaSave } from 'react-icons/fa';
import { toast } from 'sonner';
import { useInstitutionSettings, useUpdateInstitutionSettings } from '../../../../hooks/useInstitutionData';
import { useInstitutionMe } from '../../../../hooks/useInstitutionData';
import { isAdmin, canEditBilling } from '../../../../utils/institutionPermissions';
import ChipSelect from './ChipSelect';
import styles from '../InstitutionSettings.module.css';

const BILLING_INTENT_OPTIONS = [
  { value: 'patient', label: 'Patient — le patient est facturé directement' },
  { value: 'institution', label: 'Institution — l\'institution prend en charge' },
  { value: 'third_party', label: 'Tiers payeur — un tiers (assurance, commune, curatelle) est facturé' },
];

const BillingDefaultsTab = () => {
  const { data: meData } = useInstitutionMe();
  const { data, isLoading, isError } = useInstitutionSettings();
  const updateMutation = useUpdateInstitutionSettings();

  const role = meData?.institution_role;
  const canEdit = isAdmin(role) || canEditBilling(role);
  // Les sections détaillées de facturation (coordonnées, conditions de paiement)
  // ne sont visibles que par admin et billing.
  const showBillingDetails = isAdmin(role) || canEditBilling(role);

  const [form, setForm] = useState({
    // Institution billing
    billing_email: '',
    billing_address: '',
    vat_number: '',
    // Settings
    default_billing_intent: 'patient',
    default_vat_rate: '',
    default_payment_terms_days: 30,
    timeout_same_day_minutes: 5,
    timeout_default_minutes: 60,
    timezone: 'Europe/Zurich',
    // Transport UX
    default_pickup_mode: 'institution',
    entry_points: [],
    default_contact_phone: '',
  });

  // Entry point input for adding new items
  const [newEntryPoint, setNewEntryPoint] = useState('');

  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (data && !loaded) {
      const inst = data.institution || {};
      const settings = data.settings || {};
      setForm({
        billing_email: inst.billing_email || '',
        billing_address: inst.billing_address || '',
        vat_number: inst.vat_number || '',
        default_billing_intent: settings.default_billing_intent || 'patient',
        default_vat_rate: settings.default_vat_rate != null ? String(settings.default_vat_rate) : '',
        default_payment_terms_days: settings.default_payment_terms_days ?? 30,
        timeout_same_day_minutes: settings.timeout_same_day_minutes ?? 5,
        timeout_default_minutes: settings.timeout_default_minutes ?? 60,
        timezone: settings.timezone || 'Europe/Zurich',
        default_pickup_mode: settings.default_pickup_mode || 'institution',
        entry_points: settings.entry_points || [],
        default_contact_phone: settings.default_contact_phone || '',
      });
      setLoaded(true);
    }
  }, [data, loaded]);

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  // Dirty state: compare form vs server data
  const isDirty = (() => {
    if (!data) return false;
    const inst = data.institution || {};
    const settings = data.settings || {};
    return (
      form.billing_email !== (inst.billing_email || '') ||
      form.billing_address !== (inst.billing_address || '') ||
      form.vat_number !== (inst.vat_number || '') ||
      form.default_billing_intent !== (settings.default_billing_intent || 'patient') ||
      form.default_vat_rate !== (settings.default_vat_rate != null ? String(settings.default_vat_rate) : '') ||
      String(form.default_payment_terms_days) !== String(settings.default_payment_terms_days ?? 30) ||
      String(form.timeout_same_day_minutes) !== String(settings.timeout_same_day_minutes ?? 5) ||
      String(form.timeout_default_minutes) !== String(settings.timeout_default_minutes ?? 60) ||
      form.timezone !== (settings.timezone || 'Europe/Zurich') ||
      form.default_pickup_mode !== (settings.default_pickup_mode || 'institution') ||
      JSON.stringify(form.entry_points) !== JSON.stringify(settings.entry_points || []) ||
      form.default_contact_phone !== (settings.default_contact_phone || '')
    );
  })();

  const handleSave = async () => {
    // Build payload with only changed fields
    const payload = {};
    const inst = data?.institution || {};
    const settings = data?.settings || {};

    // Institution billing fields
    if (form.billing_email !== (inst.billing_email || ''))
      payload.billing_email = form.billing_email || null;
    if (form.billing_address !== (inst.billing_address || ''))
      payload.billing_address = form.billing_address || null;
    if (form.vat_number !== (inst.vat_number || ''))
      payload.vat_number = form.vat_number || null;

    // Settings fields
    if (form.default_billing_intent !== (settings.default_billing_intent || 'patient'))
      payload.default_billing_intent = form.default_billing_intent;
    
    const vatRate = form.default_vat_rate !== '' ? parseFloat(form.default_vat_rate) : null;
    if (vatRate !== settings.default_vat_rate)
      payload.default_vat_rate = vatRate;

    const paymentDays = parseInt(form.default_payment_terms_days, 10);
    if (!isNaN(paymentDays) && paymentDays !== settings.default_payment_terms_days)
      payload.default_payment_terms_days = paymentDays;

    const sameDayTimeout = parseInt(form.timeout_same_day_minutes, 10);
    if (!isNaN(sameDayTimeout) && sameDayTimeout !== settings.timeout_same_day_minutes)
      payload.timeout_same_day_minutes = sameDayTimeout;

    const defaultTimeout = parseInt(form.timeout_default_minutes, 10);
    if (!isNaN(defaultTimeout) && defaultTimeout !== settings.timeout_default_minutes)
      payload.timeout_default_minutes = defaultTimeout;

    if (form.timezone !== (settings.timezone || 'Europe/Zurich'))
      payload.timezone = form.timezone;

    // Transport UX fields
    if (form.default_pickup_mode !== (settings.default_pickup_mode || 'institution'))
      payload.default_pickup_mode = form.default_pickup_mode;
    if (JSON.stringify(form.entry_points) !== JSON.stringify(settings.entry_points || []))
      payload.entry_points = form.entry_points;
    if (form.default_contact_phone !== (settings.default_contact_phone || ''))
      payload.default_contact_phone = form.default_contact_phone || null;

    if (Object.keys(payload).length === 0) {
      toast.info('Aucune modification');
      return;
    }

    try {
      const result = await updateMutation.mutateAsync(payload);
      // Sync form with server response
      if (result) {
        const newInst = result.institution || {};
        const newSettings = result.settings || {};
        setForm({
          billing_email: newInst.billing_email || '',
          billing_address: newInst.billing_address || '',
          vat_number: newInst.vat_number || '',
          default_billing_intent: newSettings.default_billing_intent || 'patient',
          default_vat_rate: newSettings.default_vat_rate != null ? String(newSettings.default_vat_rate) : '',
          default_payment_terms_days: newSettings.default_payment_terms_days ?? 30,
          timeout_same_day_minutes: newSettings.timeout_same_day_minutes ?? 5,
          timeout_default_minutes: newSettings.timeout_default_minutes ?? 60,
          timezone: newSettings.timezone || 'Europe/Zurich',
          default_pickup_mode: newSettings.default_pickup_mode || 'institution',
          entry_points: newSettings.entry_points || [],
          default_contact_phone: newSettings.default_contact_phone || '',
        });
      }
      toast.success('Paramètres enregistrés');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la sauvegarde');
    }
  };

  if (isLoading) {
    return (
      <div className={styles.section}>
        <p>Chargement des paramètres...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className={styles.section}>
        <p style={{ color: '#c62828' }}>Erreur lors du chargement des paramètres.</p>
      </div>
    );
  }

  return (
    <div className={styles.section}>
      {/* ── Section 1 : Introduction + Règle par défaut ── */}
      <div className={styles.sectionHeader}>
        <h3>Règles de facturation</h3>
        <p style={{ color: '#666', fontSize: 13, lineHeight: 1.6 }}>
          Ces paramètres permettent de proposer automatiquement le bon mode de facturation
          lors de la création d'une demande de transport.
          Ils peuvent être ajustés au cas par cas par les utilisateurs autorisés.
        </p>
      </div>

      <div className={styles.profileForm}>
        <div className={styles.field}>
          <label htmlFor="default-billing-intent">Facturé à (par défaut)</label>
          <ChipSelect
            id="default-billing-intent"
            name="default_billing_intent"
            ariaLabel="Facturé à (par défaut)"
            block
            value={form.default_billing_intent}
            options={BILLING_INTENT_OPTIONS}
            onChange={(val) => handleChange('default_billing_intent', val)}
            disabled={!canEdit}
          />
          <span className={styles.fieldHint}>
            Utilisé lorsqu'aucune règle spécifique ne s'applique.
            Ce choix pré-remplit le champ &laquo;&nbsp;Facturé à&nbsp;&raquo; lors de chaque nouvelle demande.
          </span>
        </div>
      </div>

      {/* Bloc informatif : règles par type de transport (futur) */}
      <div style={{
        background: '#f8f9fa',
        border: '1px solid #e0e0e0',
        borderRadius: 8,
        padding: '14px 18px',
        marginTop: 16,
        marginBottom: 8,
        fontSize: 13,
        color: '#555',
        lineHeight: 1.7,
      }}>
        <strong style={{ color: '#333', fontSize: 14 }}>Règles par type de transport</strong>
        <p style={{ margin: '8px 0 6px' }}>
          Le mode de facturation peut varier selon le contexte du transport :
        </p>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #ddd', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px', fontWeight: 600, color: '#333' }}>Type de transport</th>
              <th style={{ padding: '6px 8px', fontWeight: 600, color: '#333' }}>Facturation habituelle</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '6px 8px' }}>Transport médical lié au traitement</td>
              <td style={{ padding: '6px 8px', color: '#2e7d32' }}>Institution</td>
            </tr>
            <tr style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '6px 8px' }}>Transport médical hors traitement</td>
              <td style={{ padding: '6px 8px', color: '#e65100' }}>Patient</td>
            </tr>
            <tr>
              <td style={{ padding: '6px 8px' }}>Transport non médical (privé)</td>
              <td style={{ padding: '6px 8px', color: '#e65100' }}>Patient</td>
            </tr>
          </tbody>
        </table>
        <p style={{ margin: '8px 0 0', fontSize: 12, color: '#888', fontStyle: 'italic' }}>
          Ce tableau est donné à titre indicatif. Le demandeur peut toujours adapter
          le mode de facturation lors de la création de chaque demande.
        </p>
      </div>

      {/* ── Section 2 : Coordonnées de facturation (admin + billing uniquement) ── */}
      {showBillingDetails && (
        <>
          <div className={styles.sectionHeader} style={{ marginTop: 32 }}>
            <h3>Coordonnées de facturation</h3>
            <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
              Ces informations sont utilisées lorsque l'institution est sélectionnée comme
              destinataire de la facture. Elles apparaissent sur les documents de facturation.
            </p>
          </div>

          <div className={styles.profileForm}>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Email de facturation</label>
                <input
                  type="email"
                  value={form.billing_email}
                  onChange={(e) => handleChange('billing_email', e.target.value)}
                  disabled={!canEdit}
                  placeholder="facturation@clinique.ch"
                />
                <span className={styles.fieldHint}>
                  Adresse à laquelle les factures de transport sont envoyées.
                  Distinct de l'email de contact général.
                </span>
              </div>
              <div className={styles.field}>
                <label>Numéro IDE / TVA</label>
                <input
                  type="text"
                  value={form.vat_number}
                  onChange={(e) => handleChange('vat_number', e.target.value)}
                  disabled={!canEdit}
                  placeholder="CHE-123.456.789"
                />
                <span className={styles.fieldHint}>
                  Numéro d'identification des entreprises, si applicable.
                </span>
              </div>
            </div>

            <div className={styles.field}>
              <label>Adresse de facturation</label>
              <textarea
                rows={3}
                value={form.billing_address}
                onChange={(e) => handleChange('billing_address', e.target.value)}
                disabled={!canEdit}
                placeholder="Chemin des Courbes 9, 1247 Anières"
              />
              <span className={styles.fieldHint}>
                Adresse complète telle qu'elle doit figurer sur les factures.
                Si vide, l'adresse du profil de l'institution est utilisée.
              </span>
            </div>
          </div>

          {/* ── Section 3 : Conditions de paiement (admin + billing uniquement) ── */}
          <div className={styles.sectionHeader} style={{ marginTop: 32 }}>
            <h3>Conditions de paiement</h3>
            <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
              Paramètres appliqués par défaut aux factures émises pour cette institution.
            </p>
          </div>

          <div className={styles.profileForm}>
            <div className={styles.fieldRow}>
              <div className={styles.field}>
                <label>Taux TVA par défaut (%)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="100"
                  value={form.default_vat_rate}
                  onChange={(e) => handleChange('default_vat_rate', e.target.value)}
                  disabled={!canEdit}
                  placeholder="7.7"
                />
                <span className={styles.fieldHint}>
                  Taux de TVA appliqué par défaut. En Suisse : 8.1% (normal), 2.6% (réduit),
                  ou 0% (exempté). Laissez vide si non applicable.
                </span>
              </div>
              <div className={styles.field}>
                <label>Délai de paiement (jours)</label>
                <input
                  type="number"
                  min="0"
                  max="365"
                  value={form.default_payment_terms_days}
                  onChange={(e) => handleChange('default_payment_terms_days', e.target.value)}
                  disabled={!canEdit}
                />
                <span className={styles.fieldHint}>
                  Nombre de jours accordés pour le règlement de la facture.
                  Standard : 30 jours.
                </span>
              </div>
            </div>

          </div>
        </>
      )}

      {/* ── Section 4 : Délais de réponse ── */}
      <div className={styles.sectionHeader} style={{ marginTop: 32 }}>
        <h3>Délais de réponse</h3>
        <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
          Temps accordé à un transporteur pour accepter une demande
          avant qu'elle ne soit transmise au suivant (escalade automatique).
        </p>
      </div>

      <div className={styles.profileForm}>
        <div className={styles.fieldRow}>
          <div className={styles.field}>
            <label>Demandes le jour même (minutes)</label>
            <input
              type="number"
              min="1"
              max="240"
              value={form.timeout_same_day_minutes}
              onChange={(e) => handleChange('timeout_same_day_minutes', e.target.value)}
              disabled={!canEdit}
            />
            <span className={styles.fieldHint}>
              Délai pour les transports urgents prévus le jour même (1 à 240 min).
              Recommandé : 5 min.
            </span>
          </div>
          <div className={styles.field}>
            <label>Demandes planifiées (minutes)</label>
            <input
              type="number"
              min="1"
              max="10080"
              value={form.timeout_default_minutes}
              onChange={(e) => handleChange('timeout_default_minutes', e.target.value)}
              disabled={!canEdit}
            />
            <span className={styles.fieldHint}>
              Délai pour les transports planifiés à l'avance (1 à 10 080 min).
              Recommandé : 60 min.
            </span>
          </div>
        </div>
      </div>

      {/* ── Section 5 : Paramètres de demande (transport UX) ── */}
      <div className={styles.sectionHeader} style={{ marginTop: 32 }}>
        <h3>Paramètres de demande</h3>
        <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
          Configurent le pré-remplissage du formulaire de demande de transport.
        </p>
      </div>

      <div className={styles.profileForm}>
        {/* Mode de départ par défaut */}
        <div className={styles.field}>
          <label>Lieu de départ par défaut</label>
          <div style={{ display: 'flex', gap: 16, marginTop: 4 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, cursor: canEdit ? 'pointer' : 'default' }}>
              <input
                type="radio"
                name="default_pickup_mode_edit"
                value="institution"
                checked={form.default_pickup_mode === 'institution'}
                disabled={!canEdit}
                onChange={() => handleChange('default_pickup_mode', 'institution')}
              />
              Institution (clinique / EMS)
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, cursor: canEdit ? 'pointer' : 'default' }}>
              <input
                type="radio"
                name="default_pickup_mode_edit"
                value="domicile"
                checked={form.default_pickup_mode === 'domicile'}
                disabled={!canEdit}
                onChange={() => handleChange('default_pickup_mode', 'domicile')}
              />
              Domicile du patient (IMAD)
            </label>
          </div>
          <span className={styles.fieldHint}>
            Détermine le type de trajet pré-sélectionné dans le formulaire de demande.
          </span>
        </div>

        {/* Points d'accueil */}
        <div className={styles.field}>
          <label>Points d'accueil / Entrées</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            {(form.entry_points || []).map((ep, i) => (
              <span key={i} style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                padding: '3px 10px',
                background: '#e3f2fd',
                color: '#1565c0',
                borderRadius: 12,
                fontSize: 12,
                fontWeight: 500,
              }}>
                {ep}
                {canEdit && (
                  <button
                    type="button"
                    onClick={() => {
                      const updated = form.entry_points.filter((_, idx) => idx !== i);
                      handleChange('entry_points', updated);
                    }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#1565c0', padding: 0, fontSize: 14, lineHeight: 1 }}
                    title="Retirer"
                  >
                    ×
                  </button>
                )}
              </span>
            ))}
          </div>
          {canEdit && (form.entry_points || []).length < 20 && (
            <div style={{ display: 'flex', gap: 6 }}>
              <input
                type="text"
                value={newEntryPoint}
                onChange={(e) => setNewEntryPoint(e.target.value)}
                placeholder="Ex: Réception, Urgences, Entrée ambulances"
                style={{ flex: 1, maxWidth: 300 }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    const trimmed = newEntryPoint.trim();
                    if (trimmed && !(form.entry_points || []).includes(trimmed)) {
                      handleChange('entry_points', [...(form.entry_points || []), trimmed]);
                      setNewEntryPoint('');
                    }
                  }
                }}
              />
              <button
                type="button"
                className={styles.addKeyBtn}
                style={{ width: 'auto', marginBottom: 0 }}
                onClick={() => {
                  const trimmed = newEntryPoint.trim();
                  if (trimmed && !(form.entry_points || []).includes(trimmed)) {
                    handleChange('entry_points', [...(form.entry_points || []), trimmed]);
                    setNewEntryPoint('');
                  }
                }}
              >
                Ajouter
              </button>
            </div>
          )}
          <span className={styles.fieldHint}>
            Ces suggestions apparaîtront dans le formulaire de demande pour faciliter la saisie.
            Appuyez sur Entrée ou cliquez &laquo;&nbsp;Ajouter&nbsp;&raquo; pour chaque point d'accueil.
          </span>
        </div>

        {/* Téléphone standard */}
        <div className={styles.field}>
          <label>Téléphone standard institution</label>
          <input
            type="tel"
            value={form.default_contact_phone}
            onChange={(e) => handleChange('default_contact_phone', e.target.value)}
            disabled={!canEdit}
            placeholder="+41 22 123 45 67"
            style={{ maxWidth: 280 }}
          />
          <span className={styles.fieldHint}>
            Pré-rempli comme contact sur place dans les demandes de transport.
          </span>
        </div>
      </div>

      {/* Save */}
      {canEdit && (
        <button
          className={styles.saveBtn}
          onClick={handleSave}
          disabled={updateMutation.isPending || !isDirty}
          style={{ marginTop: 24 }}
        >
          <FaSave /> {updateMutation.isPending ? 'Enregistrement...' : 'Enregistrer'}
        </button>
      )}

      {/* Note droits — pour demandeur / lecteur */}
      {!canEdit && (
        <div style={{
          marginTop: 24,
          padding: '12px 16px',
          background: '#f8f9fa',
          border: '1px solid #e0e0e0',
          borderRadius: 8,
          fontSize: 13,
          color: '#666',
          lineHeight: 1.6,
        }}>
          <strong style={{ color: '#333' }}>Consultation uniquement</strong>
          <br />
          Ces paramètres sont affichés à titre informatif. Seuls les administrateurs
          et les responsables facturation peuvent les modifier.
          {!showBillingDetails && (
            <span style={{ display: 'block', marginTop: 6, fontSize: 12, color: '#999' }}>
              Les coordonnées de facturation et les conditions de paiement ne sont
              accessibles qu'aux administrateurs et responsables facturation.
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default BillingDefaultsTab;
