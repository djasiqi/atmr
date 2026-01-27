import React, { useEffect, useMemo, useRef, useState } from 'react';
import apiClient from '../../../../../utils/apiClient';
import styles from '../../CompanySettings.module.css';
import {
  fetchBillingParties,
  fetchClinicBillingMappings,
  upsertClinicBillingMapping,
} from '../../../../../services/settingsService';
import useUrlSearchSync from '../../../../../hooks/useUrlSearchSync';

export default function ClinicBillingMappingsSection() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const clinicInputRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();

  const [mappings, setMappings] = useState([]);
  const [billingParties, setBillingParties] = useState([]);

  // Form: clinique recherchée + billing_party sélectionné
  const [clinicQuery, setClinicQuery] = useState('');
  const [clinicResults, setClinicResults] = useState([]);
  const [selectedClinic, setSelectedClinic] = useState(null);
  const [selectedBillingPartyId, setSelectedBillingPartyId] = useState('');

  const selectedClinicId = selectedClinic?.id ? Number(selectedClinic.id) : null;

  const existingMappingForSelectedClinic = useMemo(() => {
    if (!selectedClinicId) return null;
    return mappings.find((m) => Number(m.clinic_company_id) === selectedClinicId) || null;
  }, [mappings, selectedClinicId]);

  useEffect(() => {
    const run = async () => {
      try {
        setLoading(true);
        setError('');
        const [m, bp] = await Promise.all([fetchClinicBillingMappings(), fetchBillingParties()]);
        setMappings(Array.isArray(m?.data) ? m.data : Array.isArray(m?.data?.data) ? m.data.data : m?.data || []);
        setBillingParties(Array.isArray(bp?.data) ? bp.data : bp?.data?.data || []);
      } catch (e) {
        console.error('[ClinicBillingMappingsSection] load failed:', e);
        setError(e?.response?.data?.error || e?.message || 'Erreur lors du chargement');
      } finally {
        setLoading(false);
      }
    };
    run();
  }, []);

  useEffect(() => {
    if (!existingMappingForSelectedClinic) return;
    setSelectedBillingPartyId(String(existingMappingForSelectedClinic.billing_party_id || ''));
  }, [existingMappingForSelectedClinic]);

  useEffect(() => {
    if (!initialized) return;
    if (initialSearch && initialSearch !== clinicQuery) {
      setClinicQuery(initialSearch);
      setSelectedClinic(null);
      searchClinics(initialSearch);
    }
    if (shouldFocus) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      requestAnimationFrame(() => {
        clinicInputRef.current?.focus();
      });
      consumeFocus();
    }
  }, [initialized, initialSearch, shouldFocus, consumeFocus, clinicQuery]);

  const searchClinics = async (q) => {
    const query = (q || '').trim();
    if (query.length < 2) {
      setClinicResults([]);
      return;
    }
    try {
      const { data } = await apiClient.get('/companies/search', {
        params: { q: query },
      });
      const list = Array.isArray(data?.data) ? data.data : Array.isArray(data) ? data : [];
      setClinicResults(list);
    } catch (e) {
      console.error('[ClinicBillingMappingsSection] search failed:', e);
      setClinicResults([]);
    }
  };

  const onSelectClinic = (c) => {
    setSelectedClinic(c);
    setClinicQuery(c?.name || '');
    setClinicResults([]);
    setMessage('');
    setError('');
  };

  const onSave = async () => {
    try {
      setMessage('');
      setError('');
      if (!selectedClinicId) {
        setError('Veuillez sélectionner une clinique.');
        return;
      }
      const bpId = Number(selectedBillingPartyId);
      if (!bpId) {
        setError('Veuillez sélectionner un destinataire (BillingParty).');
        return;
      }
      await upsertClinicBillingMapping({
        clinic_company_id: selectedClinicId,
        billing_party_id: bpId,
        is_active: true,
      });
      // Reload mappings
      const m = await fetchClinicBillingMappings();
      setMappings(Array.isArray(m?.data) ? m.data : m?.data?.data || []);
      setMessage('✅ Mapping enregistré');
    } catch (e) {
      console.error('[ClinicBillingMappingsSection] save failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors de la sauvegarde');
    }
  };

  return (
    <div className={styles.section}>
      <h3>Facturation cliniques (P1)</h3>
      <p className={styles.mutedText}>
        Configure le destinataire de facturation (BillingParty) à utiliser quand une course est
        facturée à une clinique (S2).
      </p>

      {loading ? <p>Chargement…</p> : null}
      {error ? <p className={styles.error}>{error}</p> : null}
      {message ? <p className={styles.success}>{message}</p> : null}

      <div className={styles.formGroup}>
        <label>Clinique</label>
        <input
          className={styles.input}
          value={clinicQuery}
          onChange={(e) => {
            const v = e.target.value;
            setClinicQuery(v);
            setSelectedClinic(null);
            searchClinics(v);
          }}
          placeholder="Rechercher une entreprise (min 2 caractères)…"
          ref={clinicInputRef}
        />
        {clinicResults.length > 0 ? (
          <div className={styles.dropdownList}>
            {clinicResults.map((c) => (
              <button
                key={c.id}
                type="button"
                className={styles.dropdownItem}
                onClick={() => onSelectClinic(c)}
              >
                <strong>{c.name}</strong> <span className={styles.mutedText}>#{c.id}</span>
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div className={styles.formGroup}>
        <label>Destinataire (BillingParty)</label>
        <select
          className={styles.input}
          value={selectedBillingPartyId}
          onChange={(e) => setSelectedBillingPartyId(e.target.value)}
        >
          <option value="">— Sélectionner —</option>
          {billingParties.map((bp) => (
            <option key={bp.id} value={bp.id}>
              {bp.display_name} ({bp.type})
            </option>
          ))}
        </select>
        {existingMappingForSelectedClinic ? (
          <p className={styles.mutedText}>
            Mapping actuel: <strong>{existingMappingForSelectedClinic.billing_party_name}</strong>
          </p>
        ) : null}
      </div>

      <div className={styles.actionsRow}>
        <button type="button" className={styles.submitButton} onClick={onSave}>
          Enregistrer le mapping
        </button>
      </div>

      <hr className={styles.separator} />

      <h4>Mappings existants</h4>
      {mappings.length === 0 ? (
        <p className={styles.mutedText}>Aucun mapping configuré.</p>
      ) : (
        <div className={styles.tableContainer}>
          <table className={styles.activityTable}>
            <thead>
              <tr>
                <th>Clinique</th>
                <th>Destinataire</th>
                <th>Actif</th>
              </tr>
            </thead>
            <tbody>
              {mappings.map((m) => (
                <tr key={m.id}>
                  <td>
                    {m.clinic_company_name || '—'}{' '}
                    <span className={styles.mutedText}>#{m.clinic_company_id}</span>
                  </td>
                  <td>
                    {m.billing_party_name || '—'}{' '}
                    <span className={styles.mutedText}>#{m.billing_party_id}</span>
                  </td>
                  <td>{m.is_active ? 'Oui' : 'Non'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

