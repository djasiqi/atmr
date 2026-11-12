import React, { useEffect, useState } from 'react';
import apiClient from '../../../../utils/apiClient';

const MEANS = [
  {
    key: 'tap_to_pay',
    label: 'Tap to Pay',
    wish: 'Souhait de proposer le paiement par carte.',
    prerequisite: 'Prérequis manquant : le prestataire n’est pas encore activé.',
  },
  {
    key: 'cash',
    label: 'Espèces',
    wish: 'Souhait d’autoriser l’encaissement en espèces.',
    prerequisite: 'Prérequis manquant : la procédure de caisse n’est pas encore active.',
  },
  {
    key: 'direct_patient_invoice',
    label: 'Facture directe au patient',
    wish: 'Souhait de proposer la facturation après prestation.',
    prerequisite:
      'Prérequis manquant : le texte de signature et les conditions commerciales ne sont pas encore approuvés.',
  },
];

const EMPTY = {
  tap_to_pay: false,
  cash: false,
  direct_patient_invoice: false,
};

export default function CompanySettlementPreferences({ companyId }) {
  const [preferences, setPreferences] = useState(EMPTY);
  const [recorded, setRecorded] = useState(EMPTY);
  const [history, setHistory] = useState([]);
  const [noticeChecked, setNoticeChecked] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.resolve(apiClient.get('/companies/me/settlement-settings'))
      .then((response) => {
        if (cancelled) return;
        const data = response?.data?.data || response?.data || {};
        const next = { ...EMPTY, ...(data.preferences || {}) };
        setPreferences(next);
        setRecorded(next);
        setHistory(Array.isArray(data.history) ? data.history : []);
        setMessage('');
      })
      .catch(() => {
        if (!cancelled) {
          setMessage('');
          setError('La rubrique serveur est inactive. Aucun paiement n’est activé.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [companyId]);

  const save = async () => {
    setError('');
    setMessage('');
    setSaving(true);
    try {
      const response = await apiClient.put('/companies/me/settlement-settings', {
        preferences,
        company_id: companyId,
      });
      const data = response?.data?.data || response?.data || {};
      const saved = data.preferences || {};
      const complete = Object.keys(EMPTY).every((key) => Object.prototype.hasOwnProperty.call(saved, key));
      if (!complete || data.activation_enabled !== false) {
        setError('Enregistrement incomplet. Les préférences affichées n’ont pas été confirmées.');
        return;
      }
      const next = { ...EMPTY, ...saved };
      setPreferences(next);
      setRecorded(next);
      setHistory(Array.isArray(data.history) ? data.history : []);
      setMessage('Préférences enregistrées. Disponibilité réelle : non. Paiement non activé.');
    } catch (err) {
      setError(
        err?.response?.data?.error
          || 'Enregistrement refusé. Les paiements réels restent désactivés.'
      );
    } finally {
      setSaving(false);
    }
  };

  const checkboxStyle = {
    flex: '0 0 auto',
    width: 18,
    height: 18,
    minWidth: 18,
    maxWidth: 18,
    margin: '4px 0 0',
    padding: 0,
  };

  return (
    <section
      aria-label="Paiements et facturation"
      style={{
        width: '100%',
        maxWidth: '100%',
        boxSizing: 'border-box',
        marginBottom: 16,
        overflowX: 'hidden',
      }}
    >
      <h3 style={{ margin: '0 0 8px', fontSize: 18, overflowWrap: 'anywhere' }}>
        Paiements et facturation
      </h3>
      <p style={{ margin: '0 0 12px', overflowWrap: 'anywhere' }}>
        Une préférence enregistrée ne rend pas le moyen disponible.
      </p>
      {loading && <p role="status">Chargement des préférences.</p>}
      {MEANS.map((mean) => (
        <div
          key={mean.key}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 12,
            marginBottom: 12,
            maxWidth: '100%',
            minWidth: 0,
          }}
        >
          <label
            htmlFor={`settlement-${mean.key}`}
            style={{
              flex: '1 1 auto',
              minWidth: 0,
              whiteSpace: 'normal',
              overflowWrap: 'anywhere',
            }}
          >
            {mean.label}
            <small style={{ display: 'block', whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{mean.wish}</small>
            <small style={{ display: 'block', whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
              Préférence enregistrée : {recorded[mean.key] ? 'oui' : 'non'}.
            </small>
            <small style={{ display: 'block', whiteSpace: 'normal', overflowWrap: 'anywhere' }}>Disponibilité réelle : non. Indisponible.</small>
            <small style={{ display: 'block', overflowWrap: 'anywhere' }}>{mean.prerequisite}</small>
          </label>
          <input
            id={`settlement-${mean.key}`}
            type="checkbox"
            checked={Boolean(preferences[mean.key])}
            onChange={(event) =>
              setPreferences((current) => ({
                ...current,
                [mean.key]: event.target.checked,
              }))
            }
            style={checkboxStyle}
          />
        </div>
      ))}
      <button type="button" onClick={save} disabled={saving || loading}>
        {saving ? 'Enregistrement…' : 'Enregistrer les préférences'}
      </button>
      <section aria-label="Conventions et comptes clients" style={{ marginTop: 16 }}>
        <h4 style={{ margin: '0 0 8px', fontSize: 16 }}>Conventions et comptes clients</h4>
        <p style={{ margin: 0, overflowWrap: 'anywhere' }}>
          Les conventions et comptes mensuels existants ne sont pas modifiés ici.
        </p>
      </section>
      <label
        htmlFor="settlement-patient-notice"
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: 12,
          marginTop: 16,
          maxWidth: '100%',
          minWidth: 0,
          overflowWrap: 'anywhere',
        }}
      >
        <input
          id="settlement-patient-notice"
          type="checkbox"
          checked={noticeChecked}
          onChange={(event) => setNoticeChecked(event.target.checked)}
          style={checkboxStyle}
        />
        <span style={{ flex: '1 1 auto', minWidth: 0, whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
          Les conditions de règlement ont été communiquées au patient.
        </span>
      </label>
      <p style={{ margin: '8px 0 0', maxWidth: '100%', overflowWrap: 'anywhere' }}>
        Cette déclaration n’est pas une preuve de réception ni d’acceptation par le patient.
        Elle se fait pour une réservation précise et n’est pas enregistrée depuis cet écran.
      </p>
      {history.length > 0 && (
        <ul
          aria-label="Historique des préférences"
          style={{
            paddingLeft: 18,
            margin: '8px 0 0',
            maxWidth: '100%',
            boxSizing: 'border-box',
            overflowWrap: 'anywhere',
          }}
        >
          {history.map((item) => (
            <li key={item.revision} style={{ overflowWrap: 'anywhere' }}>
              Révision {item.revision}, auteur {item.author_user_id ?? 'inconnu'}, {item.created_at || 'heure serveur'}.
            </li>
          ))}
        </ul>
      )}
      {message && <p role="status">{message}</p>}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
