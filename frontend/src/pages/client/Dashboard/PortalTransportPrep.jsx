import React, { useMemo, useState } from 'react';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import apiClient from '../../../utils/apiClient';
import { portalTransportProfileGaps } from './portalTransportProfile';

function initialDraft(profile) {
  const access = profile?.access || {};
  const mobility = profile?.mobility || {};
  return {
    first_name: profile?.first_name || profile?.user?.first_name || '',
    last_name: profile?.last_name || profile?.user?.last_name || '',
    birth_date: profile?.birth_date || profile?.user?.birth_date || '',
    address: profile?.domicile?.address || profile?.user?.address || profile?.address || '',
    lat: profile?.domicile?.lat ?? profile?.domicile_lat ?? null,
    lon: profile?.domicile?.lon ?? profile?.domicile_lon ?? null,
    floor: access.floor || '',
    door_code: access.door_code || '',
    access_notes: access.notes || '',
    wheelchairOwn: Boolean(mobility.wheelchair_client_has),
    wheelchairNeed: Boolean(mobility.wheelchair_need),
    assistance: Boolean(mobility.needs_assistance),
  };
}

/**
 * Écran de fin d'inscription : les données stables, avant la première réservation.
 */
export default function PortalTransportPrep({ profile, clientId, authHeaders, onSaved }) {
  const [draft, setDraft] = useState(() => initialDraft(profile));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const phoneMissing = portalTransportProfileGaps(profile).includes('phone');

  const canConfirm = useMemo(() => {
    if (phoneMissing || saving) return false;
    if (!String(draft.first_name).trim() || !String(draft.last_name).trim()) return false;
    if (!String(draft.birth_date).trim()) return false;
    if (!String(draft.address).trim() || draft.lat == null) return false;
    if (draft.wheelchairOwn && draft.wheelchairNeed) return false;
    return true;
  }, [draft, phoneMissing, saving]);

  const confirm = async () => {
    setError('');
    setSaving(true);
    try {
      const payload = {
        first_name: String(draft.first_name).trim(),
        last_name: String(draft.last_name).trim(),
        birth_date: String(draft.birth_date).trim(),
        address: String(draft.address).trim(),
        domicile_lat: draft.lat,
        domicile_lon: draft.lon,
        floor: String(draft.floor || '').trim(),
        door_code: String(draft.door_code || '').trim(),
        access_notes: String(draft.access_notes || '').trim(),
        habitual_wheelchair_client_has: draft.wheelchairOwn,
        habitual_wheelchair_need: draft.wheelchairNeed,
        habitual_needs_assistance: draft.assistance,
      };
      await apiClient.put(`/clients/${clientId}`, payload, authHeaders);
      const refreshed = await apiClient.get(`/clients/${clientId}`, authHeaders);
      onSaved(refreshed.data);
    } catch (err) {
      const message =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        'Les informations n’ont pas pu être enregistrées.';
      setError(typeof message === 'string' ? message : 'Les informations n’ont pas pu être enregistrées.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="portalTransportPrep" data-testid="portal-transport-prep">
      <h2 className="portalTransportPrepTitle">Préparer vos futurs transports</h2>
      <p className="portalTransportPrepLead">
        Ces informations restent sur votre profil. Chaque réservation en garde une copie, modifiable pour cette course
        seulement.
      </p>
      {phoneMissing ? (
        <p className="error" role="alert">
          Vérifiez votre numéro de téléphone avant de confirmer ces informations.
        </p>
      ) : null}
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-first">
        Prénom
      </label>
      <input
        id="portal-prep-first"
        className="input"
        value={draft.first_name}
        onChange={(e) => setDraft((prev) => ({ ...prev, first_name: e.target.value }))}
      />
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-last">
        Nom
      </label>
      <input
        id="portal-prep-last"
        className="input"
        value={draft.last_name}
        onChange={(e) => setDraft((prev) => ({ ...prev, last_name: e.target.value }))}
      />
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-birth">
        Date de naissance
      </label>
      <input
        id="portal-prep-birth"
        className="input"
        type="date"
        value={draft.birth_date}
        onChange={(e) => setDraft((prev) => ({ ...prev, birth_date: e.target.value }))}
      />
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-address">
        Adresse principale
      </label>
      <AddressAutocomplete
        inputId="portal-prep-address"
        name="portal-prep-address"
        value={draft.address}
        onChange={(e) =>
          setDraft((prev) => ({
            ...prev,
            address: e.target.value,
            lat: null,
            lon: null,
          }))
        }
        onSelect={(item) =>
          setDraft((prev) => ({
            ...prev,
            address: item.label || '',
            lat: item?.lat ?? null,
            lon: item?.lon ?? null,
          }))
        }
        placeholder="Rue du Test 1"
      />
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-floor">
        Étage / appartement <span className="bookingClientNoteSummaryOptional">(facultatif)</span>
      </label>
      <input
        id="portal-prep-floor"
        className="input"
        value={draft.floor}
        onChange={(e) => setDraft((prev) => ({ ...prev, floor: e.target.value }))}
        placeholder="3e"
      />
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-door">
        Code / interphone <span className="bookingClientNoteSummaryOptional">(facultatif)</span>
      </label>
      <input
        id="portal-prep-door"
        className="input"
        value={draft.door_code}
        onChange={(e) => setDraft((prev) => ({ ...prev, door_code: e.target.value }))}
        placeholder="Osmani"
      />
      <label className="bookingClientNoteRowLabel" htmlFor="portal-prep-access">
        Complément d’accès <span className="bookingClientNoteSummaryOptional">(facultatif)</span>
      </label>
      <input
        id="portal-prep-access"
        className="input"
        value={draft.access_notes}
        onChange={(e) => setDraft((prev) => ({ ...prev, access_notes: e.target.value }))}
        placeholder="Accès par l'entrée arrière"
      />
      <span className="bookingClientNoteRowLabel">Besoins habituels</span>
      <div className="portalPrepChips" role="group" aria-label="Besoins habituels">
        <button
          type="button"
          className="portalPrepChip"
          aria-pressed={draft.wheelchairOwn}
          onClick={() =>
            setDraft((prev) => ({
              ...prev,
              wheelchairOwn: !prev.wheelchairOwn,
              wheelchairNeed: !prev.wheelchairOwn ? false : prev.wheelchairNeed,
            }))
          }
        >
          En fauteuil
        </button>
        <button
          type="button"
          className="portalPrepChip"
          aria-pressed={draft.wheelchairNeed}
          onClick={() =>
            setDraft((prev) => ({
              ...prev,
              wheelchairNeed: !prev.wheelchairNeed,
              wheelchairOwn: !prev.wheelchairNeed ? false : prev.wheelchairOwn,
            }))
          }
        >
          Fournir fauteuil
        </button>
        <button
          type="button"
          className="portalPrepChip"
          aria-pressed={draft.assistance}
          onClick={() => setDraft((prev) => ({ ...prev, assistance: !prev.assistance }))}
        >
          Assistance
        </button>
      </div>
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <button type="button" className="save-button" disabled={!canConfirm} onClick={confirm}>
        {saving ? 'Enregistrement…' : 'Confirmer mes informations'}
      </button>
    </div>
  );
}
