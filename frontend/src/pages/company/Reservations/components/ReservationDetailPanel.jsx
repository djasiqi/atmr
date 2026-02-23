import React, { useCallback, useEffect, useState, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  FiX, FiTruck, FiMapPin, FiInfo, FiClock, FiFileText, FiUser, FiPhone,
  FiAlertCircle, FiEdit2, FiPackage, FiHome, FiTrash2,
} from 'react-icons/fi';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import { fetchTransportVouchers } from '../../../../services/transportVoucherService';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../../components/ui/InlineTimePicker';
import useCompanySocket from '../../../../hooks/useCompanySocket';
import BookingChat from './BookingChat';
import s from './ReservationDetailPanel.module.css';

const STATUS_MAP = {
  pending:            { label: 'En attente',       css: 'statusPending' },
  accepted:           { label: 'Acceptee',         css: 'statusAccepted' },
  assigned:           { label: 'Assignee',         css: 'statusAssigned' },
  en_route:           { label: 'En route',         css: 'statusEnRoute' },
  in_progress:        { label: 'En cours',         css: 'statusInProgress' },
  completed:          { label: 'Terminee',         css: 'statusCompleted' },
  return_completed:   { label: 'Retour termine',   css: 'statusCompleted' },
  canceled:           { label: 'Annulee',          css: 'statusCancelled' },
  cancelled:          { label: 'Annulee',          css: 'statusCancelled' },
  rejected:           { label: 'Refusee',          css: 'statusCancelled' },
  no_show:            { label: 'Non presente',     css: 'statusCancelled' },
};

const VOUCHER_STATUS_LABELS = {
  draft: 'Brouillon', submitted: 'Soumis', validated: 'Valide', rejected: 'Rejete', expired: 'Expire',
};

const VOUCHER_TYPE_LABELS = {
  clinic: 'Clinique', insurance: 'Assurance', other: 'Autre',
};

const INTENT_LABELS = {
  institution: 'Institution', clinic: 'Clinique', patient: 'Patient',
  curator: 'Curateur', spc: 'SPC', other: 'Autre',
};

const formatCurrency = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return `${n.toFixed(2)} CHF`;
};

const fmtShort = (dateStr) => {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleString('fr-CH', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  });
};

const buildTimeline = (r) => {
  if (!r) return [];
  const events = [];
  const driverName = r.driver_name || r.driver?.full_name;

  // Institution request events (if booking came from an institution request)
  const it = r.institution_timeline;
  if (it) {
    const instName = it.institution_name;
    if (it.created_at) {
      events.push({
        event: `Demande creee${it.created_by_name ? ` par ${it.created_by_name}` : ''}${instName ? ` (${instName})` : ''}`,
        date: it.created_at,
      });
    }
    if (it.sent_at) events.push({ event: 'Demande envoyee', date: it.sent_at });
    if (it.accepted_at) {
      events.push({
        event: `Demande acceptee${it.accepted_by_company_name ? ` par ${it.accepted_by_company_name}` : ''}`,
        date: it.accepted_at,
      });
    }
    if (it.converted_at) events.push({ event: 'Reservation creee', date: it.converted_at });
    if (it.cancelled_at) events.push({ event: 'Demande annulee', date: it.cancelled_at });
  } else {
    if (r.created_at) events.push({ event: 'Reservation creee', date: r.created_at });
  }

  // Booking lifecycle events
  if (r.accepted_at && !it?.accepted_at) {
    events.push({ event: `Acceptee${driverName ? ` par ${driverName}` : ''}`, date: r.accepted_at });
  }
  if (r.assigned_at) events.push({ event: `Assignee${driverName ? ` a ${driverName}` : ''}`, date: r.assigned_at });
  if (r.picked_up_at || r.boarded_at) {
    events.push({ event: `Client pris en charge${driverName ? ` par ${driverName}` : ''}`, date: r.picked_up_at || r.boarded_at });
  }
  if (r.started_at) events.push({ event: 'Course demarree', date: r.started_at });
  if (r.completed_at) events.push({ event: 'Course terminee', date: r.completed_at });
  if ((r.cancelled_at || r.canceled_at) && !it?.cancelled_at) {
    const roleMap = { company: 'Entreprise', driver: 'Chauffeur', admin: 'Admin', system: 'Système' };
    const byLabel = roleMap[r.cancelled_by_role] || '';
    const reasonLabel = r.cancellation_display_label || r.cancellation_reason_code || '';
    const billable = r.is_cancellation_billable;

    let detail = 'Annulee';
    if (byLabel) detail += ` par ${byLabel}`;
    if (reasonLabel) detail += ` — ${reasonLabel}`;
    if (billable === true) detail += ' (facturee)';
    else if (billable === false) detail += ' (non facturee)';

    events.push({ event: detail, date: r.cancelled_at || r.canceled_at, type: 'cancel' });
  }

  return events.sort((a, b) => new Date(b.date) - new Date(a.date));
};

const parseDate = (r) => {
  const raw = r?.scheduled_time || r?.scheduled_date || r?.date;
  if (!raw) return '';
  const dt = new Date(raw);
  if (isNaN(dt.getTime())) return raw.slice(0, 10);
  const pad = (n) => String(n).padStart(2, '0');
  return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
};

const parseTime = (r) => {
  const raw = r?.scheduled_time || r?.scheduled_date || r?.date;
  if (!raw) return '';
  const dt = new Date(raw);
  if (isNaN(dt.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
};

const ReservationDetailPanel = ({ reservation, onClose, onSave, onDelete }) => {
  const [vouchers, setVouchers] = useState([]);
  const [loadingVouchers, setLoadingVouchers] = useState(false);
  const [searchParams] = useSearchParams();
  const lastVoucherErrorRef = useRef(null);
  const companySocket = useCompanySocket();

  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [form, setForm] = useState({});

  const isDateTimeInPast = useMemo(() => {
    if (!form.scheduled_date || !form.scheduled_time) return false;
    const combined = new Date(`${form.scheduled_date}T${form.scheduled_time}:00`);
    return !isNaN(combined.getTime()) && combined < new Date();
  }, [form.scheduled_date, form.scheduled_time]);

  const buildFormFromReservation = useCallback((r) => {
    if (!r) return {};
    const meta = r.metadata_json || {};
    const routing = meta.routing || {};
    return {
      pickup_location: r.pickup_location || '',
      dropoff_location: r.dropoff_location || '',
      scheduled_date: parseDate(r),
      scheduled_time: parseTime(r),
      amount: r.amount ?? '',
      // Accès départ
      pickup_access_notes: r.pickup_access_notes || '',
      pickup_floor: routing.pickup_floor || r.client?.floor || '',
      pickup_door_code: routing.pickup_door_code || r.client?.door_code || '',
      pickup_entry_point: routing.pickup_entry_point || '',
      // Arrivée — établissement / service / médecin
      dropoff_access_notes: r.dropoff_access_notes || '',
      medical_facility: (r.medical_facility && r.medical_facility !== 'Non spécifié') ? r.medical_facility : '',
      hospital_service: (r.hospital_service && r.hospital_service !== 'Non spécifié') ? r.hospital_service : '',
      doctor_name: (r.doctor_name && r.doctor_name !== 'Non spécifié') ? r.doctor_name : '',
      // Patient & contact
      phone: r.phone || r.client?.contact_phone || r.client?.phone || '',
      notes_medical: (r.notes_medical && r.notes_medical !== 'Aucune note') ? r.notes_medical : '',
      instructions: r.instructions || '',
      external_reference: meta.external_reference || r.external_reference || '',
    };
  }, []);

  useEffect(() => {
    if (reservation) {
      setForm(buildFormFromReservation(reservation));
      setEditing(false);
    }
  }, [reservation, buildFormFromReservation]);

  const returnTo = useMemo(() => {
    const raw = searchParams.get('returnTo');
    if (!raw) return null;
    try {
      const decoded = decodeURIComponent(raw);
      if (decoded.startsWith('/dashboard/company/') || decoded.startsWith('/company/')) return decoded;
      return null;
    } catch { return null; }
  }, [searchParams]);

  const loadVouchers = useCallback(async () => {
    if (!reservation?.id) return;
    try {
      setLoadingVouchers(true);
      const response = await fetchTransportVouchers({ booking_id: reservation.id });
      setVouchers(response?.data || []);
    } catch (e) {
      const errorKey = `${reservation.id}:${e?.response?.status || e?.message || 'unknown'}`;
      if (lastVoucherErrorRef.current !== errorKey) {
        lastVoucherErrorRef.current = errorKey;
      }
    } finally {
      setLoadingVouchers(false);
    }
  }, [reservation?.id]);

  useEffect(() => {
    if (reservation?.id) loadVouchers();
  }, [reservation?.id, loadVouchers]);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    if (!onSave || !reservation?.id) return;
    if (isDateTimeInPast) return;
    try {
      setSaving(true);
      setSaveError(null);
      // Combiner date + heure en ISO local pour le backend (pas de conversion UTC)
      const payload = { ...form };
      if (payload.scheduled_date && payload.scheduled_time) {
        payload.scheduled_time = `${payload.scheduled_date}T${payload.scheduled_time}:00`;
      }
      delete payload.scheduled_date;
      await onSave(reservation.id, payload);
      setEditing(false);
    } catch (err) {
      console.error('Erreur lors de la sauvegarde:', err);
      const errData = err?.response?.data;
      const message = errData?.message || errData?.error || 'Erreur lors de la sauvegarde';
      setSaveError(message);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setForm(buildFormFromReservation(reservation));
    setEditing(false);
    setSaveError(null);
  };

  if (!reservation) return null;

  const status = reservation.status?.toLowerCase() || 'unknown';
  const statusInfo = STATUS_MAP[status] || { label: status, css: 'statusPending' };
  const meta = reservation.metadata_json || {};
  const billedToType = reservation.billed_to_type
    || reservation.billing?.billed_to_type
    || meta.billing_resolution_intent
    || null;
  const billingIntent = meta.billing_resolution_intent || billedToType;
  const billingStatusVal = meta.billing_resolution_status;
  const isFailed = billingStatusVal && billingStatusVal.startsWith('failed');
  const isInstitutionBooking = !!meta.institution_id || !!reservation.institution_timeline;
  const chatBookingId = reservation.is_return && reservation.parent_booking_id
    ? reservation.parent_booking_id
    : reservation.id;
  const isMaterialDelivery = reservation.mission_type === 'material_delivery';

  const originalAmount = reservation?.amount_original ?? reservation?.original_amount ?? reservation?.requested_amount;
  const adjustedDelta = Number.isFinite(Number(originalAmount))
    ? Number(reservation?.amount ?? 0) - Number(originalAmount) : null;

  return (
    <div className={s.panel}>
      {/* Header */}
      <div className={s.panelHeader}>
        <div className={s.panelTitleRow}>
          <span className={s.panelTitle}>Reservation #{reservation.id}</span>
          <span className={`${s.statusBadge} ${s[statusInfo.css]}`}>{statusInfo.label}</span>
        </div>
        <div className={s.headerActions}>
          {!editing && onSave && (
            <button className={s.editBtn} onClick={() => setEditing(true)} aria-label="Editer" title="Editer">
              <FiEdit2 size={14} />
            </button>
          )}
          <button className={s.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={16} />
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      <div className={s.panelBody}>

        {/* ── EDIT MODE ── */}
        {editing ? (
          <div className={s.editForm}>

            {/* Context badge — client + id */}
            <div className={s.editContext}>
              <span className={s.editContextLabel}>
                {reservation.client?.full_name || reservation.client_name || 'Client'}
              </span>
              <span className={s.editContextSep} />
              <span className={s.editContextMeta}>#{reservation.id}</span>
            </div>

            {/* ─── Itineraire ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiMapPin size={12} className={s.editGroupIcon} />
                Itineraire
              </div>

              {/* Route visuelle : dot → ligne → dot */}
              <div className={s.editRoute}>
                <div className={s.editRouteTrack}>
                  <span className={s.editRouteDotA} />
                  <span className={s.editRouteLine} />
                  <span className={s.editRouteDotB} />
                </div>
                <div className={s.editRouteFields}>
                  <div className={s.editInputWrap}>
                    <AddressAutocomplete
                      value={form.pickup_location}
                      onChange={(e) => handleChange('pickup_location', e?.target?.value ?? e)}
                      placeholder="Depart"
                    />
                  </div>
                  <div className={s.editInputWrap}>
                    <AddressAutocomplete
                      value={form.dropoff_location}
                      onChange={(e) => handleChange('dropoff_location', e?.target?.value ?? e)}
                      placeholder="Destination"
                    />
                  </div>
                </div>
              </div>

              {/* Accès départ — compact row */}
              <div className={s.editSubLabel}>Acces depart</div>
              <div className={s.editRowTriple}>
                <input type="text" className={s.editInput} value={form.pickup_floor}
                  onChange={(e) => handleChange('pickup_floor', e.target.value)} placeholder="Etage" />
                <input type="text" className={s.editInput} value={form.pickup_door_code}
                  onChange={(e) => handleChange('pickup_door_code', e.target.value)} placeholder="Code" />
                <input type="text" className={s.editInput} value={form.pickup_entry_point}
                  onChange={(e) => handleChange('pickup_entry_point', e.target.value)} placeholder="Accueil" />
              </div>
              <input type="text" className={s.editInput} value={form.pickup_access_notes}
                onChange={(e) => handleChange('pickup_access_notes', e.target.value)}
                placeholder="Consignes depart" style={{ marginTop: 6 }} />
            </div>

            <div className={s.editDivider} />

            {/* ─── Destination ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiHome size={12} className={s.editGroupIcon} />
                Destination
              </div>
              <input type="text" className={s.editInput} value={form.medical_facility}
                onChange={(e) => handleChange('medical_facility', e.target.value)}
                placeholder="Etablissement / Lieu" />
              <div className={s.editRow} style={{ marginTop: 8 }}>
                <input type="text" className={s.editInput} value={form.hospital_service}
                  onChange={(e) => handleChange('hospital_service', e.target.value)}
                  placeholder="Service" />
                <input type="text" className={s.editInput} value={form.doctor_name}
                  onChange={(e) => handleChange('doctor_name', e.target.value)}
                  placeholder="Medecin" />
              </div>
              <input type="text" className={s.editInput} value={form.dropoff_access_notes}
                onChange={(e) => handleChange('dropoff_access_notes', e.target.value)}
                placeholder="Consignes arrivee" style={{ marginTop: 8 }} />
            </div>

            <div className={s.editDivider} />

            {/* ─── Horaire ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiClock size={12} className={s.editGroupIcon} />
                Horaire
              </div>
              <div className={s.editRow}>
                <div className={s.editField}>
                  <label className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}>Date</label>
                  <InlineDatePicker
                    value={form.scheduled_date}
                    onChange={(v) => handleChange('scheduled_date', v)}
                    placeholder="Date"
                  />
                </div>
                <div className={s.editField}>
                  <label className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}>Heure</label>
                  <InlineTimePicker
                    value={form.scheduled_time}
                    onChange={(v) => handleChange('scheduled_time', v)}
                    className={isDateTimeInPast ? s.fieldErrorInput : ''}
                  />
                </div>
              </div>
              {isDateTimeInPast && (
                <div className={s.fieldErrorHint}>La date et l'heure sont dans le passe</div>
              )}
            </div>

            <div className={s.editDivider} />

            {/* ─── Montant ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiFileText size={12} className={s.editGroupIcon} />
                Montant
              </div>
              <input type="number" step="0.01" className={s.editInput} value={form.amount}
                onChange={(e) => handleChange('amount', e.target.value)} placeholder="CHF" />
            </div>

            <div className={s.editDivider} />

            {/* ─── Patient & notes ─── */}
            {!isMaterialDelivery ? (
              <div className={s.editGroup}>
                <div className={s.editGroupTitle}>
                  <FiUser size={12} className={s.editGroupIcon} />
                  Patient
                </div>
                <div className={s.editRow}>
                  <input type="tel" className={s.editInput} value={form.phone}
                    onChange={(e) => handleChange('phone', e.target.value)} placeholder="Telephone" />
                  <input type="text" className={s.editInput} value={form.external_reference}
                    onChange={(e) => handleChange('external_reference', e.target.value)} placeholder="Ref. DPI" />
                </div>
                <textarea className={s.editTextarea} value={form.notes_medical}
                  onChange={(e) => handleChange('notes_medical', e.target.value)}
                  placeholder="Pathologie, difficultes, mobilite..." rows={2} style={{ marginTop: 8 }} />
                <textarea className={s.editTextarea} value={form.instructions}
                  onChange={(e) => handleChange('instructions', e.target.value)}
                  placeholder="Instructions chauffeur" rows={2} style={{ marginTop: 8 }} />
              </div>
            ) : (
              <div className={s.editGroup}>
                <div className={s.editGroupTitle}>
                  <FiPackage size={12} className={s.editGroupIcon} />
                  Livraison
                </div>
                <textarea className={s.editTextarea} value={form.instructions}
                  onChange={(e) => handleChange('instructions', e.target.value)}
                  placeholder="Description du materiel" rows={3} />
              </div>
            )}

            {/* Erreur de sauvegarde */}
            {saveError && (
              <div className={s.saveErrorBanner}>{saveError}</div>
            )}

            {/* Footer actions — sticky */}
            <div className={s.editFooter}>
              <button type="button" className={s.editCancelBtn} onClick={handleCancel} disabled={saving}>
                Annuler
              </button>
              <button type="button" className={s.editSaveBtn} onClick={handleSave} disabled={saving}>
                {saving ? 'Enregistrement...' : 'Enregistrer'}
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* ── VIEW MODE ── */}

            {/* Informations principales */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FiUser size={13} /></div>
                <h3 className={s.sectionTitle}>Informations</h3>
              </div>
              <div className={s.summaryGrid}>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Client</span>
                  <span className={s.summaryValue}>{reservation.client?.full_name || reservation.client_name || '-'}</span>
                </div>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Horaire</span>
                  <span className={s.summaryValue}>{renderBookingDateTime(reservation)}</span>
                </div>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Montant</span>
                  <span className={s.summaryValue}>{formatCurrency(reservation.amount)}</span>
                </div>
                {reservation.driver_name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Chauffeur</span>
                    <span className={s.summaryValue}>{reservation.driver_name}</span>
                  </div>
                )}
                {reservation.client?.institution_name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Institution</span>
                    <span className={s.summaryValue}>{reservation.client.institution_name}</span>
                  </div>
                )}
                {(reservation.client?.contact_phone || reservation.client?.phone) && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Telephone</span>
                    <span className={s.summaryValue}>{reservation.client.contact_phone || reservation.client.phone}</span>
                  </div>
                )}
                {reservation.client?.birth_date && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Date de naissance</span>
                    <span className={s.summaryValue}>
                      {new Date(reservation.client.birth_date).toLocaleDateString('fr-CH')}
                    </span>
                  </div>
                )}
              </div>
              {Number.isFinite(Number(originalAmount)) && Number.isFinite(Number(adjustedDelta)) && Math.abs(adjustedDelta) >= 0.01 && (
                <div className={s.adjustedNote}>
                  Montant saisi : {formatCurrency(originalAmount)} — Ajuste : {adjustedDelta >= 0 ? '+' : '-'}{formatCurrency(Math.abs(adjustedDelta))}
                </div>
              )}
            </div>

            {/* Trajet */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FiMapPin size={13} /></div>
                <h3 className={s.sectionTitle}>Trajet</h3>
              </div>
              <div className={s.route}>
                <div className={s.routeTrack}>
                  <div className={`${s.routeDot} ${s.routeDotStart}`} />
                  <div className={s.routeLine} />
                  <div className={`${s.routeDot} ${s.routeDotEnd}`} />
                </div>
                <div className={s.routeStops}>
                  <div className={s.routeStop}>
                    <div className={s.routeStopLabel}>Depart</div>
                    <div className={s.routeStopAddress}>{reservation.pickup_location || '-'}</div>
                    {/* Access details départ */}
                    {(reservation.client?.floor || reservation.client?.door_code || reservation.client?.access_notes || reservation.pickup_access_notes) && (
                      <div className={s.routeStopMeta}>
                        {reservation.client?.floor && <span>Etage {reservation.client.floor}</span>}
                        {reservation.client?.door_code && <span>Code {reservation.client.door_code}</span>}
                        {(reservation.pickup_access_notes || reservation.client?.access_notes) && (
                          <span>{reservation.pickup_access_notes || reservation.client.access_notes}</span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className={s.routeStop}>
                    <div className={s.routeStopLabel}>Arrivee</div>
                    <div className={s.routeStopAddress}>{reservation.dropoff_location || '-'}</div>
                    {reservation.dropoff_access_notes && (
                      <div className={s.routeStopMeta}>
                        <span>{reservation.dropoff_access_notes}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {reservation.is_return && (
                <div className={s.roundTripBadge}>Aller-retour</div>
              )}
            </div>

            {/* Destination — établissement, service, médecin */}
            {(() => {
              const facility = reservation.medical_facility && reservation.medical_facility !== 'Non spécifié' ? reservation.medical_facility : null;
              const service = reservation.hospital_service && reservation.hospital_service !== 'Non spécifié' ? reservation.hospital_service : null;
              const doctor = reservation.doctor_name && reservation.doctor_name !== 'Non spécifié' ? reservation.doctor_name : null;
              if (!facility && !service && !doctor) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconBlue}`}><FiHome size={13} /></div>
                    <h3 className={s.sectionTitle}>Destination</h3>
                  </div>
                  <div className={s.detailGrid}>
                    {facility && (
                      <div className={s.detailItem}>
                        <span className={s.detailLabel}>Etablissement</span>
                        <span className={s.detailValue}>{facility}</span>
                      </div>
                    )}
                    {service && (
                      <div className={s.detailItem}>
                        <span className={s.detailLabel}>Service</span>
                        <span className={s.detailValue}>{service}</span>
                      </div>
                    )}
                    {doctor && (
                      <div className={s.detailItem}>
                        <span className={s.detailLabel}>Medecin</span>
                        <span className={s.detailValue}>{doctor}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* Patient — pathologie, notes, ref DPI */}
            {(() => {
              const notesMed = reservation.notes_medical && reservation.notes_medical !== 'Aucune note' ? reservation.notes_medical : null;
              const instructions = reservation.instructions || null;
              const phone = reservation.phone || null;
              const extRef = (meta.external_reference || reservation.external_reference) || null;
              const wheelchair = reservation.wheelchair_need || reservation.wheelchair_client_has;
              if (!notesMed && !instructions && !phone && !extRef && !wheelchair) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FiInfo size={13} /></div>
                    <h3 className={s.sectionTitle}>Patient & notes</h3>
                  </div>
                  {phone && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}><FiPhone size={11} /> Telephone</span>
                      <span className={s.infoValue}>{phone}</span>
                    </div>
                  )}
                  {extRef && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}><FiFileText size={11} /> Ref. DPI</span>
                      <span className={s.infoValue}>{extRef}</span>
                    </div>
                  )}
                  {wheelchair && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}><FiTruck size={11} /> Mobilite</span>
                      <span className={s.infoValue}>
                        {reservation.wheelchair_need ? 'Fauteuil requis' : 'Fauteuil client'}
                      </span>
                    </div>
                  )}
                  {notesMed && (
                    <div className={s.notesBlock}>{notesMed}</div>
                  )}
                  {instructions && (
                    <div className={s.notesBlockMuted}>{instructions}</div>
                  )}
                </div>
              );
            })()}

            {/* Facturation */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FiFileText size={13} /></div>
                <h3 className={s.sectionTitle}>Facturation</h3>
              </div>
              {(() => {
                const isCancelled = ['canceled', 'cancelled'].includes(status);
                const billable = reservation.is_cancellation_billable;

                if (isCancelled && billable === false) {
                  return (
                    <div className={`${s.billingStatus} ${s.billingStatusCancelled}`}>
                      <FiAlertCircle size={13} />
                      Annulee — non facturee
                      {reservation.cancellation_display_label && (
                        <span className={s.billingMuted} style={{ marginLeft: 4 }}>
                          ({reservation.cancellation_display_label})
                        </span>
                      )}
                    </div>
                  );
                }

                const resolvedIntent = billingIntent || billedToType || 'patient';
                const intentLabel = INTENT_LABELS[resolvedIntent] || resolvedIntent;
                const instName = meta.institution_name || reservation.institution_timeline?.institution_name;

                if (isCancelled && billable === true) {
                  const feeAmount = reservation.cancellation_fee_amount;
                  const feePct = reservation.cancellation_fee_percent;
                  return (
                    <>
                      <div className={`${s.billingStatus} ${s.billingStatusDanger}`}>
                        Annulee — facturee
                        <span className={`${s.billingBadge} ${s.billingBadgeDanger}`}>Facturation maintenue</span>
                      </div>
                      {feeAmount != null && feeAmount > 0 && (
                        <p className={s.billingMuted} style={{ fontWeight: 600 }}>
                          Frais d'annulation : {Number(feeAmount).toFixed(2)} CHF
                          {feePct != null ? ` (${feePct}%)` : ''}
                        </p>
                      )}
                      <p className={s.billingMuted}>
                        Facture a : {intentLabel}
                        {instName ? ` — ${instName}` : ''}
                      </p>
                      {reservation.cancellation_display_label && (
                        <p className={s.billingMuted}>Motif : {reservation.cancellation_display_label}</p>
                      )}
                    </>
                  );
                }

                if (isInstitutionBooking) {
                  return (
                    <>
                      <div className={`${s.billingStatus} ${isFailed ? s.billingStatusDanger : s.billingStatusSuccess}`}>
                        Facture a : {intentLabel}
                        {billingStatusVal && (
                          <span className={`${s.billingBadge} ${isFailed ? s.billingBadgeDanger : s.billingBadgeSuccess}`}>
                            {isFailed ? 'Action requise' : 'Resolu'}
                          </span>
                        )}
                      </div>
                      {instName && (
                        <p className={s.billingMuted}>Institution : {instName}</p>
                      )}
                      {isFailed && (
                        <div className={s.billingWarning}>
                          <FiAlertCircle size={13} />
                          <span>Informations de facturation incompletes. Verifiez le dossier client.</span>
                        </div>
                      )}
                    </>
                  );
                }

                return (
                  <div className={`${s.billingStatus} ${s.billingStatusPatient}`}>
                    Facture a : {intentLabel}
                  </div>
                );
              })()}
            </div>

            {/* Mini-chat institution (thread unifié A/R via parent) */}
            {isInstitutionBooking && (
              <BookingChat
                bookingId={chatBookingId}
                socket={companySocket}
                closed={['completed', 'return_completed', 'canceled', 'cancelled'].includes(status)}
              />
            )}

            {/* Historique */}
            {(() => {
              const timeline = buildTimeline(reservation);
              if (timeline.length === 0) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FiClock size={13} /></div>
                    <h3 className={s.sectionTitle}>Historique</h3>
                  </div>
                  <div className={s.timeline}>
                    {timeline.map((item, i) => (
                      <div key={i} className={`${s.timelineItem} ${item.type === 'cancel' ? s.timelineItemCancel : ''}`}>
                        <div className={s.timelineEvent}>{item.event}</div>
                        <div className={s.timelineDate}>{fmtShort(item.date)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Vouchers */}
            {!loadingVouchers && vouchers.length > 0 && (
              <div className={s.section}>
                <div className={s.sectionHeader}>
                  <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FiTruck size={13} /></div>
                  <h3 className={s.sectionTitle}>Bons de transport</h3>
                </div>
                <div className={s.voucherList}>
                  {vouchers.map((v) => (
                    <div key={v.id} className={s.voucherCard}>
                      <div className={s.voucherHeader}>
                        <span className={s.voucherTitle}>Bon #{v.id} - {VOUCHER_TYPE_LABELS[v.type] || v.type}</span>
                        <span className={s.voucherStatus}>{VOUCHER_STATUS_LABELS[v.status] || v.status}</span>
                      </div>
                      {v.external_ref && (
                        <div className={s.voucherInfo}>Ref : {v.external_ref}</div>
                      )}
                      {v.valid_from && (
                        <div className={s.voucherInfo}>
                          Periode : {v.valid_to
                            ? `${new Date(v.valid_from).toLocaleDateString('fr-CH')} - ${new Date(v.valid_to).toLocaleDateString('fr-CH')}`
                            : `A partir du ${new Date(v.valid_from).toLocaleDateString('fr-CH')}`
                          }
                        </div>
                      )}
                      {v.files && v.files.length > 0 && (
                        <div className={s.voucherInfo}>
                          {v.files.map((f) => (
                            <a key={f.id} href={f.file_url} target="_blank" rel="noopener noreferrer" className={s.voucherLink}>
                              {f.filename}
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action: Annuler / Supprimer */}
            {onDelete && ['pending', 'accepted', 'assigned', 'en_route'].includes(status) && (
              <div className={s.dangerZone}>
                <button
                  type="button"
                  className={s.deleteBtn}
                  onClick={() => onDelete(reservation)}
                >
                  <FiTrash2 size={13} />
                  {['en_route', 'assigned'].includes(status)
                    ? 'Annuler la course'
                    : 'Supprimer la reservation'}
                </button>
              </div>
            )}

            {/* IN_PROGRESS: annulation indisponible */}
            {status === 'in_progress' && (
              <div className={s.inProgressNotice}>
                <FiAlertCircle size={13} />
                Course en cours — annulation indisponible
              </div>
            )}

            {/* Return link */}
            {returnTo && (
              <button className={s.returnBtn} onClick={() => window.location.assign(returnTo)}>
                Retour au controle facturation
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ReservationDetailPanel;
