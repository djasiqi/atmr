// pages/institution/Requests/InstitutionRequestCreate.jsx
/**
 * REFERENCE UI — Institution Transport Request Form
 *
 * This form is the visual baseline for all institution-facing flows.
 * Any deviation from this design pattern must be justified.
 *
 * Design: Card-based layout, Lirie brand tokens, teal palette.
 * See: docs/brand/lirie-brand-guidelines.md
 *
 * Features:
 * - Card 1: Mission type, patient, route (FROM→TO visual), datetime + shortcuts
 * - Card 2: Contextual access details (conditional on trip type)
 * - Card 3: Needs chips, contact, billing
 * - Card 4: Advanced details (accordion)
 * - Footer: Summary + badges + CTA
 * - Pre-fill from patient, phone cleaning, auto-focus
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaSave, FaPaperPlane, FaTimes } from 'react-icons/fa';
import AsyncCreatableSelect from 'react-select/async-creatable';
import {
  useCreateRequest,
  useSendRequest,
  useInstitutionPatients,
  useInstitutionMe,
  useInstitutionSettings,
} from '../../../hooks/useInstitutionData';
import { listPatients } from '../../../services/institutionService';
import { canEditBilling } from '../../../utils/institutionPermissions';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import PatientFormModal from '../Patients/PatientFormModal';
import { toast } from 'sonner';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../components/ui/InlineTimePicker';
import styles from './InstitutionRequestForm.module.css';

const BILLING_INTENTS = [
  { value: 'patient', label: 'Patient' },
  { value: 'institution', label: 'Institution' },
  { value: 'insurance', label: 'Assurance' },
  { value: 'other', label: 'Autre' },
];

const TRIP_TYPES = [
  { value: 'inst_to_dest', label: 'Institution → Dest.', pickupType: 'institution', dropoffType: 'other' },
  { value: 'dom_to_dest', label: 'Domicile → Dest.', pickupType: 'domicile', dropoffType: 'other' },
  { value: 'return_home', label: 'Retour domicile', pickupType: 'institution', dropoffType: 'domicile' },
  { value: 'other', label: 'Autre', pickupType: 'other', dropoffType: 'other' },
];

const EMPTY_FORM = {
  mission_type: 'patient_transport',
  patient_id: '',
  external_reference: '',
  scheduled_time: '',
  scheduled_time_type: 'arrival',
  // Trip routing
  trip_type: 'inst_to_dest',
  pickup_type: 'institution',
  dropoff_type: 'other',
  pickup_location: '',
  pickup_floor: '',
  pickup_door_code: '',
  pickup_entry_point: '',
  pickup_instructions: '',
  dropoff_location: '',
  dropoff_floor: '',
  dropoff_door_code: '',
  dropoff_entry_point: '',
  dropoff_instructions: '',
  pickup_establishment: '',
  pickup_service: '',
  pickup_doctor: '',
  dropoff_establishment: '',
  dropoff_service: '',
  dropoff_doctor: '',
  floor_elevator_info: '',
  round_trip: true,
  return_time: '',
  return_date: '',
  return_hour: '',
  requires_wheelchair: false,
  requires_stretcher: false,
  requires_vehicle_wheelchair: false,
  requires_oxygen: false,
  requires_assistance: false,
  // Contacts
  requester_name: '',
  requester_phone: '',
  requester_service: '',
  onsite_is_different: false,
  onsite_name: '',
  onsite_phone: '',
  notes: '',
  delivery_description: '',
  billing_intent: 'patient',
  is_urgent: false,
};

const InstitutionRequestCreate = ({ onClose, onSuccess }) => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const isModal = typeof onClose === 'function';

  const { data: meData } = useInstitutionMe();
  const { data: patientsData } = useInstitutionPatients({ per_page: 100 });
  const { data: settingsData } = useInstitutionSettings();
  const createMutation = useCreateRequest();
  const sendMutation = useSendRequest();

  const institutionRole = meData?.institution_role;
  const canBilling = canEditBilling(institutionRole);
  const patientsItems = patientsData?.patients || patientsData?.items;
  const patients = useMemo(() => patientsItems || [], [patientsItems]);

  // Refs for focus management
  const destinationRef = useRef(null);
  const datetimeRef = useRef(null);

  // Form state
  const [formData, setFormData] = useState({ ...EMPTY_FORM });
  const [showQuickPatient, setShowQuickPatient] = useState(false);
  const [selectedPatientOption, setSelectedPatientOption] = useState(null);

  // ── Patient search (AsyncCreatableSelect) ──
  const formatDob = useCallback((dobStr) => {
    if (!dobStr) return '';
    try {
      const d = new Date(dobStr);
      if (isNaN(d.getTime())) return '';
      return d.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
    } catch { return ''; }
  }, []);

  const formatPatientOption = useCallback((p) => {
    const dobLabel = formatDob(p.dob);
    const parts = [`${p.last_name} ${p.first_name}`];
    if (dobLabel) parts.push(dobLabel);
    if (p.external_reference) parts.push(`DPI: ${p.external_reference}`);
    return {
      value: p.id,
      label: parts.join(' — '),
      raw: p,
    };
  }, [formatDob]);

  const defaultPatientOptions = useMemo(
    () => patients.map(formatPatientOption),
    [patients, formatPatientOption]
  );

  const loadPatientOptions = useCallback(async (inputValue) => {
    if (!inputValue || inputValue.length < 1) return defaultPatientOptions;
    try {
      const result = await listPatients({ query: inputValue, per_page: 30 });
      // Backend returns { patients: [...], total, page, ... }
      const items = result?.patients || result?.items || result?.data || [];
      const list = Array.isArray(items) ? items : [];
      return list.map(formatPatientOption);
    } catch {
      return [];
    }
  }, [formatPatientOption, defaultPatientOptions]);
  const [sendAfterCreate, setSendAfterCreate] = useState(true);
  const [patientPrefilled, setPatientPrefilled] = useState(false);

  // Selected patient object
  const selectedPatient = useMemo(() => {
    if (!formData.patient_id) return null;
    return patients.find(p => String(p.id) === String(formData.patient_id)) || null;
  }, [formData.patient_id, patients]);

  // Derived: institution address from meData
  const institutionAddress = meData?.address || '';

  // ── Pre-fill billing_intent + trip type + contacts from institution settings ──
  useEffect(() => {
    if (!settingsData) return;
    const settings = settingsData.settings || {};
    const defaultIntent = settings.default_billing_intent;

    setFormData(prev => {
      const updates = {};

      // Billing intent
      if (defaultIntent && BILLING_INTENTS.some(b => b.value === defaultIntent) && prev.billing_intent === 'patient') {
        updates.billing_intent = defaultIntent;
      }

      // Trip type from default_pickup_mode
      const mode = settings.default_pickup_mode || 'institution';
      if (prev.trip_type === 'inst_to_dest') {
        // Only override at init
        const defaultTrip = mode === 'domicile' ? 'dom_to_dest' : 'inst_to_dest';
        const def = TRIP_TYPES.find(t => t.value === defaultTrip);
        if (def) {
          updates.trip_type = defaultTrip;
          updates.pickup_type = def.pickupType;
          updates.dropoff_type = def.dropoffType;
        }
      }

      // Onsite phone pre-fill from default_contact_phone
      if (!prev.onsite_phone && settings.default_contact_phone) {
        updates.onsite_phone = settings.default_contact_phone;
      }

      // Pickup location pre-fill for institution mode
      if (updates.pickup_type === 'institution' || (!updates.pickup_type && prev.pickup_type === 'institution')) {
        if (!prev.pickup_location && institutionAddress) {
          updates.pickup_location = institutionAddress;
        }
      }

      if (Object.keys(updates).length > 0) {
        return { ...prev, ...updates };
      }
      return prev;
    });
  }, [settingsData, meData, institutionAddress]);

  // ── Pre-fill requester contact from connected user (independent of settingsData) ──
  useEffect(() => {
    if (!meData?.user) return;
    setFormData(prev => {
      const updates = {};
      if (!prev.requester_name) {
        const name = [meData.user.first_name, meData.user.last_name].filter(Boolean).join(' ');
        if (name) updates.requester_name = name;
      }
      if (!prev.requester_phone) {
        const phone = meData.user.phone || meData.contact_phone || '';
        if (phone) updates.requester_phone = phone;
      }
      if (Object.keys(updates).length > 0) {
        return { ...prev, ...updates };
      }
      return prev;
    });
  }, [meData]);

  // ── Pre-fill from patient data when patient is selected ──
  const prefillFromPatient = useCallback((patient) => {
    if (!patient) return;
    setFormData(prev => {
      const updates = {};
      // Build patient address
      const addressParts = [patient.address, patient.postal_code, patient.city]
        .filter(Boolean);
      const patientAddr = addressParts.length > 0 ? addressParts.join(', ') : '';

      // For domicile pickup: fill pickup with patient address
      if (prev.pickup_type === 'domicile' && patientAddr && !prev.pickup_location) {
        updates.pickup_location = patientAddr;
      }
      // For domicile dropoff (return_home): fill dropoff with patient address
      if (prev.dropoff_type === 'domicile' && patientAddr) {
        updates.dropoff_location = patientAddr;
      }
      // For "other" mode: fill pickup like before
      if (prev.pickup_type === 'other' && patientAddr && !prev.pickup_location) {
        updates.pickup_location = patientAddr;
      }

      // Logistics (domicile & other modes)
      if (prev.pickup_type !== 'institution') {
        if (patient.door_code && !prev.pickup_door_code) {
          updates.pickup_door_code = patient.door_code;
        }
        if (patient.floor && !prev.pickup_floor) {
          updates.pickup_floor = patient.floor;
        }
        const accessParts = [];
        if (patient.access_notes) accessParts.push(patient.access_notes);
        if (patient.residence_name) accessParts.push(`Résidence: ${patient.residence_name}`);
        if (accessParts.length > 0 && !prev.floor_elevator_info) {
          updates.floor_elevator_info = accessParts.join(' — ');
        }
      }
      if (!prev.external_reference && patient.external_reference) {
        updates.external_reference = patient.external_reference;
      }

      if (Object.keys(updates).length > 0) {
        return { ...prev, ...updates };
      }
      return prev;
    });
    setPatientPrefilled(true);
  }, []);

  const handlePatientChange = useCallback((patientId, patientObj) => {
    setFormData(prev => ({ ...prev, patient_id: patientId }));
    setPatientPrefilled(false);
    if (patientId) {
      const patient = patientObj || patients.find(p => String(p.id) === String(patientId));
      if (patient) {
        prefillFromPatient(patient);
      }
      // Auto-focus: destination if pickup is institution, else datetime
      setTimeout(() => {
        if (destinationRef.current) {
          destinationRef.current.focus();
        } else if (datetimeRef.current) {
          datetimeRef.current.focus();
        }
      }, 100);
    }
  }, [patients, prefillFromPatient]);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  // ── Trip type change handler ──
  const getPatientAddress = useCallback(() => {
    if (!selectedPatient) return '';
    const parts = [selectedPatient.address, selectedPatient.postal_code, selectedPatient.city].filter(Boolean);
    return parts.length > 0 ? parts.join(', ') : '';
  }, [selectedPatient]);

  const handleTripTypeChange = useCallback((tripTypeValue) => {
    const def = TRIP_TYPES.find(t => t.value === tripTypeValue);
    if (!def) return;
    setFormData(prev => {
      const updates = {
        trip_type: tripTypeValue,
        pickup_type: def.pickupType,
        dropoff_type: def.dropoffType,
      };
      // Auto-fill pickup for institution mode
      if (def.pickupType === 'institution' && institutionAddress) {
        updates.pickup_location = institutionAddress;
      }
      // Auto-fill pickup for domicile mode (patient address)
      if (def.pickupType === 'domicile') {
        const addr = getPatientAddress();
        if (addr) updates.pickup_location = addr;
      }
      // Auto-fill dropoff for domicile mode (return home — patient address)
      if (def.dropoffType === 'domicile') {
        const addr = getPatientAddress();
        if (addr) updates.dropoff_location = addr;
      }
      // Auto-fill dropoff for institution mode
      if (def.dropoffType === 'institution' && institutionAddress) {
        updates.dropoff_location = institutionAddress;
      }
      // Clear fields on "other" type
      if (def.dropoffType === 'other') {
        updates.dropoff_location = '';
      }
      if (def.pickupType === 'other') {
        updates.pickup_location = '';
      }
      // Clear irrelevant fields when switching
      if (def.pickupType === 'institution') {
        updates.pickup_floor = '';
        updates.pickup_door_code = '';
      }
      if (def.pickupType !== 'institution') {
        updates.pickup_entry_point = '';
        updates.pickup_instructions = '';
      }
      if (def.dropoffType === 'institution') {
        updates.dropoff_floor = '';
        updates.dropoff_door_code = '';
      }
      if (def.dropoffType !== 'institution') {
        updates.dropoff_entry_point = '';
        updates.dropoff_instructions = '';
      }
      return { ...prev, ...updates };
    });
  }, [institutionAddress, getPatientAddress]);

  // ── Datetime shortcuts ──
  const setTimeShortcut = useCallback((minutesFromNow) => {
    const d = new Date();
    d.setMinutes(d.getMinutes() + minutesFromNow);
    const pad = (n) => String(n).padStart(2, '0');
    const val = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    setFormData(prev => ({
      ...prev,
      scheduled_time: val,
      is_urgent: minutesFromNow === 0,
      ...(minutesFromNow === 0 ? { scheduled_time_type: 'departure' } : {}),
    }));
  }, []);

  const setTimeTomorrow9 = useCallback(() => {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    d.setHours(9, 0, 0, 0);
    const pad = (n) => String(n).padStart(2, '0');
    const val = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T09:00`;
    setFormData(prev => ({ ...prev, scheduled_time: val }));
  }, []);

  // ── Auto-fill return date (sans heure) quand A/R activé ──
  useEffect(() => {
    if (!formData.round_trip || !formData.scheduled_time) return;
    setFormData(prev => {
      if (prev.return_date) return prev; // ne pas écraser si déjà rempli
      const d = new Date(prev.scheduled_time);
      const pad = (n) => String(n).padStart(2, '0');
      return { ...prev, return_date: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` };
    });
  }, [formData.round_trip, formData.scheduled_time]);

  // ── Phone normalizer (accepts Swiss formats, cleans for storage) ──
  const cleanPhone = (raw) => {
    if (!raw) return '';
    // Remove spaces, dots, dashes, parens
    let cleaned = raw.replace(/[\s.\-()]/g, '');
    // Swiss local → international: 0xx → +41xx
    if (cleaned.startsWith('0') && !cleaned.startsWith('00')) {
      cleaned = '+41' + cleaned.slice(1);
    }
    return cleaned;
  };

  // ── Build payload for backend ──
  const buildPayload = () => {
    // Convention métier retour : sans return_hour → 00:00 = « heure retour à définir » (sentinelle)
    const returnHourPart = formData.return_hour?.trim() || '00:00';
    const payload = {
      mission_type: formData.mission_type,
      patient_id: formData.patient_id ? Number(formData.patient_id) : null,
      scheduled_time: new Date(formData.scheduled_time).toISOString(),
      scheduled_time_type: formData.scheduled_time_type || 'departure',
      pickup_location: formData.pickup_location || (formData.pickup_type === 'institution' ? institutionAddress : ''),
      dropoff_location: formData.dropoff_location || (formData.dropoff_type === 'institution' ? institutionAddress : ''),
      is_round_trip: formData.round_trip,
      is_urgent: formData.is_urgent || false,
      return_time: formData.round_trip && formData.return_date
        ? new Date(`${formData.return_date}T${returnHourPart}`).toISOString()
        : null,
      billing_intent: formData.billing_intent,
      notes: formData.notes || null,
    };
    if (formData.external_reference?.trim()) {
      payload.external_reference = formData.external_reference.trim();
    }

    // Logistics pickup
    if (formData.pickup_floor) payload.pickup_floor = formData.pickup_floor;
    if (formData.pickup_door_code) payload.pickup_door_code = formData.pickup_door_code;
    if (formData.dropoff_floor) payload.dropoff_floor = formData.dropoff_floor;
    if (formData.dropoff_door_code) payload.dropoff_door_code = formData.dropoff_door_code;
    if (formData.floor_elevator_info) payload.floor_elevator_info = formData.floor_elevator_info;

    // Location types and entry points (direct columns on TransportRequest)
    payload.pickup_type = formData.pickup_type || null;
    payload.dropoff_type = formData.dropoff_type || null;
    payload.pickup_entry_point = formData.pickup_entry_point || null;
    payload.dropoff_entry_point = formData.dropoff_entry_point || null;

    // Routing details also kept in billing_details.routing for rétrocompatibilité
    payload.billing_details = {
      ...(payload.billing_details || {}),
      routing: {
        pickup_type: formData.pickup_type,
        dropoff_type: formData.dropoff_type,
        pickup_entry_point: formData.pickup_entry_point || null,
        dropoff_entry_point: formData.dropoff_entry_point || null,
        pickup_instructions: formData.pickup_instructions || null,
        dropoff_instructions: formData.dropoff_instructions || null,
        pickup_establishment: formData.pickup_establishment || null,
        pickup_service: formData.pickup_service || null,
        pickup_doctor: formData.pickup_doctor || null,
        dropoff_establishment: formData.dropoff_establishment || null,
        dropoff_service: formData.dropoff_service || null,
        dropoff_doctor: formData.dropoff_doctor || null,
      },
    };

    // Mobility as structured JSONB
    payload.mobility = {
      wheelchair: formData.requires_wheelchair,
      stretcher: formData.requires_stretcher,
      vehicle_wheelchair: formData.requires_vehicle_wheelchair,
      oxygen: formData.requires_oxygen,
      needs_assistance: formData.requires_assistance,
      walking: !formData.requires_wheelchair && !formData.requires_stretcher,
    };

    // Contact on site (enriched structure, retrocompatible)
    const requesterName = formData.requester_name || 'Contact institution';
    const requesterPhone = cleanPhone(formData.requester_phone);
    const onsitePhone = cleanPhone(formData.onsite_phone);
    payload.contact_on_site = {
      // Retrocompat fields (used by existing code)
      name: formData.onsite_is_different ? (formData.onsite_name || requesterName) : requesterName,
      phone: formData.onsite_is_different ? (onsitePhone || requesterPhone) : requesterPhone,
      // Enriched fields
      requester_name: requesterName,
      requester_phone: requesterPhone,
      requester_service: formData.requester_service || null,
      onsite_is_different: formData.onsite_is_different,
      onsite_name: formData.onsite_is_different ? (formData.onsite_name || '') : '',
      onsite_phone: formData.onsite_is_different ? onsitePhone : '',
    };

    // Delivery
    if (formData.mission_type === 'material_delivery') {
      payload.delivery_description = formData.delivery_description;
    }

    return payload;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (createMutation.isPending || sendMutation.isPending) {
      return;
    }

    // Validation
    if (!formData.scheduled_time) {
      toast.error('Date et heure requises');
      return;
    }
    const effectivePickup = formData.pickup_location || (formData.pickup_type === 'institution' ? institutionAddress : '');
    const effectiveDropoff = formData.dropoff_location || (formData.dropoff_type === 'institution' ? institutionAddress : '');
    if (!effectivePickup || !effectiveDropoff) {
      toast.error('Adresses de départ et arrivée requises');
      return;
    }
    if (formData.mission_type === 'material_delivery' && !formData.delivery_description) {
      toast.error('Description de la livraison requise');
      return;
    }
    if (formData.requires_assistance && !formData.notes?.trim()) {
      toast.error('Décrivez le besoin d\'assistance (Pathologie / Difficultés)');
      return;
    }
    if (formData.round_trip && !formData.return_date) {
      toast.error('Date de retour requise pour un trajet aller-retour');
      return;
    }

    // Billing validation: bloquer si erreur critique et envoi immédiat
    if (sendAfterCreate && billingWarnings.some(w => w.level === 'error')) {
      toast.error('Corrigez les problèmes de facturation avant d\'envoyer la demande.');
      return;
    }

    try {
      const payload = buildPayload();
      const result = await createMutation.mutateAsync(payload);

      if (sendAfterCreate) {
        await sendMutation.mutateAsync({ requestId: result.id, options: {} });
        toast.success('Demande créée et envoyée');
      } else {
        toast.success('Demande créée en brouillon');
      }

      if (isModal && onSuccess) {
        onSuccess(result);
      } else {
        navigate(`/dashboard/institution/${public_id}/requests/${result.id}`);
      }
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur lors de la création');
    }
  };

  // ── Details indicator (count filled optional fields in accordion) ──
  const advancedFilledCount = useMemo(() => {
    let count = 0;
    if (formData.pickup_entry_point) count++;
    if (formData.pickup_instructions) count++;
    if (formData.dropoff_entry_point) count++;
    if (formData.dropoff_instructions) count++;
    if (formData.pickup_floor) count++;
    if (formData.pickup_door_code) count++;
    if (formData.floor_elevator_info) count++;
    if (formData.dropoff_floor) count++;
    if (formData.dropoff_door_code) count++;
    if (formData.requester_name) count++;
    if (formData.requester_phone) count++;
    if (formData.requester_service) count++;
    if (formData.onsite_name) count++;
    if (formData.onsite_phone) count++;
    if (formData.external_reference) count++;
    if (formData.notes) count++;
    return count;
  }, [formData.pickup_entry_point, formData.pickup_instructions, formData.dropoff_entry_point, formData.dropoff_instructions, formData.pickup_floor, formData.pickup_door_code, formData.floor_elevator_info, formData.dropoff_floor, formData.dropoff_door_code, formData.requester_name, formData.requester_phone, formData.requester_service, formData.onsite_name, formData.onsite_phone, formData.external_reference, formData.notes]);

  // ── Billing validation warnings ──
  const billingWarnings = useMemo(() => {
    const warnings = [];
    const intent = formData.billing_intent;

    if (intent === 'institution') {
      const instAddress = settingsData?.institution?.billing_address || meData?.address;
      if (!instAddress || !instAddress.trim()) {
        warnings.push({
          level: 'error',
          message: "Votre institution n'a pas d'adresse de facturation configurée. Le transporteur ne pourra pas facturer correctement.",
          action: 'Allez dans Paramètres > Facturation pour renseigner l\'adresse.',
        });
      }
    }

    if (intent === 'other' || intent === 'insurance') {
      // Tiers payeur sans billing_details (pas encore implémenté dans le form)
      warnings.push({
        level: 'info',
        message: 'Pensez à communiquer les coordonnées du tiers payeur au transporteur.',
      });
    }

    if (intent === 'patient' && selectedPatient) {
      const hasAddress = selectedPatient.address && selectedPatient.address.trim();
      if (!hasAddress) {
        warnings.push({
          level: 'warning',
          message: `Le patient ${selectedPatient.last_name} n'a pas d'adresse de domicile renseignée. La facture patient pourrait être incomplète.`,
        });
      }
    }

    return warnings;
  }, [formData.billing_intent, settingsData, meData, selectedPatient]);

  const handleClose = () => {
    if (isModal) {
      onClose();
    } else {
      navigate(`/dashboard/institution/${public_id}/requests`);
    }
  };

  return (
    <div className={`${styles.formWrapper} ${isModal ? styles.formWrapperModal : ''}`} data-tour-id="institution-request-create">
      <div className={styles.pageHeader}>
        <div className={styles.pageHeaderLeft}>
          {!isModal && (
            <button className={styles.backLink} onClick={handleClose}>
              <FaArrowLeft /> Demandes
            </button>
          )}
          <h1 className={styles.pageTitle}>Nouvelle demande</h1>
          <p className={styles.pageSubtitle}>Renseignez le trajet, ajoutez les détails si nécessaire.</p>
        </div>
        {isModal && (
          <button type="button" className={styles.closeBtn} onClick={handleClose} aria-label="Fermer">
            <FaTimes />
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className={styles.formOuter}>
        <div className={styles.form}>

        {/* ═══ COLONNE GAUCHE — Essentiel ═══ */}
        <div className={styles.columnLeft} data-tour-id="institution-request-form-left">

          {/* Patient selector + Mission type on same row */}
          <div className={styles.missionPatientRow}>
            {/* Patient selector (first = takes remaining space) */}
              <div data-tour-id="institution-request-patient">
              <label htmlFor="patient-select" className={styles.formLabel}>Patient</label>
              <AsyncCreatableSelect
                inputId="patient-select"
                aria-label="Patient"
                cacheOptions
                defaultOptions={defaultPatientOptions}
                loadOptions={loadPatientOptions}
                onChange={(option) => {
                  if (option) {
                    setSelectedPatientOption(option);
                    handlePatientChange(option.value, option.raw);
                  } else {
                    setSelectedPatientOption(null);
                    handlePatientChange('');
                  }
                }}
                onCreateOption={() => setShowQuickPatient(true)}
                value={selectedPatientOption}
                placeholder={formData.mission_type === 'material_delivery' ? 'Patient / destinataire (optionnel)…' : 'Nom, prénom ou date de naissance…'}
                formatCreateLabel={(input) => `+ Nouveau patient "${input}"`}
                noOptionsMessage={({ inputValue }) => inputValue ? 'Aucun patient trouvé' : 'Tapez pour rechercher…'}
                loadingMessage={() => 'Recherche…'}
                isClearable
                classNamePrefix="react-select"
                filterOption={() => true}
                formatOptionLabel={(option, { context }) => {
                  if (option.__isNew__) return option.label;
                  const p = option.raw;
                  if (!p) return option.label;
                  if (context === 'value') {
                    return `${p.last_name} ${p.first_name}`;
                  }
                  return (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontWeight: 500, color: '#1E293B' }}>
                        {p.last_name} {p.first_name}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: '#94A3B8', whiteSpace: 'nowrap' }}>
                        {formatDob(p.dob)}{p.external_reference ? ` · ${p.external_reference}` : ''}
                      </span>
                    </div>
                  );
                }}
                menuPortalTarget={typeof window !== 'undefined' ? document.body : null}
                menuPosition="fixed"
                styles={{ menuPortal: (base) => ({ ...base, zIndex: 'var(--z-modal-popover)' }) }}
              />
            </div>

            {/* Mission type segment (right side) */}
            <div className={styles.missionSegment}>
              <button type="button"
                className={`${styles.missionBtn} ${formData.mission_type === 'patient_transport' ? styles.missionBtnActive : ''}`}
                onClick={() => handleChange('mission_type', 'patient_transport')}>
                Patient
              </button>
              <button type="button"
                className={`${styles.missionBtn} ${formData.mission_type === 'material_delivery' ? styles.missionBtnActive : ''}`}
                onClick={() => handleChange('mission_type', 'material_delivery')}>
                Livraison
              </button>
            </div>
          </div>

          {selectedPatient && patientPrefilled && (
            <div className={styles.prefillTag}>
              Pré-rempli depuis {selectedPatient.first_name} {selectedPatient.last_name}
              <button type="button" className={styles.resetLink} onClick={() => { setFormData(prev => ({ ...prev, pickup_location: '', pickup_floor: '', pickup_door_code: '', floor_elevator_info: '' })); setPatientPrefilled(false); }}>Annuler</button>
            </div>
          )}


          {/* Trip type selector (segment control) */}
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Type de trajet</label>
            <div className={styles.tripSegment}>
              {TRIP_TYPES.map(tt => (
                <button key={tt.value} type="button"
                  className={`${styles.tripSegmentBtn} ${formData.trip_type === tt.value ? styles.tripSegmentBtnActive : ''}`}
                  onClick={() => handleTripTypeChange(tt.value)}>
                  {tt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Addresses with visual route */}
          <div className={styles.formGroup} data-tour-id="institution-request-destination">
            <div className={styles.routeBlock}>
              {/* Row 1: label Départ */}
              <label htmlFor="pickup_location" className={styles.routeLabel}>Départ</label>

              {/* Row 2: green dot + pickup input */}
              <span className={styles.routeDot} />
              {(formData.pickup_type === 'institution' || formData.pickup_type === 'domicile') ? (
                <input type="text" id="pickup_location"
                  value={formData.pickup_location || (formData.pickup_type === 'institution' ? institutionAddress : '')}
                  readOnly
                  className={`${styles.routeInput} ${styles.routeReadonly}`}
                  placeholder={formData.pickup_type === 'domicile' && !formData.pickup_location ? 'Sélectionnez un patient' : ''} />
              ) : (
                <AddressAutocomplete
                  name="pickup_location"
                  inputId="pickup_location"
                  value={formData.pickup_location}
                  onChange={(e) => handleChange('pickup_location', e.target.value)}
                  onSelect={(item) => {
                    const address = item.label || item.address || '';
                    const placeName = item.name || '';
                    const isDoctorPattern = /^(dr\.?|prof\.?|méd\.?|med\.?|docteur|professeur)\s/i;
                    setFormData(prev => {
                      const updates = { ...prev, pickup_location: address, pickup_establishment: '', pickup_doctor: '' };
                      if (placeName && placeName !== item.address) {
                        if (isDoctorPattern.test(placeName)) {
                          updates.pickup_doctor = placeName;
                        } else {
                          updates.pickup_establishment = placeName;
                        }
                      }
                      return updates;
                    });
                  }}
                  placeholder="Saisir ou choisir l'adresse"
                  inputClassName={styles.routeInput}
                  required
                />
              )}

              {/* Row 3: connector + label Destination */}
              <span className={styles.routeConnector} />
              <label htmlFor="dropoff_location" className={styles.routeLabel}>Destination</label>

              {/* Row 4: red dot + dropoff input */}
              <span className={`${styles.routeDot} ${styles.routeDotEnd}`} />
              {(formData.dropoff_type === 'institution' || formData.dropoff_type === 'domicile') ? (
                <input type="text" id="dropoff_location"
                  value={formData.dropoff_location || (formData.dropoff_type === 'institution' ? institutionAddress : '')}
                  readOnly
                  className={`${styles.routeInput} ${styles.routeReadonly}`}
                  placeholder={formData.dropoff_type === 'domicile' && !formData.dropoff_location ? 'Sélectionnez un patient' : ''} />
              ) : (
                <AddressAutocomplete
                  name="dropoff_location"
                  inputId="dropoff_location"
                  value={formData.dropoff_location}
                  onChange={(e) => handleChange('dropoff_location', e.target.value)}
                  onSelect={(item) => {
                    const address = item.label || item.address || '';
                    const placeName = item.name || '';
                    const isDoctorPattern = /^(dr\.?|prof\.?|méd\.?|med\.?|docteur|professeur)\s/i;
                    setFormData(prev => {
                      // Reset both fields, then fill the correct one
                      const updates = { ...prev, dropoff_location: address, dropoff_establishment: '', dropoff_doctor: '' };
                      if (placeName && placeName !== item.address) {
                        if (isDoctorPattern.test(placeName)) {
                          updates.dropoff_doctor = placeName;
                        } else {
                          updates.dropoff_establishment = placeName;
                        }
                      }
                      return updates;
                    });
                  }}
                  placeholder="Adresse d'arrivée"
                  inputClassName={styles.routeInput}
                  required
                />
              )}

              {/* Swap button (3rd column, spans all rows, centered) */}
              <button type="button" className={styles.swapBtn} title="Inverser"
                onClick={() => {
                  setFormData(prev => ({
                    ...prev,
                    pickup_location: prev.dropoff_location,
                    dropoff_location: prev.pickup_location,
                    pickup_type: prev.dropoff_type,
                    dropoff_type: prev.pickup_type,
                    pickup_entry_point: prev.dropoff_entry_point,
                    dropoff_entry_point: prev.pickup_entry_point,
                    pickup_instructions: prev.dropoff_instructions,
                    dropoff_instructions: prev.pickup_instructions,
                    pickup_floor: prev.dropoff_floor,
                    dropoff_floor: prev.pickup_floor,
                    pickup_door_code: prev.dropoff_door_code,
                    dropoff_door_code: prev.pickup_door_code,
                    pickup_establishment: prev.dropoff_establishment,
                    dropoff_establishment: prev.pickup_establishment,
                    pickup_service: prev.dropoff_service,
                    dropoff_service: prev.pickup_service,
                    pickup_doctor: prev.dropoff_doctor,
                    dropoff_doctor: prev.pickup_doctor,
                  }));
                }}>↕</button>
            </div>
          </div>

          {/* Datetime + time type toggle + AR, shortcuts below */}
          <div className={styles.formGroup}>
            <div className={styles.dateTimeLabelRow}>
              <label htmlFor="scheduled_time" className={styles.formLabel} style={{ margin: 0 }}>Date & heure *</label>
              <div className={styles.timeTypeRow}>
              <button
                type="button"
                className={`${styles.timeTypeBtn} ${formData.scheduled_time_type === 'departure' ? styles.timeTypeBtnActive : ''}`}
                onClick={() => handleChange('scheduled_time_type', 'departure')}
              >
                Départ
              </button>
              <button
                type="button"
                className={`${styles.timeTypeBtn} ${formData.scheduled_time_type === 'arrival' ? styles.timeTypeBtnActive : ''}`}
                onClick={() => handleChange('scheduled_time_type', 'arrival')}
              >
                Rendez-vous
              </button>
              </div>
            </div>

            <div className={styles.whenRow} data-tour-id="institution-request-datetime">
              <InlineDatePicker
                value={formData.scheduled_time ? formData.scheduled_time.split('T')[0] : ''}
                onChange={(dateVal) => {
                  const timePart = formData.scheduled_time?.split('T')[1] || '';
                  handleChange('scheduled_time', dateVal && timePart ? `${dateVal}T${timePart}` : dateVal ? `${dateVal}T` : '');
                }}
                placeholder="Date"
              />
              <InlineTimePicker
                value={formData.scheduled_time ? formData.scheduled_time.split('T')[1] || '' : ''}
                onChange={(timeVal) => {
                  const datePart = formData.scheduled_time?.split('T')[0] || '';
                  handleChange('scheduled_time', datePart && timeVal ? `${datePart}T${timeVal}` : datePart ? `${datePart}T${timeVal}` : '');
                }}
                placeholder="Heure"
              />
              <button type="button"
                className={`${styles.needsChip} ${formData.round_trip ? styles.needsChipActive : ''}`}
                aria-pressed={formData.round_trip}
                onClick={() => handleChange('round_trip', !formData.round_trip)}>
                ⇄ A/R
              </button>
            </div>

            <div className={styles.tripPills}>
              <button type="button" className={styles.whenShortcut} onClick={() => setTimeShortcut(0)} aria-label="Urgent"
                style={formData.is_urgent ? { borderColor: 'var(--danger)', background: '#FEE2E2', color: 'var(--danger)', fontWeight: 600 } : undefined}
                >🚨 Urgent</button>
              <button type="button" className={styles.whenShortcut} onClick={setTimeTomorrow9} aria-label="Demain 9h">Demain 9h</button>
            </div>
          </div>

          {formData.round_trip && (
            <div className={styles.formGroup}>
              <label className={styles.formLabel}>Retour</label>
              <p className={styles.billingHint} style={{ marginBottom: 6 }}>
                L&apos;heure de retour est facultative. Sans heure, le retour sera enregistré comme « à définir ».
              </p>
              <div style={{ display: 'flex', gap: 8 }}>
                <InlineDatePicker
                  value={formData.return_date}
                  onChange={(v) => handleChange('return_date', v)}
                  placeholder="Date retour"
                />
                <InlineTimePicker
                  value={formData.return_hour}
                  onChange={(v) => handleChange('return_hour', v)}
                  placeholder="HH:MM"
                />
              </div>
            </div>
          )}

          {/* Billing inline (left column, compact) */}
          <div className={styles.formGroup}>
            <label htmlFor="billing_intent" className={styles.formLabel}>Facturé à</label>
            <select id="billing_intent" value={formData.billing_intent} onChange={(e) => handleChange('billing_intent', e.target.value)}
              className={styles.inputFull} disabled={!canBilling}>
              {BILLING_INTENTS.map(opt => (<option key={opt.value} value={opt.value}>{opt.label}</option>))}
            </select>
            {!canBilling && <span className={styles.billingHint}>Géré par l'institution</span>}
          </div>

          {/* Contact demandeur */}
          <div className={styles.formGroup}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <label htmlFor="requester_name" className={styles.formLabel} style={{ margin: 0 }}>Contact</label>
              <button type="button"
                onClick={() => {
                  const next = !formData.onsite_is_different;
                  handleChange('onsite_is_different', next);
                  if (next) setFormData(prev => ({ ...prev, onsite_name: '', onsite_phone: '' }));
                }}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '3px 10px', border: '1px solid var(--border)', borderRadius: 'var(--radius-pill, 20px)',
                  background: formData.onsite_is_different ? 'var(--brand-light, #E0F2F1)' : 'transparent',
                  color: formData.onsite_is_different ? 'var(--brand, #00796B)' : 'var(--text-muted, #94A3B8)',
                  fontSize: '0.72rem', fontWeight: 500, cursor: 'pointer', lineHeight: 1,
                  transition: 'all 0.15s ease',
                }}>
                {formData.onsite_is_different ? '✕ Annuler' : '✎ Modifier'}
              </button>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {formData.onsite_is_different ? (
                <>
                  <input type="text" id="onsite_name" value={formData.onsite_name} onChange={(e) => handleChange('onsite_name', e.target.value)}
                    placeholder="Nom" className={styles.inputFull} style={{ flex: 1 }} />
                  <input type="tel" id="onsite_phone" value={formData.onsite_phone} onChange={(e) => handleChange('onsite_phone', e.target.value)}
                    placeholder="Tél. (+41...)" className={styles.inputFull} style={{ flex: 1 }} />
                  <input type="text" id="requester_service" value={formData.requester_service} onChange={(e) => handleChange('requester_service', e.target.value)}
                    placeholder="Service" className={styles.inputFull} style={{ flex: 1 }} />
                </>
              ) : (
                <>
                  <input type="text" id="requester_name" value={formData.requester_name} readOnly
                    placeholder="Nom" className={`${styles.inputFull} ${styles.readonlyInput}`} style={{ flex: 1 }} />
                  <input type="tel" id="requester_phone" value={formData.requester_phone} readOnly
                    placeholder="Tél." className={`${styles.inputFull} ${styles.readonlyInput}`} style={{ flex: 1 }} />
                  <input type="text" id="requester_service" value={formData.requester_service} onChange={(e) => handleChange('requester_service', e.target.value)}
                    placeholder="Service" className={styles.inputFull} style={{ flex: 1 }} />
                </>
              )}
            </div>
          </div>
        </div>

        {/* ═══ COLONNE DROITE — Détails & contexte ═══ */}
        <div className={styles.columnRight} data-tour-id="institution-request-form-tooltip">
          <div className={styles.detailsPanel}>

            {/* ═══ SECTION 1 — Infos départ ═══ */}
            <h2 className={styles.detailsPanelTitle}>📍 Départ</h2>

            {/* Départ institution → Service / Bâtiment */}
            {formData.pickup_type === 'institution' && (
              <div className={styles.detailsGroup}>
                <label htmlFor="pickup_entry_point" className={styles.detailsLabel}>Service / Bâtiment</label>
                <input type="text" list="entry_points_list" value={formData.pickup_entry_point}
                  onChange={(e) => handleChange('pickup_entry_point', e.target.value)}
                  placeholder="Ex: Réception, Bât. C, Étage 3" className={styles.detailsInput} id="pickup_entry_point" />
                <datalist id="entry_points_list">
                  {(settingsData?.settings?.entry_points || []).map((ep, i) => (<option key={i} value={ep} />))}
                </datalist>
              </div>
            )}

            {/* Départ externe (après swap ou type "autre") → Établissement / Service / Médecin */}
            {formData.pickup_type === 'other' && (
              <>
                <div className={styles.detailsGroup}>
                  <label htmlFor="pickup_establishment" className={styles.detailsLabel}>Établissement / Lieu</label>
                  <input type="text" id="pickup_establishment" value={formData.pickup_establishment || ''}
                    onChange={(e) => handleChange('pickup_establishment', e.target.value)}
                    placeholder="Ex: HUG, Clinique des Grangettes" className={styles.detailsInput} />
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="pickup_service" className={styles.detailsLabel}>Service</label>
                  <input type="text" id="pickup_service" value={formData.pickup_service || ''}
                    onChange={(e) => handleChange('pickup_service', e.target.value)}
                    placeholder="Ex: Radiologie, Urgences, Cardiologie" className={styles.detailsInput} />
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="pickup_doctor" className={styles.detailsLabel}>Médecin</label>
                  <input type="text" id="pickup_doctor" value={formData.pickup_doctor || ''}
                    onChange={(e) => handleChange('pickup_doctor', e.target.value)}
                    placeholder="Ex: Dr. Martin, Prof. Dupont" className={styles.detailsInput} />
                </div>
              </>
            )}

            {/* Départ domicile → Accès domicile */}
            {formData.pickup_type === 'domicile' && (
              <div className={styles.detailsGroup}>
                <label className={styles.detailsLabel}>Accès domicile</label>
                <div className={styles.detailsRow}>
                  <input type="text" id="pickup_floor" value={formData.pickup_floor} onChange={(e) => handleChange('pickup_floor', e.target.value)}
                    placeholder="Étage" className={styles.detailsInputSm} />
                  <input type="text" id="pickup_door_code" value={formData.pickup_door_code} onChange={(e) => handleChange('pickup_door_code', e.target.value)}
                    placeholder="Code porte" className={styles.detailsInputSm} />
                  <input type="text" id="pickup_entry_point_dom" value={formData.pickup_entry_point} onChange={(e) => handleChange('pickup_entry_point', e.target.value)}
                    placeholder="Accueil" className={styles.detailsInputSm} />
                </div>
                <input type="text" id="floor_elevator_info" value={formData.floor_elevator_info} onChange={(e) => handleChange('floor_elevator_info', e.target.value)}
                  placeholder="Ex: concierge, ascenseur gauche" className={styles.detailsInput} style={{ marginTop: 6 }} />
              </div>
            )}

            <hr className={styles.detailsDivider} />

            {/* ═══ SECTION 2 — Infos arrivée ═══ */}
            <h2 className={styles.detailsPanelTitle}>
              {formData.dropoff_type === 'domicile' ? '🏠 Arrivée — Domicile' : '🏥 Arrivée'}
            </h2>

            {/* Arrivée institution → Service / Bâtiment */}
            {formData.dropoff_type === 'institution' && (
              <div className={styles.detailsGroup}>
                <label htmlFor="dropoff_entry_point" className={styles.detailsLabel}>Service / Bâtiment</label>
                <input type="text" list="entry_points_list_dropoff" value={formData.dropoff_entry_point}
                  onChange={(e) => handleChange('dropoff_entry_point', e.target.value)}
                  placeholder="Ex: Réception, Bât. C, Étage 3" className={styles.detailsInput} id="dropoff_entry_point" />
                <datalist id="entry_points_list_dropoff">
                  {(settingsData?.settings?.entry_points || []).map((ep, i) => (<option key={i} value={ep} />))}
                </datalist>
              </div>
            )}

            {/* Arrivée externe (destination standard) → Établissement / Service / Médecin */}
            {formData.dropoff_type === 'other' && (
              <>
                <div className={styles.detailsGroup}>
                  <label htmlFor="dropoff_establishment" className={styles.detailsLabel}>Établissement / Lieu</label>
                  <input type="text" id="dropoff_establishment" value={formData.dropoff_establishment || ''}
                    onChange={(e) => handleChange('dropoff_establishment', e.target.value)}
                    placeholder="Ex: HUG, Clinique des Grangettes" className={styles.detailsInput} />
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="dropoff_service" className={styles.detailsLabel}>Service</label>
                  <input type="text" id="dropoff_service" value={formData.dropoff_service || ''}
                    onChange={(e) => handleChange('dropoff_service', e.target.value)}
                    placeholder="Ex: Radiologie, Urgences, Cardiologie" className={styles.detailsInput} />
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="dropoff_doctor" className={styles.detailsLabel}>Médecin</label>
                  <input type="text" id="dropoff_doctor" value={formData.dropoff_doctor || ''}
                    onChange={(e) => handleChange('dropoff_doctor', e.target.value)}
                    placeholder="Ex: Dr. Martin, Prof. Dupont" className={styles.detailsInput} />
                </div>
              </>
            )}

            {/* Arrivée domicile → Accès domicile */}
            {formData.dropoff_type === 'domicile' && (
              <>
                <div className={styles.detailsGroup}>
                  <label className={styles.detailsLabel}>Accès domicile</label>
                  <div className={styles.detailsRow}>
                    <input type="text" id="dropoff_floor" value={formData.dropoff_floor} onChange={(e) => handleChange('dropoff_floor', e.target.value)}
                      placeholder="Étage" className={styles.detailsInputSm} />
                    <input type="text" id="dropoff_door_code" value={formData.dropoff_door_code} onChange={(e) => handleChange('dropoff_door_code', e.target.value)}
                      placeholder="Code porte" className={styles.detailsInputSm} />
                    <input type="text" id="dropoff_entry_point_dom" value={formData.dropoff_entry_point} onChange={(e) => handleChange('dropoff_entry_point', e.target.value)}
                      placeholder="Accueil" className={styles.detailsInputSm} />
                  </div>
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="dropoff_instructions_dom" className={styles.detailsLabel}>Consignes arrivée</label>
                  <input type="text" id="dropoff_instructions_dom" value={formData.dropoff_instructions || ''}
                    onChange={(e) => handleChange('dropoff_instructions', e.target.value)}
                    placeholder="Ex: concierge, ascenseur gauche, sonner 2x" className={styles.detailsInput} />
                </div>
              </>
            )}


            <hr className={styles.detailsDivider} />

            {/* ═══ SECTION 3 — Infos patient ═══ */}
            <h2 className={styles.detailsPanelTitle}>👤 Patient & contact</h2>

            {formData.mission_type === 'patient_transport' && (
              <div className={styles.needsChips} style={{ marginBottom: 10, flexWrap: 'nowrap' }}>
                {/* Fauteuil / Prendre chaise — mutuellement exclusifs */}
                <button type="button" aria-pressed={formData.requires_wheelchair} style={{ flex: 1 }}
                  className={`${styles.needsChip} ${formData.requires_wheelchair ? styles.needsChipActive : ''}`}
                  onClick={() => setFormData(prev => ({ ...prev, requires_wheelchair: !prev.requires_wheelchair, requires_vehicle_wheelchair: false }))}>
                  ♿ Fauteuil
                </button>
                <button type="button" aria-pressed={formData.requires_vehicle_wheelchair} style={{ flex: 1 }}
                  className={`${styles.needsChip} ${formData.requires_vehicle_wheelchair ? styles.needsChipActive : ''}`}
                  onClick={() => setFormData(prev => ({ ...prev, requires_vehicle_wheelchair: !prev.requires_vehicle_wheelchair, requires_wheelchair: false }))}>
                  🏥 Prendre chaise
                </button>
                {/* Assistance — indépendant */}
                <button type="button" aria-pressed={formData.requires_assistance} style={{ flex: 1 }}
                  className={`${styles.needsChip} ${formData.requires_assistance ? styles.needsChipActive : ''}`}
                  onClick={() => handleChange('requires_assistance', !formData.requires_assistance)}>Assistance</button>
              </div>
            )}

            {formData.mission_type === 'material_delivery' ? (
              <div className={styles.detailsGroup}>
                <label htmlFor="delivery_description" className={styles.detailsLabel}>Description du matériel *</label>
                <textarea id="delivery_description" value={formData.delivery_description} onChange={(e) => handleChange('delivery_description', e.target.value)}
                  placeholder="Ex: lit médicalisé, fauteuil roulant" rows={2} required className={styles.detailsTextarea} />
              </div>
            ) : (
              <div className={styles.detailsGroup}>
                <label htmlFor="patient_notes" className={styles.detailsLabel}>
                  Pathologie / Difficultés {formData.requires_assistance && <span style={{ color: 'var(--danger, #e53935)', fontWeight: 600 }}>*</span>}
                </label>
                <textarea id="patient_notes" value={formData.notes} onChange={(e) => handleChange('notes', e.target.value)}
                  placeholder={formData.requires_assistance ? 'Obligatoire — décrivez le besoin d\'assistance…' : 'Ex: patient anxieux, mobilité réduite, sous perfusion…'}
                  rows={2} className={styles.detailsTextarea}
                  required={formData.requires_assistance}
                  style={formData.requires_assistance && !formData.notes ? { borderColor: 'var(--danger, #e53935)' } : undefined} />
              </div>
            )}

            <div className={styles.detailsGroup}>
              <label htmlFor="external_reference" className={styles.detailsLabel}>Réf. DPI / dossier</label>
              <input type="text" id="external_reference" value={formData.external_reference} onChange={(e) => handleChange('external_reference', e.target.value)}
                placeholder="Ex: 2024-12345" className={styles.detailsInput} />
            </div>


          </div>
        </div>
        </div>

        {/* ═══ Footer (ManualBookingForm pattern) ═══ */}
        <div className={styles.formFooter}>
          <div className={styles.footerSummary}>
            <span className={styles.footerSummaryText}>
              {selectedPatient ? `${selectedPatient.first_name} ${selectedPatient.last_name}` : 'Patient non sélectionné'}
              {formData.scheduled_time ? ` · ${new Date(formData.scheduled_time).toLocaleString('fr-CH', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}` : ''}
            </span>
            <span className={styles.footerBadges}>
              {formData.round_trip && <span className={styles.footerBadge}>A/R</span>}
              {sendAfterCreate && <span className={styles.footerBadge}>Envoi auto</span>}
              {advancedFilledCount > 0 && <span className={styles.footerBadge}>{advancedFilledCount} détail{advancedFilledCount > 1 ? 's' : ''}</span>}
            </span>
          </div>
          <div className={styles.footerBody}>
            <div className={styles.footerLeft}>
              <button type="button" className={styles.btnGhost} onClick={handleClose}>Annuler</button>
            </div>
            <div className={styles.footerRight}>
              <label className={styles.sendToggle}>
                <input type="checkbox" id="send_after_create" checked={sendAfterCreate} onChange={(e) => setSendAfterCreate(e.target.checked)} />
                <span>Envoyer aux transporteurs</span>
              </label>
              <button type="submit" className={styles.btnPrimary} disabled={createMutation.isPending || sendMutation.isPending} data-tour-id="institution-request-submit">
                {sendAfterCreate ? <><FaPaperPlane /> Envoyer</> : <><FaSave /> Enregistrer</>}
              </button>
            </div>
          </div>
        </div>
      </form>

      {/* ══════ Modal création patient (hors du <form> pour éviter form imbriqué) ══════ */}
      {showQuickPatient && (
        <PatientFormModal
          onClose={() => setShowQuickPatient(false)}
          onSaved={(patient) => {
            if (patient) {
              setSelectedPatientOption(formatPatientOption(patient));
              handlePatientChange(patient.id, patient);
              toast.success(`Patient ${patient.last_name} ${patient.first_name} ajouté`);
            }
          }}
        />
      )}


      
    </div>
  );
};

export default InstitutionRequestCreate;
