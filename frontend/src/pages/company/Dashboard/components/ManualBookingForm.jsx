// frontend/src/pages/company/Dashboard/components/ManualBookingForm.jsx (fixed)

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import NewClientModal from '../../Clients/components/NewClientModal';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import {
  createManualBooking,
  searchClients,
  createClient,
  fetchClientActiveStay,
  createClientStay,
  linkClientBillingParty,
} from '../../../../services/companyService';
import Input from './ui/Input';
import Label from './ui/Label';
import {
  useIsolatedField,
  IsolatedTextInput,
  IsolatedTextarea,
} from './ui/IsolatedFormField';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import apiClient from '../../../../utils/apiClient';
import { fetchBillingSettings, simulatePricing } from '../../../../services/settingsService';

// ⬇️ Assure-toi que ces chemins correspondent à ta structure réelle
import EstablishmentSelect from '../../../../components/common/EstablishmentSelect';
import ServiceSelect from '../../../../components/common/ServiceSelect';

import { extractMedicalServiceInfo } from '../../../../utils/medicalExtract';
import { shortAddress } from './formatAddress';
import { toast } from 'sonner';
import styles from './ManualBookingForm.module.css';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../../components/ui/InlineTimePicker';
import ManualBookingClientSelect from './ManualBookingClientSelect';
import { useBookingFormFocusGuard } from './useBookingFormFocusGuard';

/** Trim + vide → null. Réutilisable pour champs optionnels (medical_facility, hospital_service, notes_medical, etc.). */
function cleanOptionalText(s) {
  if (s == null) return null;
  const t = String(s).trim();
  return t === '' ? null : t;
}

const ensureIsoDatetimeWithSeconds = (value) => {
  if (!value || typeof value !== 'string') {
    return value;
  }

  const trimmed = value.trim();

  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(trimmed)) {
    return `${trimmed}:00`;
  }

  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(trimmed)) {
    return trimmed;
  }

  return trimmed;
};

// ⚡ Helper pour combiner date et time en datetime ISO 8601 (comme ensureIsoDatetimeWithSeconds)
const combineDateAndTime = (dateStr, timeStr) => {
  if (!dateStr || !dateStr.trim()) return undefined;
  if (!timeStr || !timeStr.trim()) return undefined;

  // 🔍 Debug
  console.log('[combineDateAndTime] Input - dateStr:', dateStr, 'timeStr:', timeStr);

  // Nettoyer dateStr : extraire uniquement YYYY-MM-DD (au cas où c'est déjà un datetime)
  const dateMatch = dateStr.trim().match(/^(\d{4}-\d{2}-\d{2})/);
  if (!dateMatch) {
    console.warn('[combineDateAndTime] Format de date invalide:', dateStr);
    return undefined;
  }
  const cleanDate = dateMatch[1]; // YYYY-MM-DD
  console.log('[combineDateAndTime] cleanDate:', cleanDate);

  // Nettoyer timeStr : extraire uniquement HH:mm (au cas où c'est déjà un datetime complet)
  // Supprimer toute date qui pourrait être présente dans timeStr
  let cleanTime = String(timeStr).trim();

  // ⚡ Si timeStr contient un 'T', c'est probablement un datetime complet
  // Extraire seulement la partie time après le dernier 'T'
  if (cleanTime.includes('T')) {
    // Utiliser lastIndexOf pour trouver le dernier 'T' et prendre tout ce qui suit
    const lastTIndex = cleanTime.lastIndexOf('T');
    if (lastTIndex !== -1 && lastTIndex < cleanTime.length - 1) {
      cleanTime = cleanTime.substring(lastTIndex + 1);
    }
  }

  // Extraire uniquement HH:mm (supprimer les secondes, millisecondes, timezone, etc.)
  // Regex pour extraire HH:mm même si d'autres éléments sont présents
  const timeExtractMatch = cleanTime.match(
    /^(\d{1,2}):(\d{2})(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/
  );
  if (timeExtractMatch) {
    const [, hours, minutes] = timeExtractMatch;
    // S'assurer que les heures et minutes sont valides
    const h = parseInt(hours, 10);
    const m = parseInt(minutes, 10);
    if (h >= 0 && h <= 23 && m >= 0 && m <= 59) {
      cleanTime = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
    } else {
      console.warn('[combineDateAndTime] Heure ou minutes invalides:', h, m);
      return undefined;
    }
  }

  console.log('[combineDateAndTime] cleanTime après nettoyage:', cleanTime);

  // Vérifier que cleanTime est au format HH:mm valide
  const timeMatch = cleanTime.match(/^(\d{2}):(\d{2})$/);
  if (!timeMatch) {
    console.warn(
      '[combineDateAndTime] Format de time invalide après nettoyage:',
      cleanTime,
      '(original:',
      timeStr,
      ')'
    );
    return undefined;
  }

  const [, hours, minutes] = timeMatch;

  // Format final : YYYY-MM-DDTHH:mm:00 (identique à ensureIsoDatetimeWithSeconds)
  const result = `${cleanDate}T${hours}:${minutes}:00`;
  console.log('[combineDateAndTime] Résultat final:', result);
  return result;
};

const SIM_CACHE_TTL_MS = 60 * 1000;
const SIM_DEBOUNCE_MS = 180;
const COORD_PRECISION = 5;

const roundCoord = (value) => {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Number(value).toFixed(COORD_PRECISION);
};

const isValidCoordPair = (coords) =>
  coords &&
  coords.lat != null &&
  coords.lon != null &&
  Number.isFinite(Number(coords.lat)) &&
  Number.isFinite(Number(coords.lon));

const isValidPreferentialAmount = (value) => {
  if (value == null) return false;
  const amount = Number(value);
  return Number.isFinite(amount) && amount > 0;
};

const toIsoWithTimezone = (isoLocalString) => {
  if (!isoLocalString) return undefined;
  const parsed = new Date(isoLocalString);
  if (Number.isNaN(parsed.getTime())) return undefined;
  return parsed.toISOString();
};

const pickupAtBucket = (value) => {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.length >= 16 && raw.includes('T')) {
    return raw.slice(0, 16);
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return raw;
  const pad = (v) => String(v).padStart(2, '0');
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}T${pad(
    parsed.getHours()
  )}:${pad(parsed.getMinutes())}`;
};

const routePointsSignature = (routePoints) => {
  if (!Array.isArray(routePoints) || routePoints.length < 2) return '';
  const first = routePoints[0] || {};
  const last = routePoints[routePoints.length - 1] || {};
  const fLat = roundCoord(first.lat);
  const fLon = roundCoord(first.lng ?? first.lon);
  const lLat = roundCoord(last.lat);
  const lLon = roundCoord(last.lng ?? last.lon);
  return `${routePoints.length}:${fLat || ''}:${fLon || ''}:${lLat || ''}:${lLon || ''}`;
};

export default function ManualBookingForm({ onSuccess, onClose, onSubmitStart }) {
  const queryClient = useQueryClient();
  const formRef = useRef(null);
  useBookingFormFocusGuard(formRef);

  // Swap pickup ↔ dropoff
  const handleSwapAddresses = () => {
    setPickupLocation(dropoffLocation);
    setDropoffLocation(pickupLocation);
    setPickupCoords(dropoffCoords);
    setDropoffCoords(pickupCoords);
  };

  // --- Établissement (objet choisi) + champ texte (saisie en cours)
  const [establishment, setEstablishment] = React.useState(null); // { id, label?, display_name?, address, ... }
  const [establishmentText, setEstablishmentText] = useState(''); // texte tapé
  const [serviceObj, setServiceObj] = React.useState(null); // { id, name }

  // --- Adresses + coordonnées
  const [pickupLocation, setPickupLocation] = useState('');
  const [dropoffLocation, setDropoffLocation] = useState('');
  const [pickupCoords, setPickupCoords] = useState({ lat: null, lon: null });
  const [dropoffCoords, setDropoffCoords] = useState({ lat: null, lon: null });
  const [estimatedDuration, setEstimatedDuration] = useState(null); // Durée estimée en minutes
  const [routePointsForPricing, setRoutePointsForPricing] = useState([]);

  // --- Client
  const [selectedClient, setSelectedClient] = useState(null);
  const [showClientModal, setShowClientModal] = useState(false);
  const [activeStay, setActiveStay] = useState(null); // Séjour actif avec infos clinique
  const [billToPatient, setBillToPatient] = useState(false); // Override: facturation patient

  // --- Date/heure de l'aller (séparés pour UX)
  const [scheduledDate, setScheduledDate] = useState('');
  const [scheduledHour, setScheduledHour] = useState('');

  // --- Aller-retour
  const [isRoundTrip, setIsRoundTrip] = useState(false);
  const [returnDate, setReturnDate] = useState(''); // Date de retour
  const [returnTime, setReturnTime] = useState(''); // Heure de retour (optionnel)

  const scheduledTime = scheduledDate && scheduledHour
    ? `${scheduledDate}T${scheduledHour}` : '';

  useEffect(() => {
    if (isRoundTrip && scheduledDate && !returnDate) {
      setReturnDate(scheduledDate);
    }
  }, [isRoundTrip, scheduledDate, returnDate]);

  useEffect(() => {
    if (!scheduledDate || !scheduledHour) return;

    const dateMatch = String(scheduledDate).match(/^(\d{4})-(\d{2})-(\d{2})$/);
    const timeMatch = String(scheduledHour).match(/^(\d{2}):(\d{2})$/);
    if (!dateMatch || !timeMatch) return;

    const [, y, m, d] = dateMatch;
    const [, hh, mm] = timeMatch;
    const selected = new Date(
      Number(y),
      Number(m) - 1,
      Number(d),
      Number(hh),
      Number(mm),
      0,
      0
    );
    if (Number.isNaN(selected.getTime())) return;

    const now = new Date();
    const isToday =
      Number(y) === now.getFullYear() &&
      Number(m) === now.getMonth() + 1 &&
      Number(d) === now.getDate();
    if (!isToday) return;

    if (selected.getTime() >= now.getTime()) return;

    const corrected = new Date(now.getTime() + 5 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    const correctedDate = `${corrected.getFullYear()}-${pad(corrected.getMonth() + 1)}-${pad(
      corrected.getDate()
    )}`;
    const correctedHour = `${pad(corrected.getHours())}:${pad(corrected.getMinutes())}`;

    // Si l'heure saisie est dans le passe pour aujourd'hui, repositionner automatiquement a maintenant + 5 min.
    if (scheduledDate !== correctedDate) setScheduledDate(correctedDate);
    if (scheduledHour !== correctedHour) setScheduledHour(correctedHour);
  }, [scheduledDate, scheduledHour]);

  // --- Livraison matériel
  const [isMaterialDelivery, setIsMaterialDelivery] = useState(false);
  const [deliveryDescription, setDeliveryDescription] = useState('');

  // --- Récurrence
  const [isRecurring, setIsRecurring] = useState(false);
  const [recurrenceType, setRecurrenceType] = useState('weekly'); // daily, weekly, custom
  const [recurrenceEndDate, setRecurrenceEndDate] = useState('');
  const [selectedDays, setSelectedDays] = useState([]); // Pour la récurrence personnalisée
  const [occurrences, setOccurrences] = useState(4); // Nombre d'occurrences

  // --- Infos médicales libres (toujours disponibles)
  const [medicalFacility, setMedicalFacility] = useState('');
  const {
    valueRef: doctorNameRef,
    sync: syncDoctorName,
    externalValue: doctorNameExternal,
  } = useIsolatedField('');
  const {
    valueRef: hospitalServiceRef,
    sync: syncHospitalService,
    externalValue: hospitalServiceExternal,
  } = useIsolatedField('');
  const {
    valueRef: notesMedicalRef,
    sync: syncNotesMedical,
    externalValue: notesMedicalExternal,
  } = useIsolatedField('');
  const {
    valueRef: pickupAccessNotesRef,
    externalValue: pickupAccessNotesExternal,
  } = useIsolatedField('');
  const {
    valueRef: dropoffAccessNotesRef,
    externalValue: dropoffAccessNotesExternal,
  } = useIsolatedField('');
  const [wheelchairOptions, setWheelchairOptions] = useState({
    clientHasWheelchair: false,
    needWheelchair: false,
  });


  const buildClinicAddress = useCallback((clinic) => {
    if (!clinic) return '';
    const parts = [];
    const hasStructured =
      clinic.domicile_address_line1 || clinic.domicile_zip || clinic.domicile_city;

    if (hasStructured && clinic.domicile_address_line1) {
      parts.push(clinic.domicile_address_line1);
      if (clinic.domicile_address_line2) {
        parts.push(clinic.domicile_address_line2);
      }
    } else if (clinic.address) {
      parts.push(clinic.address);
    }

    const postalCity = [clinic.domicile_zip, clinic.domicile_city].filter(Boolean).join(' ');
    const base = parts.join(', ');
    if (postalCity && !base.includes(postalCity)) {
      parts.push(postalCity);
    } else {
      if (clinic.domicile_zip && !base.includes(clinic.domicile_zip)) {
        parts.push(clinic.domicile_zip);
      }
      if (clinic.domicile_city && !base.includes(clinic.domicile_city)) {
        parts.push(clinic.domicile_city);
      }
    }

    return parts.filter(Boolean).join(', ');
  }, []);

  const getClinicPickupAddress = useCallback(
    (clinic) => {
      if (!clinic) return '';
      const structured = buildClinicAddress(clinic);
      if (structured) return structured;

      const directFallbacks = [
        clinic.address,
        clinic.domicile_address,
        clinic.billing_address,
        clinic.name,
      ];
      return directFallbacks.find((v) => typeof v === 'string' && v.trim())?.trim() || '';
    },
    [buildClinicAddress]
  );

  // Calculer la durée estimée en temps réel avec OSRM (routes réelles)
  React.useEffect(() => {
    const calculateDuration = async () => {
      if (pickupCoords.lat && pickupCoords.lon && dropoffCoords.lat && dropoffCoords.lon) {
        setRoutePointsForPricing([]);
        try {
          // ✅ Appel à l'API OSRM pour obtenir la durée réelle du trajet
          const response = await apiClient.get('/osrm/route', {
            params: {
              pickup_lat: pickupCoords.lat,
              pickup_lon: pickupCoords.lon,
              dropoff_lat: dropoffCoords.lat,
              dropoff_lon: dropoffCoords.lon,
            },
            timeout: 4000, // ⏱️ Timeout très court (4s) : fallback rapide si OSRM indisponible
          });

          if (response.data && response.data.duration) {
            // Convertir de secondes en minutes
            const durationMinutes = Math.round(response.data.duration / 60);
            const _distanceKm = (response.data.distance / 1000).toFixed(1);

            setEstimatedDuration(durationMinutes);
            const routePoints = Array.isArray(response.data.route)
              ? response.data.route
                  .filter((pair) => Array.isArray(pair) && pair.length >= 2)
                  .map((pair) => ({
                    lat: Number(pair[0]),
                    lng: Number(pair[1]),
                  }))
                  .filter((pt) => Number.isFinite(pt.lat) && Number.isFinite(pt.lng))
              : [];
            setRoutePointsForPricing(routePoints);
          } else {
            console.warn('⚠️ Pas de durée retournée par OSRM');
            setEstimatedDuration(null);
            setRoutePointsForPricing([]);
          }
        } catch (error) {
          // ⚡ Timeout OSRM est normal (fail-fast) → utiliser fallback silencieusement
          const isTimeout = error.code === 'ECONNABORTED' || error.message?.includes('timeout');
          if (isTimeout) {
            console.debug(
              '⏱️ OSRM timeout (comportement attendu), utilisation du fallback Haversine'
            );
          } else {
            console.warn('⚠️ Erreur calcul durée OSRM (non-timeout):', error.message || error);
          }

          // ⚠️ Fallback: Calcul Haversine approximatif en cas d'erreur OSRM
          try {
            const R = 6371; // Rayon de la Terre en km
            const dLat = ((dropoffCoords.lat - pickupCoords.lat) * Math.PI) / 180;
            const dLon = ((dropoffCoords.lon - pickupCoords.lon) * Math.PI) / 180;
            const a =
              Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos((pickupCoords.lat * Math.PI) / 180) *
                Math.cos((dropoffCoords.lat * Math.PI) / 180) *
                Math.sin(dLon / 2) *
                Math.sin(dLon / 2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
            const distanceKm = R * c;

            // Vitesse moyenne en ville : 30 km/h
            const durationMinutes = Math.round((distanceKm / 30) * 60);

            setEstimatedDuration(durationMinutes);
            setRoutePointsForPricing([]);
            if (!isTimeout) {
              // Seulement afficher le warning pour les vraies erreurs (pas les timeouts)
              console.debug(`📍 Durée estimée (Haversine) : ${durationMinutes} min`);
            }
          } catch (fallbackError) {
            console.error('❌ Erreur fallback Haversine:', fallbackError);
            setEstimatedDuration(null);
            setRoutePointsForPricing([]);
          }
        }
      } else {
        setEstimatedDuration(null);
        setRoutePointsForPricing([]);
      }
    };

    calculateDuration();
  }, [pickupCoords.lat, pickupCoords.lon, dropoffCoords.lat, dropoffCoords.lon]);

  // Helper: min pour <input type="datetime-local"> au format local (pas UTC)
  const _minLocalDatetime = (() => {
    const d = new Date(Date.now() + 5 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
      d.getHours()
    )}:${pad(d.getMinutes())}`;
  })();

  const applyDateTimePreset = (preset) => {
    const pad = (n) => String(n).padStart(2, '0');
    const now = new Date();
    let target;
    switch (preset) {
      case 'now30':
        target = new Date(now.getTime() + 30 * 60 * 1000);
        break;
      case 'now1h':
        target = new Date(now.getTime() + 60 * 60 * 1000);
        break;
      case 'tomorrow9':
        target = new Date(now);
        target.setDate(target.getDate() + 1);
        target.setHours(9, 0, 0, 0);
        break;
      default:
        return;
    }
    setScheduledDate(`${target.getFullYear()}-${pad(target.getMonth() + 1)}-${pad(target.getDate())}`);
    setScheduledHour(`${pad(target.getHours())}:${pad(target.getMinutes())}`);
  };

  // === Handlers établissement / service ===
  const handleEstablishmentTextChange = (newText) => {
    setEstablishmentText(newText);
    // Si l'utilisateur modifie le texte ou repasse en saisie libre, on efface l’établissement et le service
    if (establishment && newText !== (establishment.label || establishment.display_name || '')) {
      setEstablishment(null);
      setServiceObj(null);
      syncHospitalService('');
    }
  };

  // Verrouillage : si établissement devient null (désélection / autre chemin), ne pas garder un ancien service
  React.useEffect(() => {
    if (!establishment) {
      setServiceObj(null);
      syncHospitalService('');
    }
  }, [establishment]);

  const onPickEstablishment = (estab) => {
    console.log('🏥 ManualBookingForm.onPickEstablishment:', estab);
    setEstablishment(estab);
    setEstablishmentText(estab?.label || estab?.display_name || '');

    // Toujours réinitialiser le service (sélection ou désélection → pas d’ancienne valeur gardée)
    setServiceObj(null);
    syncHospitalService('');

    if (estab) console.log('🏥 Establishment set, ID:', estab?.id, 'Type:', typeof estab?.id);
  };

  const onChangeService = useCallback(
    (srv) => {
      console.log('🏥 ManualBookingForm.onChangeService:', srv);
      setServiceObj(srv || null);
      syncHospitalService(srv?.name || '');

      // --- Logique pour la destination ---
      if (srv && establishment) {
        setDropoffLocation(establishment.address || '');
        setDropoffCoords({
          lat: establishment.lat ?? null,
          lon: establishment.lon ?? null,
        });
      }

      // --- NOUVEAU : Ajoute les détails du service dans les notes ---
      if (srv) {
        const details = [srv.building, srv.site_note, srv.floor, srv.phone].filter(Boolean); // Garde uniquement les valeurs qui existent

        if (details.length > 0) {
          const detailsText = details.join(' - ');
          // Ajoute les détails aux notes existantes (s'il y en a)
          syncNotesMedical((prevNotes) =>
            prevNotes ? `${prevNotes}\n${detailsText}` : detailsText
          );
        }
      }
    },
    [establishment]
  );

  // === State montant + pricing auto ===
  const [amount, setAmount] = useState('');
  const [amountSource, setAmountSource] = useState(null); // preferential | simulated | manual
  const [amountLocked, setAmountLocked] = useState(false);
  const [lastPricingUpdateAt, setLastPricingUpdateAt] = useState(null);
  const [pricingWarning, setPricingWarning] = useState('');
  const [isSimulatingPricing, setIsSimulatingPricing] = useState(false);
  const [activePricingProfileId, setActivePricingProfileId] = useState(null);
  const [activePricingProfileVersionId, setActivePricingProfileVersionId] = useState(null);

  const suppressManualAmountRef = useRef(false);
  const simulateRequestSeqRef = useRef(0);
  const simulateKeyRef = useRef('');
  const simulateAbortRef = useRef(null);
  const simulateDebounceRef = useRef(null);
  const simCacheRef = useRef(new Map());

  const applySystemAmount = useCallback((value, source) => {
    suppressManualAmountRef.current = true;
    setAmount(value != null ? String(value) : '');
    setAmountSource(source || null);
    setLastPricingUpdateAt(new Date());
    queueMicrotask(() => {
      suppressManualAmountRef.current = false;
    });
  }, []);

  // === Gestion des jours de la semaine pour récurrence ===
  // ⚠️ IDs correspondent à Python weekday() : 0=Lundi, 1=Mardi, etc.
  const weekDays = [
    { id: 0, label: 'Lundi', short: 'L' },
    { id: 1, label: 'Mardi', short: 'Ma' },
    { id: 2, label: 'Mercredi', short: 'Me' },
    { id: 3, label: 'Jeudi', short: 'J' },
    { id: 4, label: 'Vendredi', short: 'V' },
    { id: 5, label: 'Samedi', short: 'S' },
    { id: 6, label: 'Dimanche', short: 'D' },
  ];

  const toggleDay = (dayId) => {
    setSelectedDays((prev) =>
      prev.includes(dayId) ? prev.filter((d) => d !== dayId) : [...prev, dayId]
    );
  };

  // === Charger les clients par défaut au montage ===
  const [defaultClientOptions, setDefaultClientOptions] = useState([]);

  React.useEffect(() => {
    const loadDefaultClients = async () => {
      try {
        console.log('📥 Chargement des clients par défaut...');
        const clients = await searchClients('');
        console.log('✅ Clients chargés:', clients);

        const options = clients.map((c) => {
          // 🏥 Pour les institutions, afficher le nom de l'institution
          let label = `Client #${c.id}`; // Par défaut

          if (c.is_institution && c.institution_name) {
            label = `🏥 ${c.institution_name}`;
          } else {
            // Pour les clients normaux, utiliser full_name, first_name et last_name
            // (le backend retourne ces champs depuis Client.serialize)
            if (c.full_name && c.full_name !== 'Nom non renseigné') {
              label = c.full_name;
            } else {
              const firstName = c.first_name || '';
              const lastName = c.last_name || '';
              
              if (firstName || lastName) {
                label = `${firstName} ${lastName}`.trim();
              }
            }
          }

          return {
            value: c.id,
            label: label,
            raw: c,
          };
        });

        setDefaultClientOptions(options);
        console.log('📋 Options par défaut disponibles:', options.length);
      } catch (error) {
        console.error('❌ Erreur chargement clients par défaut:', error);
      }
    };

    loadDefaultClients();
  }, []);

  const pricingTriggerKey = useMemo(() => {
    if (!isValidCoordPair(pickupCoords) || !isValidCoordPair(dropoffCoords)) {
      return '';
    }
    const pickupLat = roundCoord(pickupCoords.lat);
    const pickupLon = roundCoord(pickupCoords.lon);
    const dropoffLat = roundCoord(dropoffCoords.lat);
    const dropoffLon = roundCoord(dropoffCoords.lon);
    if (!pickupLat || !pickupLon || !dropoffLat || !dropoffLon) {
      return '';
    }
    return [
      pickupLat,
      pickupLon,
      dropoffLat,
      dropoffLon,
      String(Boolean(isRoundTrip)),
      String(activePricingProfileVersionId || ''),
      pickupAtBucket(scheduledTime),
      routePointsSignature(routePointsForPricing),
    ].join('|');
  }, [
    pickupCoords,
    dropoffCoords,
    isRoundTrip,
    activePricingProfileVersionId,
    scheduledTime,
    routePointsForPricing,
  ]);

  const getPreferentialAmount = useCallback(
    ({ client, stay, patientBillingOverride }) => {
      const clinicRate = stay?.clinic?.preferential_rate;
      if (!patientBillingOverride && isValidPreferentialAmount(clinicRate)) {
        return { amount: Number(clinicRate).toFixed(2), source: 'preferential' };
      }
      const clientRate = client?.preferential_rate;
      if (isValidPreferentialAmount(clientRate)) {
        return { amount: Number(clientRate).toFixed(2), source: 'preferential' };
      }
      return null;
    },
    []
  );

  const runPricingSimulation = useCallback(async () => {
    if (!pricingTriggerKey || amountLocked || amountSource === 'preferential') {
      return;
    }
    if (!activePricingProfileVersionId) {
      setPricingWarning('Profil tarifaire actif introuvable. Saisissez un montant ou contactez le support.');
      return;
    }

    const cached = simCacheRef.current.get(pricingTriggerKey);
    if (cached && Date.now() - cached.cachedAt <= SIM_CACHE_TTL_MS) {
      if (!amountLocked && amountSource !== 'preferential' && simulateKeyRef.current === pricingTriggerKey) {
        applySystemAmount(Number(cached.amount).toFixed(2), 'simulated');
        setPricingWarning(cached.warning || '');
      }
      return;
    }

    setIsSimulatingPricing(true);
    setPricingWarning('');
    simulateRequestSeqRef.current += 1;
    const requestSeq = simulateRequestSeqRef.current;
    simulateKeyRef.current = pricingTriggerKey;
    if (simulateAbortRef.current) {
      simulateAbortRef.current.abort();
    }
    const abortController = new AbortController();
    simulateAbortRef.current = abortController;

    const payload = {
      pricing_profile_version_id: activePricingProfileVersionId,
      booking: {
        pickup_at: toIsoWithTimezone(ensureIsoDatetimeWithSeconds(scheduledTime)) || new Date().toISOString(),
        is_round_trip: Boolean(isRoundTrip),
        pickup_lat: Number(pickupCoords.lat),
        pickup_lng: Number(pickupCoords.lon),
        dropoff_lat: Number(dropoffCoords.lat),
        dropoff_lng: Number(dropoffCoords.lon),
        route_points: Array.isArray(routePointsForPricing) && routePointsForPricing.length > 1
          ? routePointsForPricing
          : undefined,
      },
    };

    try {
      const response = await simulatePricing(payload, { signal: abortController.signal });
      const warningList = Array.isArray(response?.warnings)
        ? response.warnings
        : Array.isArray(response?.breakdown?.warnings)
          ? response.breakdown.warnings
          : [];
      const blockingReasons = Array.isArray(response?.blocking_reasons)
        ? response.blocking_reasons
        : [];
      const confidence = String(response?.confidence || '').toLowerCase();
      const hasDistanceUnavailable = warningList.includes('distance_unavailable');
      const hasZoneUnresolved = warningList.includes('zone_unresolved');
      const modelUsed = String(response?.breakdown?.model_used || '').toLowerCase();
      const isDistanceDependentModel = modelUsed === 'distance' || modelUsed === 'hybrid_stack';
      const responseAmount = Number(response?.amount);

      if (
        requestSeq !== simulateRequestSeqRef.current ||
        pricingTriggerKey !== simulateKeyRef.current ||
        amountLocked ||
        amountSource === 'preferential'
      ) {
        return;
      }

      if ((confidence === 'blocked' || blockingReasons.length > 0) && !Number.isFinite(responseAmount)) {
        if (blockingReasons.includes('zone_unresolved')) {
          setPricingWarning('Calcul indisponible: zonage introuvable pour ce trajet. Saisissez un montant ou réessayez.');
          return;
        }
        if (blockingReasons.includes('zone_unresolved_timeout')) {
          setPricingWarning('Calcul indisponible: délai de calcul zonage dépassé. Réessayez dans quelques secondes.');
          return;
        }
        if (blockingReasons.includes('distance_unavailable')) {
          setPricingWarning('Calcul indisponible (OSRM). Saisissez un montant ou réessayez.');
          return;
        }
        setPricingWarning('Calcul précis temporairement indisponible. Réessayez.');
        return;
      }

      if (hasDistanceUnavailable && isDistanceDependentModel) {
        setPricingWarning('Calcul indisponible (OSRM). Saisissez un montant ou réessayez.');
        return;
      }

      if (hasZoneUnresolved && !Number.isFinite(responseAmount)) {
        setPricingWarning('Calcul indisponible: zonage introuvable pour ce trajet. Saisissez un montant ou réessayez.');
        return;
      }

      if (Number.isFinite(responseAmount) && responseAmount > 0) {
        applySystemAmount(responseAmount.toFixed(2), 'simulated');
        if (warningList.includes('zone_unresolved_fallback')) {
          setPricingWarning('Zonage partiellement résolu: calcul appliqué avec fallback conservateur.');
        } else {
            setPricingWarning('');
        }
        simCacheRef.current.set(pricingTriggerKey, {
          amount: responseAmount,
          warning: warningList.includes('zone_unresolved_fallback')
            ? 'Zonage partiellement résolu: calcul appliqué avec fallback conservateur.'
            : '',
          cachedAt: Date.now(),
        });
      }
    } catch (error) {
      if (error?.name !== 'CanceledError' && error?.name !== 'AbortError') {
        const apiWarnings = Array.isArray(error?.response?.data?.warnings)
          ? error.response.data.warnings
          : [];
        const blockingReasons = Array.isArray(error?.response?.data?.blocking_reasons)
          ? error.response.data.blocking_reasons
          : [];
        const isDistanceBlocking = apiWarnings.includes('distance_unavailable')
          || blockingReasons.includes('distance_unavailable');
        if (isDistanceBlocking) {
          setPricingWarning('Calcul indisponible (OSRM). Saisissez un montant ou réessayez.');
        } else {
          setPricingWarning('Calcul temporairement indisponible. Saisissez un montant ou réessayez.');
        }
      }
    } finally {
      if (requestSeq === simulateRequestSeqRef.current) {
        setIsSimulatingPricing(false);
      }
    }
  }, [
    pricingTriggerKey,
    amountLocked,
    amountSource,
    activePricingProfileVersionId,
    isRoundTrip,
    pickupCoords,
    dropoffCoords,
    routePointsForPricing,
    scheduledTime,
    applySystemAmount,
  ]);

  // === Clients ===
  const handleSelectClient = useCallback(async (clientObj) => {
    console.log('👤 Client sélectionné:', clientObj);
    setSelectedClient(clientObj);
    setActiveStay(null); // Réinitialiser le séjour actif
    setBillToPatient(false); // Réinitialiser l'override
    setAmountLocked(false);
    setAmountSource(null);
    setAmount('');
    setPricingWarning('');

    const client = clientObj?.raw;
    const clientId = client?.id || clientObj?.value?.id;

    if (!clientId) {
      console.warn('⚠️ Pas d\'ID client disponible');
      return;
    }

    // 🏥 Vérifier si le client a un séjour actif
    try {
      const stayResponse = await fetchClientActiveStay(clientId);
      const stayData = stayResponse?.data;

      if (stayData && stayData.clinic) {
        setActiveStay(stayData);
        const clinic = stayData.clinic;
        // Préremplir l'établissement médical avec la clinique d'hospitalisation
        if (clinic?.name) {
          setMedicalFacility(clinic.name);
          setEstablishmentText(clinic.name);
          setEstablishment(null);
          setServiceObj(null);
          syncHospitalService('');
        }

        // Set pickup to clinic address (structured > raw > clinic name as last resort)
        const clinicAddress = getClinicPickupAddress(clinic);
        if (clinicAddress) {
          setPickupLocation(clinicAddress);
          setPickupCoords({
            lat: clinic.latitude || null,
            lon: clinic.longitude || null,
          });
          console.log(`🏥 Client hospitalisé: utilisation adresse clinique ${clinic.name}`);
          console.log(`📍 Adresse clinique: ${clinicAddress}`);
        } else {
          console.warn('⚠️ Aucune adresse disponible pour la clinique:', clinic.name);
        }

        return;
      }
    } catch (error) {
      console.warn('⚠️ Erreur lors de la récupération du séjour actif:', error);
      // Continuer avec l'adresse du client en cas d'erreur
    }

    // Pas de séjour actif ou erreur → utiliser l'adresse du client
    // 📍 Récupérer l'adresse exacte du client avec priorités
    let homeAddress = '';
    let homeGPS = { lat: null, lon: null };

    // Priorité 1: domicile (adresse structurée complète + GPS)
    if (client?.domicile?.address) {
      // Construire l'adresse complète : Rue, Numéro, Code postal, Ville
      const parts = [client.domicile.address, client.domicile.zip, client.domicile.city].filter(
        Boolean
      );
      homeAddress = parts.join(', ');

      // 📍 IMPORTANT: Charger aussi les GPS du domicile !
      if (client.domicile.lat && client.domicile.lon) {
        homeGPS = {
          lat: client.domicile.lat,
          lon: client.domicile.lon,
        };
        console.log(`📍 GPS du domicile chargés: ${homeGPS.lat}, ${homeGPS.lon}`);
      }
    }
    // Priorité 2: billing_address (adresse de facturation)
    else if (client?.billing_address) {
      homeAddress = client.billing_address;

      // Charger GPS de facturation si disponibles
      if (client.billing_lat && client.billing_lon) {
        homeGPS = {
          lat: client.billing_lat,
          lon: client.billing_lon,
        };
        console.log(`📍 GPS de facturation chargés: ${homeGPS.lat}, ${homeGPS.lon}`);
      }
    }
    // Priorité 3: adresse utilisateur (peut être un nom de résidence)
    else if (client?.user?.address) {
      homeAddress = client.user.address;
    }
    // Priorité 4: adresse client directe
    else if (client?.address) {
      homeAddress = client.address;
    }

    if (homeAddress) {
      setPickupLocation(homeAddress);
      setPickupCoords(homeGPS); // ✅ Charger les GPS du client
      console.log(`📍 Adresse du client: ${homeAddress}`);
    }

  }, [getClinicPickupAddress, syncHospitalService]);

  const handleCreateClientOption = useCallback(() => {
    setShowClientModal(true);
  }, []);

  const handleBillToPatientChange = useCallback((checked) => {
    setBillToPatient(checked);
  }, []);

  const loadClientOptions = useCallback(async (q) => {
    try {
      console.log('🔍 Recherche de clients avec query:', q);

      // Si pas de recherche, charger tous les clients (limité par le backend)
      const clients = q ? await searchClients(q) : await searchClients('');

      console.log('✅ Clients trouvés:', clients.length);

      const options = clients.map((c) => {
        // 🏥 Pour les institutions, afficher le nom de l'institution
        let label = `Client #${c.id}`; // Par défaut

        if (c.is_institution && c.institution_name) {
          label = `🏥 ${c.institution_name}`;
        } else {
          // Pour les clients normaux, utiliser full_name, first_name et last_name
          // (le backend retourne ces champs depuis Client.serialize)
          if (c.full_name && c.full_name !== 'Nom non renseigné') {
            label = c.full_name;
          } else {
            const firstName = c.first_name || '';
            const lastName = c.last_name || '';
            
            if (firstName || lastName) {
              label = `${firstName} ${lastName}`.trim();
            }
          }
        }

        return {
          value: c.id,
          label: label,
          raw: c,
        };
      });

      console.log('📋 Options formatées:', options);
      return options;
    } catch (e) {
      console.error('❌ searchClients error', e);
      return [];
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    const loadBillingContext = async () => {
      try {
        const response = await fetchBillingSettings();
        const payload = response?.data || response || {};
        if (!mounted) return;
        setActivePricingProfileId(payload.active_pricing_profile_id || null);
        setActivePricingProfileVersionId(payload.active_pricing_profile_version_id || null);
        if (!payload.active_pricing_profile_version_id) {
          setPricingWarning('Profil tarifaire actif introuvable. Le montant restera manuel.');
        }
      } catch (error) {
        console.warn('[ManualBookingForm] Impossible de charger la version pricing active', error);
      }
    };
    loadBillingContext();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!activeStay?.clinic || billToPatient) {
      return;
    }
    const clinic = activeStay.clinic;
    const clinicAddress = getClinicPickupAddress(clinic);
    if (!clinicAddress) {
      return;
    }
    setPickupLocation(clinicAddress);
    setPickupCoords({
      lat: clinic.latitude || null,
      lon: clinic.longitude || null,
    });
  }, [activeStay, billToPatient, getClinicPickupAddress]);

  useEffect(() => {
    const preferential = getPreferentialAmount({
      client: selectedClient?.raw || null,
      stay: activeStay,
      patientBillingOverride: billToPatient,
    });
    if (preferential) {
      if (!amountLocked) {
        applySystemAmount(preferential.amount, preferential.source);
        setPricingWarning('');
      }
      return;
    }
    if (!amountLocked && amountSource === 'preferential') {
      applySystemAmount('', null);
    }
  }, [
    selectedClient,
    activeStay,
    billToPatient,
    amountLocked,
    amountSource,
    getPreferentialAmount,
    applySystemAmount,
  ]);

  useEffect(() => {
    if (simulateDebounceRef.current) {
      clearTimeout(simulateDebounceRef.current);
      simulateDebounceRef.current = null;
    }
    if (!pricingTriggerKey || amountLocked || amountSource === 'preferential') {
      return;
    }
    simulateDebounceRef.current = setTimeout(() => {
      runPricingSimulation();
    }, SIM_DEBOUNCE_MS);
    return () => {
      if (simulateDebounceRef.current) {
        clearTimeout(simulateDebounceRef.current);
      }
    };
  }, [pricingTriggerKey, amountLocked, amountSource, runPricingSimulation]);

  useEffect(
    () => () => {
      if (simulateAbortRef.current) {
        simulateAbortRef.current.abort();
      }
      if (simulateDebounceRef.current) {
        clearTimeout(simulateDebounceRef.current);
      }
    },
    []
  );

  // === Notes médicales → extraction
  function handleNotesMedicalBlur(e) {
    const value = e.target.value;
    // ❌ Supprimé : syncNotesMedical(value) car déjà géré par onChange

    const extracted = extractMedicalServiceInfo(value);

    if (extracted.medical_facility) setMedicalFacility(extracted.medical_facility);
    if (extracted.hospital_service) {
      syncHospitalService(extracted.hospital_service);
      setServiceObj(null); // pas d'ID → on laisse vide ; l'utilisateur choisira dans la liste
    }
    if (extracted.doctor_name) syncDoctorName(extracted.doctor_name);
    
    // Helper pour vérifier si un texte existe déjà dans les notes (normalisé)
    const hasTextInNotes = (notes, text) => {
      if (!notes || !text) return false;
      const normalize = (t) => t.toLowerCase().replace(/[🏢\s]/g, '').trim();
      return normalize(notes).includes(normalize(text));
    };
    
    // building/floor -> concat dans notes seulement s'ils ne sont pas déjà présents
    let notes = value || '';
    if (extracted.building && !hasTextInNotes(notes, extracted.building)) {
      notes += (notes ? '\n' : '') + extracted.building;
    }
    if (extracted.floor && !hasTextInNotes(notes, extracted.floor)) {
      notes += (notes ? '\n' : '') + extracted.floor;
    }
    if (notes !== value) syncNotesMedical(notes);
  }

  // P2.1: Résumé enrichi — confirm bar (summaryText + badges séparés)
  const buildFooterSummary = () => {
    const badges = [];
    if (isRoundTrip) badges.push('AR');
    if (isRecurring) badges.push('Récurrente');
    if (isMaterialDelivery) badges.push('Livraison');

    if (isMaterialDelivery) {
      const desc = deliveryDescription?.trim();
      const descShort = desc && desc.length > 40 ? desc.slice(0, 39) + '…' : desc;
      const routeLabel = descShort
        ? `Livraison · ${descShort}`
        : 'Livraison · Description manquante';
      return {
        summaryText: routeLabel,
        summaryBadges: badges,
      };
    }
    if (!selectedClient) {
      return { summaryText: 'Client non sélectionné', summaryBadges: badges };
    }
    if (!scheduledTime) {
      return { summaryText: 'Date/heure manquante', summaryBadges: badges };
    }
    if (!pickupLocation || !dropoffLocation) {
      return { summaryText: 'Trajet incomplet', summaryBadges: badges };
    }
    const client = selectedClient?.label || 'Client';
    let dateStr = '—';
    try {
      const d = new Date(scheduledTime);
      dateStr = d.toLocaleDateString('fr-CH', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      /* ignore */
    }
    const pickup = shortAddress(pickupLocation) || '…';
    const dropoff = shortAddress(dropoffLocation) || '…';
    const routeLabel = `${pickup} → ${dropoff}`;
    const summaryText = `${client} · ${dateStr} · ${routeLabel}`;
    return { summaryText, summaryBadges: badges };
  };

  const footerSummary = buildFooterSummary();

  // === Mutations API ===
  // Gestion de la création de client
  const handleSaveNewClient = async (clientData, { existingClient } = {}) => {
    let createdClient = existingClient || null;
    try {
      const { hospitalization, billing_party_link, ...clientPayload } = clientData || {};
      const newClient = createdClient || (await createClient(clientPayload));
      createdClient = newClient;

      const createdClientId = newClient?.id || newClient?.data?.id || newClient?.client?.id;
      if (!createdClientId) {
        throw new Error('Client créé mais identifiant introuvable.');
      }

      if (hospitalization) {
        await createClientStay(createdClientId, {
          company_id: parseInt(hospitalization.company_id, 10),
          start_date: hospitalization.start_date,
          end_date: hospitalization.end_date || null,
          notes: hospitalization.notes || null,
        });
      }

      if (billing_party_link) {
        await linkClientBillingParty(createdClientId, {
          billing_party_id: billing_party_link.billing_party_id,
          role: billing_party_link.role || null,
          is_default: !!billing_party_link.is_default,
          contact_name: billing_party_link.contact_name || null,
          contact_email: billing_party_link.contact_email || null,
          contact_phone: billing_party_link.contact_phone || null,
        });
      }

      const clientOption = {
        value: newClient.id,
        label: `${newClient.user?.first_name ?? newClient.first_name ?? ''} ${
          newClient.user?.last_name ?? newClient.last_name ?? ''
        }`.trim(),
        raw: newClient,
      };

      await handleSelectClient(clientOption);
      queryClient.invalidateQueries(['clients']);
      setShowClientModal(false);
      toast.success('Client créé !');

      if (newClient.billing_address || newClient.domicile?.address) {
        const homeAddress =
          newClient.billing_address ||
          [newClient.domicile?.address, newClient.domicile?.zip, newClient.domicile?.city]
            .filter(Boolean)
            .join(', ');
        setPickupLocation(homeAddress);
      }
    } catch (err) {
      console.error('API createClient error:', err?.response?.data || err);
      const errorMessage = err?.response?.data?.error || err.message || 'Erreur création client';
      if (createdClient || err?.createdClient) {
        toast.error(`Client créé, mais erreur sur les informations complémentaires: ${errorMessage}`);
        const wrappedError = new Error(errorMessage);
        wrappedError.createdClient = createdClient || err?.createdClient;
        throw wrappedError;
      }
      toast.error(errorMessage);
      throw err; // Pour que NewClientModal affiche l'erreur
    }
  };

  const bookingMutation = useMutation({
    mutationFn: createManualBooking,
    onSuccess: (data) => {
      toast.success('Réservation créée !');
      // ✅ Reset billToPatient après succès
      setBillToPatient(false);
      onSuccess?.(data);
    },
    onError: (err) => {
      // ⚡ Ne pas logger les 401 temporaires qui sont gérés par le refresh automatique
      const is401Refresh =
        err?.response?.status === 401 && err?.config?._retryAfterRefresh === undefined;
      if (!is401Refresh) {
        // 🆕 Logger toute la structure de l'erreur pour debug
        console.error('❌ createManualBooking error:', err);
        console.error('📋 err.response:', err?.response);
        console.error('📋 err.response?.data:', err?.response?.data);
        console.error('📋 err.message:', err?.message);
        console.error('📋 err.toString():', err?.toString());

        // 🆕 Afficher les détails des erreurs de validation
        const errorData = err?.response?.data || err?.data || err;
        console.error("📋 Structure complète de l'erreur:", JSON.stringify(errorData, null, 2));

        if (errorData?.errors) {
          console.error('Détails des erreurs de validation:', errorData.errors);

          // 🔍 Extraire récursivement tous les champs en erreur (gérer la structure nested)
          const extractErrors = (obj, prefix = '', depth = 0) => {
            const extracted = [];
            if (!obj || typeof obj !== 'object' || depth > 10) return extracted; // Protection contre récursion infinie

            for (const [key, value] of Object.entries(obj)) {
              // Ignorer les clés spéciales et les structures "errors" nested qui sont juste des wrappers
              if (key === 'message' || key.startsWith('_')) continue;

              // ⚡ Si on trouve "errors" nested, descendre directement dedans sans ajouter au préfixe
              if (key === 'errors' && value && typeof value === 'object' && !Array.isArray(value)) {
                extracted.push(...extractErrors(value, prefix, depth + 1));
                continue;
              }

              const fieldPath = prefix ? `${prefix}.${key}` : key;

              if (Array.isArray(value)) {
                // Liste de messages directement - c'est un champ réel en erreur
                extracted.push({ field: fieldPath, messages: value });
              } else if (value && typeof value === 'object' && !Array.isArray(value)) {
                // Objet nested, extraire récursivement
                extracted.push(...extractErrors(value, fieldPath, depth + 1));
              } else if (value) {
                // Message unique
                extracted.push({ field: fieldPath, messages: [String(value)] });
              }
            }
            return extracted;
          };

          const allErrors = extractErrors(errorData.errors);

          if (allErrors.length > 0) {
            // Construire un message détaillé avec tous les champs en erreur
            const errorMessages = allErrors.map(({ field, messages }) => {
              const msgList = Array.isArray(messages) ? messages : [String(messages)];
              return `• ${field}: ${msgList.join(', ')}`;
            });
            toast.error(`Erreur de validation:\n${errorMessages.join('\n')}`, { duration: 10000 });
            return;
          }
        }
      }

      // Message d'erreur générique
      const errorMessage =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        `Erreur création réservation : ${err.message || 'Erreur inconnue'}`;

      toast.error(errorMessage);
    },
  });

  // === Soumission ===
  const handleSubmit = (e) => {
    e.preventDefault();

    if (!selectedClient) {
      toast.error('Veuillez sélectionner un client');
      return;
    }

    // 🔒 Sécurité : si un texte d’établissement est présent mais aucun établissement choisi (pas d’id)
    // On tolère le texte libre : on prévient juste
    if (establishmentText && !establishment?.id) {
      console.warn('[ManualBookingForm] établissement en texte libre :', establishmentText);
    }

    // Vérifier que la date & heure de l'aller est définie
    if (!scheduledTime) {
      toast.error('Veuillez sélectionner la date & heure de départ');
      return;
    }

    // Vérifier que si c'est un aller-retour, la date de retour est définie
    if (isRoundTrip && !returnDate) {
      toast.error('Veuillez sélectionner la date du retour');
      return;
    }

    // Montant : obligatoire pour transport patient, ignoré pour livraison matériel
    if (!isMaterialDelivery && (!amount || parseFloat(amount) <= 0)) {
      toast.error('Veuillez saisir un montant valide pour la course');
      return;
    }

    // Livraison matériel : description obligatoire
    if (isMaterialDelivery && !deliveryDescription?.trim()) {
      toast.error('Veuillez saisir la description de la livraison');
      return;
    }

    // 🔍 Debug coordonnées GPS
    console.log('📍 Coordonnées pickup:', pickupCoords);
    console.log('📍 Coordonnées dropoff:', dropoffCoords);

    const payload = {
      client_id: selectedClient.value,
      pickup_location: pickupLocation,
      dropoff_location: dropoffLocation,
      pickup_lat: pickupCoords.lat ?? undefined,
      pickup_lon: pickupCoords.lon ?? undefined,
      dropoff_lat: dropoffCoords.lat ?? undefined,
      dropoff_lon: dropoffCoords.lon ?? undefined,
      scheduled_time: ensureIsoDatetimeWithSeconds(scheduledTime),
      is_round_trip: !!isRoundTrip,
      amount: isMaterialDelivery ? undefined : (amount ? parseFloat(amount) : 0),
      amount_source: isMaterialDelivery ? undefined : (amountSource || 'manual'),
      amount_locked: isMaterialDelivery ? undefined : Boolean(amountLocked),
      pricing_profile_id: activePricingProfileId || undefined,
      pricing_profile_version_id: activePricingProfileVersionId || undefined,
      ...(isMaterialDelivery && {
        mission_type: 'material_delivery',
        delivery_description: (deliveryDescription || '').trim() || null,
      }),

      // Champs médicaux : trim + vide → null, clé absente si null (pas de " " envoyé)
      ...(cleanOptionalText(
        medicalFacility ||
          establishment?.display_name ||
          establishment?.label ||
          establishmentText
      ) != null && {
        medical_facility: cleanOptionalText(
          medicalFacility ||
            establishment?.display_name ||
            establishment?.label ||
            establishmentText
        ),
      }),
      ...(cleanOptionalText(hospitalServiceRef.current || serviceObj?.name) != null && {
        hospital_service: cleanOptionalText(hospitalServiceRef.current || serviceObj?.name),
      }),
      ...(cleanOptionalText(doctorNameRef.current) != null && {
        doctor_name: cleanOptionalText(doctorNameRef.current),
      }),
      ...(cleanOptionalText(notesMedicalRef.current) != null && {
        notes_medical: cleanOptionalText(notesMedicalRef.current),
      }),
      ...(cleanOptionalText(pickupAccessNotesRef.current) != null && {
        pickup_access_notes: cleanOptionalText(pickupAccessNotesRef.current),
      }),
      ...(cleanOptionalText(dropoffAccessNotesRef.current) != null && {
        dropoff_access_notes: cleanOptionalText(dropoffAccessNotesRef.current),
      }),
      wheelchair_client_has: wheelchairOptions.clientHasWheelchair || undefined,
      wheelchair_need: wheelchairOptions.needWheelchair || undefined,

      // Nouveaux champs structurés
      establishment_id: establishment?.id ?? undefined,
      medical_service_id: serviceObj?.id ?? undefined,
      
      // Override facturation patient (si client hospitalisé mais on veut facturer au patient)
      // ✅ IMPORTANT: Envoyer UNIQUEMENT si billToPatient === true (pas undefined si false)
      ...(billToPatient ? { bill_to_patient: true } : {}),
      // Si aller-retour : toujours envoyer return_date
      // return_time seulement si une heure est spécifiée (sinon = "heure à confirmer")
      // ⚡ return_time doit être au format ISO 8601 complet (YYYY-MM-DDTHH:mm:ss)
      ...(isRoundTrip && returnDate
        ? (() => {
            // Si returnTime est vide/null, ne PAS envoyer return_time (heure à confirmer)
            if (!returnTime || !returnTime.trim()) {
              return {
                return_date: returnDate,
                // Pas de return_time → signifie "heure à confirmer"
              };
            }

            // Si returnTime est rempli, combiner avec returnDate pour créer un datetime ISO complet
            // 🔍 Debug pour voir les valeurs avant formatage
            console.log('🔍 [ReturnTime] returnDate:', returnDate, 'type:', typeof returnDate);
            console.log(
              '🔍 [ReturnTime] returnTime (raw):',
              returnTime,
              'type:',
              typeof returnTime
            );
            const formattedReturnTime = combineDateAndTime(returnDate, returnTime);
            console.log('🔍 [ReturnTime] formaté:', formattedReturnTime);

            if (!formattedReturnTime) {
              console.warn('⚠️ [ReturnTime] Échec du formatage, retour sans return_time');
              return {
                return_date: returnDate,
                // Pas de return_time si le formatage échoue
              };
            }

            return {
              return_date: returnDate,
              return_time: formattedReturnTime,
            };
          })()
        : {}),

      // Récurrence (ne pas envoyer si is_recurring est false)
      ...(isRecurring
        ? {
            is_recurring: true,
            recurrence_type: recurrenceType || undefined,
            recurrence_days:
              recurrenceType === 'custom' && selectedDays?.length > 0 ? selectedDays : undefined,
            recurrence_end_date: recurrenceEndDate || undefined,
            occurrences: occurrences > 0 ? occurrences : undefined,
          }
        : {}), // ⚡ Ne rien envoyer si la récurrence est désactivée (is_recurring aura la valeur par défaut false côté backend)
    };

    console.log('[ManualBookingForm] payload:', payload);
    console.log('🔄 Récurrence activée:', isRecurring);
    console.log('📅 Type de récurrence:', recurrenceType);
    console.log('🗓️ Jours sélectionnés:', selectedDays);
    console.log("🔢 Nombre d'occurrences:", occurrences);
    // 🔍 Trace pour debug facturation
    console.log('💳 [Facturation] billToPatient:', billToPatient);
    console.log('💳 [Facturation] activeStay:', activeStay);
    console.log('💳 [Facturation] bill_to_patient dans payload:', payload.bill_to_patient);

    // Fermer la modale immédiatement apres validation locale pour eviter la latence percue.
    onSubmitStart?.(payload);
    bookingMutation.mutate(payload);
  };

  return (
    <div
      className={styles.formWrapper}
      data-testid="manual-booking-form-wrapper"
      data-tour-id="manual-booking-form"
    >
      <div className={styles.modalHeader}>
        <h2 className={styles.modalTitle}>Créer une réservation</h2>
        <p className={styles.modalSubtitle}>
          Renseignez le trajet, puis ajoutez les détails si nécessaire.
        </p>
      </div>
      <form ref={formRef} onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.formScrollBody}>
          {/* COLONNE GAUCHE */}
          <div className={styles.columnLeft} data-tour-id="booking-left-panel">
          <ManualBookingClientSelect
            selectedClient={selectedClient}
            defaultClientOptions={defaultClientOptions}
            loadClientOptions={loadClientOptions}
            onChange={handleSelectClient}
            onCreateOption={handleCreateClientOption}
            activeStay={activeStay}
            billToPatient={billToPatient}
            onBillToPatientChange={handleBillToPatientChange}
            getClinicPickupAddress={getClinicPickupAddress}
          />

          {/* Lieu de prise en charge + Swap + Destination — P0: id/htmlFor a11y */}
          <div className={styles.formGroup} data-tour-id="booking-addresses">
            <div className={styles.addressesRow}>
              <div className={styles.addressesFields}>
                <Label htmlFor="pickup_location">Lieu de prise en charge *</Label>
                <AddressAutocomplete
                name="pickup_location"
                inputId="pickup_location"
                value={pickupLocation}
                onChange={(e) => {
                  setPickupLocation(e.target.value);
                  setPickupCoords({ lat: null, lon: null });
                }}
                onSelect={(item) => {
                  console.log('📍 [Pickup] Item sélectionné:', item);
                  const address = item.label || item.address || '';
                  setPickupLocation(address);
                  setPickupCoords({
                    lat: item.lat ?? null,
                    lon: item.lon ?? null,
                  });
                  console.log(`📍 [Pickup] Adresse: ${address}, GPS: ${item.lat}, ${item.lon}`);
                }}
                placeholder="Saisir ou choisir l'adresse"
                required
              />
                <Label htmlFor="dropoff_location">Lieu de destination *</Label>
                <AddressAutocomplete
                name="dropoff_location"
                inputId="dropoff_location"
                value={dropoffLocation}
              onChange={(e) => {
                setDropoffLocation(e.target.value);
                setDropoffCoords({ lat: null, lon: null });
              }}
              onSelect={(item) => {
                // DEBUG : Afficher la structure complète de l'item
                console.log('🏥 [Destination sélectionnée] item complet:', item);
                console.log('🏥 [Destination] item.label:', item.label);
                console.log('🏥 [Destination] item.main_text:', item.main_text);
                console.log('🏥 [Destination] item.address:', item.address);
                console.log('🏥 [Destination] item.secondary_text:', item.secondary_text);
                console.log('🏥 [Destination] item.types:', item.types);

                // ✅ Fonction pour extraire et nettoyer les informations d'étage
                const extractFloorInfo = (text) => {
                  const floorRegex = /(au\s+)?(\d{1,2}(?:er|ème|e)?)\s+étage/gi;
                  const matches = text.match(floorRegex);
                  if (matches && matches.length > 0) {
                    return {
                      floor: matches[0].trim(),
                      cleanedText: text
                        .replace(floorRegex, '')
                        .replace(/^[,\s-]+|[,\s-]+$/g, '')
                        .trim(),
                    };
                  }
                  return { floor: null, cleanedText: text };
                };

                // Fonction pour vérifier si l'étage existe déjà dans les notes
                const hasFloorInNotes = (notes, floorInfo) => {
                  if (!notes || !floorInfo) return false;
                  // Normaliser les deux textes pour la comparaison (enlever emoji, espaces, etc.)
                  const normalizeFloor = (text) => text.toLowerCase().replace(/[🏢\s]/g, '').trim();
                  const normalizedFloor = normalizeFloor(floorInfo);
                  // Vérifier si l'étage (normalisé) existe dans les notes
                  return normalizeFloor(notes).includes(normalizedFloor);
                };

                // ✅ Utiliser main_text pour Google Places (nom sans adresse), sinon label
                // main_text = "HUG Maternité" ou "Dr méd. Lakki Brahim"
                // label = "HUG Maternité, Boulevard de la Cluse, Genève, Suisse" (complet)
                let establishmentName = item.main_text || item.label || item.name || '';

                // Nettoyer le nom d'établissement pour retirer l'étage si présent
                const { floor: floorFromName, cleanedText: cleanedEstablishmentName } =
                  extractFloorInfo(establishmentName);
                establishmentName = cleanedEstablishmentName || establishmentName;

                // ✅ PRÉSERVER l'adresse originale sélectionnée par l'utilisateur
                // Utiliser le label original (celui affiché dans le modal et sélectionné)
                const originalLabel = item.label || '';
                let fullAddress = originalLabel;
                let floorInfo = floorFromName; // Étage extrait du nom

                // Nettoyer l'adresse pour retirer l'étage si présent dans le label original
                const { floor: floorFromLabel, cleanedText: cleanedLabel } = extractFloorInfo(originalLabel);
                floorInfo = floorInfo || floorFromLabel;
                
                // Utiliser l'adresse nettoyée, ou l'originale si pas d'étage
                fullAddress = cleanedLabel || originalLabel;

                console.log('🏥 [Destination] fullAddress final:', fullAddress);
                console.log('🏥 [Destination] establishmentName final:', establishmentName);
                console.log('🏥 [Destination] floorInfo:', floorInfo);

                // Utiliser l'adresse complète pour la destination
                setDropoffLocation(fullAddress);
                setDropoffCoords({
                  lat: item.lat ?? null,
                  lon: item.lon ?? null,
                });

                // ✅ Pour Google Places, utiliser directement main_text sans extraction
                // Pour éviter que "Hôpitaux Universitaires de Genève (HUG)" devienne "HUG"
                const isGooglePlace = item.source === 'google_places';

                const txt = (establishmentName || '').toLowerCase();

                // ⚠️ IMPORTANT : Vérifier DOCTEUR en PREMIER (avant établissement)
                // Car les docteurs ont aussi le type "health" qui déclencherait looksLikeMedical
                // Mais exclure les centres d'imagerie et cliniques (ce sont des établissements, pas des docteurs)
                const isMedicalCenter = 
                  txt.includes('centre d\'imagerie') ||
                  txt.includes('centre d imagerie') ||
                  txt.includes('clinique') ||
                  txt.includes('hôpital') ||
                  txt.includes('hopital');
                
                const looksLikeDoctor = !isMedicalCenter && (
                  txt.includes('dr ') ||
                  txt.includes('dr.') ||
                  txt.includes('dr méd') ||
                  txt.includes('docteur') ||
                  txt.includes('cabinet') ||
                  txt.includes('médecin') ||
                  item.types?.includes('doctor')
                );

                const looksLikeMedical =
                  txt.includes('hôpital') ||
                  txt.includes('hopital') ||
                  txt.includes('clinique') ||
                  txt.includes('centre médical') ||
                  txt.includes('centre d\'imagerie') ||
                  txt.includes('centre d imagerie') ||
                  txt.includes('imagerie médicale') ||
                  txt.includes('radiology') ||
                  txt.includes('radiologie') ||
                  txt.includes('centre d') ||
                  txt.includes('centre collectif') ||
                  txt.includes('hébergement') ||
                  txt.includes('hebergement') ||
                  txt.includes('ehpad') ||
                  txt.includes('ems') ||
                  txt.includes('foyer') ||
                  txt.includes('résidence') ||
                  txt.includes('maison de retraite') ||
                  item.types?.includes('hospital') ||
                  item.types?.includes('health');

                // ⚠️ ORDRE DE VÉRIFICATION :
                // 1️⃣ Médecin → "Nom du médecin"
                // 2️⃣ Établissement médical/social → "Établissement médical"
                // 3️⃣ Tout le reste → "Notes médicales"

                // ✅ 1️⃣ VÉRIFIER D'ABORD SI C'EST UN MÉDECIN
                if (looksLikeDoctor) {
                  console.log(
                    '✅ [Destination] Cabinet médical/docteur détecté:',
                    establishmentName
                  );
                  // ✅ Pour Google Places, nettoyer juste avant la virgule
                  if (isGooglePlace) {
                    const cleanName = establishmentName.split(',')[0].trim();
                    syncDoctorName(cleanName);
                    // NE PAS remplir l'établissement pour un docteur
                    setEstablishmentText('');
                    setMedicalFacility('');
                    // Ajouter l'étage dans les notes si présent et pas déjà présent
                    if (floorInfo) {
                      syncNotesMedical((prevNotes) => {
                        if (hasFloorInNotes(prevNotes, floorInfo)) {
                          return prevNotes; // L'étage existe déjà, ne pas dupliquer
                        }
                        const floorNote = `🏢 ${floorInfo}`;
                        return prevNotes ? `${prevNotes}\n${floorNote}` : floorNote;
                      });
                    }
                  } else {
                    // Pour Photon/autre, utiliser l'extraction
                    const extracted = extractMedicalServiceInfo(establishmentName);
                    syncDoctorName(extracted.doctor_name || establishmentName.split(',')[0].trim());
                    setEstablishmentText('');
                    setMedicalFacility('');
                  }
                }
                // ✅ 2️⃣ VÉRIFIER SI C'EST UN ÉTABLISSEMENT MÉDICAL/SOCIAL
                else if (looksLikeMedical) {
                  console.log('✅ [Destination] Établissement médical détecté:', establishmentName);
                  // ✅ Pour Google Places, utiliser directement le nom sans extraction
                  if (isGooglePlace) {
                    setEstablishmentText(establishmentName);
                    setMedicalFacility(establishmentName);
                    // Ajouter l'étage dans les notes si présent et pas déjà présent
                    if (floorInfo) {
                      syncNotesMedical((prevNotes) => {
                        if (hasFloorInNotes(prevNotes, floorInfo)) {
                          return prevNotes; // L'étage existe déjà, ne pas dupliquer
                        }
                        const floorNote = `🏢 ${floorInfo}`;
                        return prevNotes ? `${prevNotes}\n${floorNote}` : floorNote;
                      });
                    }
                  } else {
                    // Pour Photon/autre, utiliser l'extraction
                    const extracted = extractMedicalServiceInfo(establishmentName);
                    setEstablishmentText(extracted.medical_facility || establishmentName);
                    setMedicalFacility(extracted.medical_facility || establishmentName);
                    if (extracted.doctor_name) syncDoctorName(extracted.doctor_name);
                    if (extracted.hospital_service) syncHospitalService(extracted.hospital_service);
                  }
                }
                // ✅ 3️⃣ TOUT LE RESTE → NOTES MÉDICALES
                else {
                  if (isGooglePlace) {
                    // Pour Google Places : restaurants, parcs, lieux publics, etc.
                    // Vérifier si c'est un établissement nommé (pas juste une adresse de rue)
                    const hasTypes = item.types?.length > 0;
                    const isNamedPlace = item.types?.some(
                      (t) => t !== 'geocode' && t !== 'route' && t !== 'street_address'
                    );

                    if (hasTypes && isNamedPlace) {
                      // C'est un lieu nommé (restaurant, magasin, parc, etc.)
                      console.log('📍 [Destination] Lieu public/POI détecté:', establishmentName);
                      let locationNote = `📍 Rendez-vous: ${establishmentName}`;
                      syncNotesMedical((prevNotes) => {
                        // Construire la note avec l'étage si présent et pas déjà dans prevNotes
                        let finalNote = locationNote;
                        if (floorInfo && !hasFloorInNotes(prevNotes, floorInfo)) {
                          finalNote += `\n🏢 ${floorInfo}`;
                        }
                        return prevNotes ? `${prevNotes}\n${finalNote}` : finalNote;
                      });
                      // Ne PAS remplir les champs médicaux
                      setEstablishmentText('');
                      setMedicalFacility('');
                      syncDoctorName('');
                    } else {
                      // Juste une adresse de rue → ne rien faire
                      console.log('ℹ️ [Destination] Adresse de rue normale:', establishmentName);
                    }
                  } else {
                    // Pour Photon/autre, essayer l'extraction
                    const extracted = extractMedicalServiceInfo(establishmentName);
                    if (extracted.medical_facility || extracted.doctor_name) {
                      console.log('✅ [Destination] Info médicale extraite:', extracted);
                      setEstablishmentText(extracted.medical_facility || '');
                      setMedicalFacility(extracted.medical_facility || '');
                      if (extracted.doctor_name) syncDoctorName(extracted.doctor_name);
                      if (extracted.hospital_service)
                        syncHospitalService(extracted.hospital_service || '');
                    }
                  }
                }

                // Ne pas appeler syncHospitalService/setServiceObj ici pour éviter d'écraser
                setServiceObj(null);
              }}
              placeholder="Saisir ou choisir l'adresse"
            />
              </div>
              <button
                type="button"
                className={styles.swapButton}
                onClick={handleSwapAddresses}
                title="Inverser pickup/destination"
                aria-label="Inverser le lieu de prise en charge et la destination"
              >
                ↕
              </button>
            </div>
          </div>

          {/* Date & heure aller */}
          <div className={styles.formGroup} data-tour-id="booking-datetime">
            <Label>Date &amp; heure de départ *</Label>
            <div className={styles.dateTimeRow}>
              <InlineDatePicker
                value={scheduledDate}
                onChange={(v) => setScheduledDate(v)}
                placeholder="Date"
              />
              <InlineTimePicker
                value={scheduledHour}
                onChange={(v) => setScheduledHour(v)}
                placeholder="Heure"
              />
              <div className={styles.datePresetGroup} data-tour-id="booking-time-presets">
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  data-tour-id="booking-plus30"
                  onClick={() => applyDateTimePreset('now30')}
                >
                  +30 min
                </button>
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  onClick={() => applyDateTimePreset('now1h')}
                >
                  +1h
                </button>
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  onClick={() => applyDateTimePreset('tomorrow9')}
                >
                  Demain 9h
                </button>
              </div>
            </div>

            <div className={styles.chipsRow}>
              <button
                type="button"
                className={`${styles.chip} ${isRoundTrip ? styles.isActive : ''}`}
                data-tour-id="booking-roundtrip-toggle"
                onClick={() => setIsRoundTrip(!isRoundTrip)}
                aria-pressed={isRoundTrip}
              >
                ⇄ Trajet AR
              </button>
              <button
                type="button"
                className={`${styles.chip} ${isRecurring ? styles.isActive : ''}`}
                onClick={() => setIsRecurring(!isRecurring)}
                aria-pressed={isRecurring}
              >
                🔄 Récurrente
              </button>
              <button
                type="button"
                className={`${styles.chip} ${isMaterialDelivery ? styles.isActive : ''}`}
                onClick={() => {
                  const next = !isMaterialDelivery;
                  setIsMaterialDelivery(next);
                  if (!next) setDeliveryDescription('');
                  if (next && amount && parseFloat(amount) > 0) {
                    toast.info('Montant ignoré pour les livraisons (prix fixe appliqué).');
                  }
                }}
                aria-pressed={isMaterialDelivery}
              >
                📦 Livraison
              </button>
            </div>

            {isMaterialDelivery && (
              <div className={`${styles.formGroup} ${styles.deliveryDescriptionGroup}`}>
                <Label>Description de la livraison *</Label>
                <Input
                  type="text"
                  name="delivery_description"
                  value={deliveryDescription}
                  onChange={(e) => setDeliveryDescription(e.target.value)}
                  placeholder="Ex: Livraison médicament, Oxygène, Documents…"
                  required={isMaterialDelivery}
                />
                <span className={styles.medicalFieldHint}>
                  Ex: Livraison médicament, Oxygène, Documents
                </span>
              </div>
            )}

            {isRoundTrip && (
              <div className={styles.returnTimeGroup} data-tour-id="booking-return-config">
                <Label>Date &amp; heure de retour</Label>
                <div className={styles.dateTimeRow}>
                  <InlineDatePicker
                    value={returnDate}
                    onChange={(v) => setReturnDate(v)}
                    placeholder="Date"
                  />
                  <InlineTimePicker
                    value={returnTime}
                    onChange={(v) => setReturnTime(v)}
                    placeholder="Heure (opt.)"
                  />
                </div>
                <small className={styles.returnTimeHint}>
                  Si l'heure n'est pas définie, elle pourra être planifiée plus tard
                </small>
              </div>
            )}

            {isRecurring && (
              <div className={styles.recurrenceConfig}>
                {/* Type de récurrence */}
                <div className={styles.recurrenceGroup}>
                  <Label>Type de récurrence</Label>
                  <select
                    value={recurrenceType}
                    onChange={(e) => {
                      setRecurrenceType(e.target.value);
                    }}
                    className={styles.recurrenceSelect}
                  >
                    <option value="daily">📅 Tous les jours</option>
                    <option value="weekly">📆 Toutes les semaines</option>
                    <option value="custom">⚙️ Jours personnalisés</option>
                  </select>
                </div>

                {/* Jours personnalisés */}
                {recurrenceType === 'custom' && (
                  <div className={styles.recurrenceGroup}>
                    <Label>
                      Sélectionner les jours ({selectedDays.length} sélectionné
                      {selectedDays.length > 1 ? 's' : ''})
                    </Label>
                    <div className={styles.daysSelector}>
                      {weekDays.map((day) => (
                        <button
                          key={day.id}
                          type="button"
                          className={`${styles.dayButton} ${
                            selectedDays.includes(day.id) ? styles.daySelected : ''
                          }`}
                          onClick={() => {
                            toggleDay(day.id);
                            console.log(
                              `🗓️ Jour ${day.label} (ID: ${day.id}) ${
                                selectedDays.includes(day.id) ? 'désélectionné' : 'sélectionné'
                              }`
                            );
                          }}
                          title={day.label}
                        >
                          {day.short}
                        </button>
                      ))}
                    </div>
                    {selectedDays.length === 0 && (
                      <div className={styles.recurrenceWarning}>
                        ⚠️ Veuillez sélectionner au moins un jour
                      </div>
                    )}
                  </div>
                )}

                {/* Nombre d'occurrences */}
                <div className={styles.recurrenceGroup}>
                  <Label>Nombre de répétitions</Label>
                  <Input
                    type="number"
                    min="1"
                    max="52"
                    value={occurrences}
                    onChange={(e) => setOccurrences(parseInt(e.target.value) || 1)}
                    placeholder="Ex: 4 (pour 4 semaines)"
                  />
                  <div className={styles.recurrenceHint}>
                    {occurrences > 0 && (
                      <span>
                        {recurrenceType === 'custom' && selectedDays.length > 0 ? (
                          <>
                            ℹ️ Créera {occurrences} × {selectedDays.length} jour
                            {selectedDays.length > 1 ? 's' : ''} ={' '}
                            {occurrences * selectedDays.length} réservation
                            {occurrences * selectedDays.length > 1 ? 's' : ''}
                            {isRoundTrip && (
                              <>
                                {' '}
                                (×2 avec aller-retour = {occurrences * selectedDays.length * 2} au
                                total)
                              </>
                            )}
                          </>
                        ) : (
                          <>
                            ℹ️ Créera {occurrences} réservation
                            {occurrences > 1 ? 's' : ''}
                            {isRoundTrip && (
                              <> (×2 avec aller-retour = {occurrences * 2} au total)</>
                            )}
                          </>
                        )}
                      </span>
                    )}
                  </div>
                </div>

                {/* Date de fin (optionnel) */}
                <div className={styles.recurrenceGroup}>
                  <Label>Jusqu'au (optionnel)</Label>
                  <InlineDatePicker
                    value={recurrenceEndDate}
                    onChange={(v) => setRecurrenceEndDate(v)}
                    placeholder="Date de fin"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Montant (masqué pour livraison : prix fixe entreprise) */}
          <div className={styles.formGroup} data-tour-id="booking-amount">
            <div className={styles.amountLabel}>
              <Label htmlFor="amount">
                Montant {isMaterialDelivery ? '(ignoré pour livraison)' : '*'}
              </Label>
              {amountSource === 'preferential' && (
                <span className={styles.preferentialBadge}>💰 Tarif préférentiel</span>
              )}
              {amountSource === 'simulated' && (
                <span className={styles.preferentialBadge}>⚙️ Calculé automatiquement</span>
              )}
              {amountSource === 'manual' && (
                <span className={styles.preferentialBadge}>✍️ Modifié manuellement</span>
              )}
            </div>
            <Input
              id="amount"
              type="number"
              name="amount"
              step="0.01"
              value={amount}
              onChange={(e) => {
                const nextValue = e.target.value;
                setAmount(nextValue);
                if (!suppressManualAmountRef.current) {
                  setAmountSource('manual');
                  setAmountLocked(true);
                  setPricingWarning('');
                }
              }}
              placeholder="Ex: 45.00"
              disabled={isMaterialDelivery}
            />
            {isMaterialDelivery && (
              <span className={styles.amountHint}>
                Montant géré automatiquement pour les livraisons matériel.
              </span>
            )}
            {!isMaterialDelivery && amountLocked && (
              <span className={styles.amountHint}>
                Montant verrouillé manuellement.
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  style={{ marginLeft: 8 }}
                  onClick={() => {
                    setAmountLocked(false);
                    setAmountSource(null);
                    runPricingSimulation();
                  }}
                >
                  Recalculer
                </button>
              </span>
            )}
            {!isMaterialDelivery && isSimulatingPricing && (
              <span className={styles.amountHint}>
                {amountSource === 'simulated' && amount
                  ? 'Mise à jour du montant exact en cours…'
                  : 'Calcul du montant en cours…'}
              </span>
            )}
            {!isMaterialDelivery && pricingWarning && (
              <span className={styles.amountHint}>{pricingWarning}</span>
            )}
            {!isMaterialDelivery && lastPricingUpdateAt && (
              <span className={styles.amountHint}>
                Dernière mise à jour: {lastPricingUpdateAt.toLocaleTimeString('fr-CH')}
              </span>
            )}
            {estimatedDuration && (
              <div className={styles.estimatedDurationBadge}>
                ⏱️ Durée estimée : <strong>{estimatedDuration} min</strong>
              </div>
            )}
          </div>
        </div>

        {/* COLONNE DROITE - Informations médicales (section secondaire) */}
        <div className={styles.columnRight}>
          <div className={styles.medicalSection} data-tour-id="booking-medical-section">
            <h3 className={styles.medicalSectionTitle}>🏥 Informations médicales</h3>

            {/* Établissement médical */}
            <div className={styles.medicalFormGroup}>
              <Label htmlFor="establishment-select">Établissement</Label>
              <EstablishmentSelect
                inputId="establishment-select"
                value={establishmentText}
                onChange={handleEstablishmentTextChange}
                onPickEstablishment={onPickEstablishment}
                placeholder="HUG, Clinique La Colline, Grangettes, La Tour…"
                inputClassName={styles.input}
              />
            </div>

            {/* Service / Bâtiment */}
            <div className={styles.medicalFormGroup}>
              <Label htmlFor="hospital_service">Service / Bât.</Label>
              {establishment?.id ? (
                <ServiceSelect
                  key={establishment?.id}
                  inputId="hospital_service"
                  establishmentId={establishment?.id}
                  value={serviceObj}
                  onChange={onChangeService}
                  placeholder="Ex: CHIR, Urgences adultes, Cardiologie…"
                />
              ) : (
                <IsolatedTextInput
                  id="hospital_service"
                  type="text"
                  name="hospital_service"
                  externalValue={hospitalServiceExternal}
                  valueRef={hospitalServiceRef}
                  placeholder="Ex: CHIR, Urgences, Cardiologie…"
                />
              )}
            </div>

            {/* Nom du médecin */}
            <div className={styles.medicalFormGroup}>
              <Label htmlFor="doctor_name">Médecin</Label>
              <IsolatedTextInput
                id="doctor_name"
                type="text"
                name="doctor_name"
                externalValue={doctorNameExternal}
                valueRef={doctorNameRef}
                placeholder="Ex : Dr Dupont"
              />
            </div>

            {/* Notes médicales */}
            <div className={styles.medicalFormGroup}>
              <Label htmlFor="notes_medical">Notes</Label>
              <IsolatedTextarea
                id="notes_medical"
                name="notes_medical"
                externalValue={notesMedicalExternal}
                valueRef={notesMedicalRef}
                onBlur={handleNotesMedicalBlur}
                placeholder="Instructions particulières, bâtiment, étage…"
                rows={2}
                className={`${styles.textarea} ${styles.textareaCompact}`}
              />
            </div>

            {/* Instructions accès pickup */}
            <div className={styles.medicalFormGroup}>
              <Label htmlFor="pickup_access_notes">Accès pickup</Label>
              <IsolatedTextInput
                id="pickup_access_notes"
                type="text"
                name="pickup_access_notes"
                externalValue={pickupAccessNotesExternal}
                valueRef={pickupAccessNotesRef}
                placeholder="Ex: entrée arrière, sonner à…, table…, appeler avant…"
              />
            </div>

            {/* Instructions accès destination */}
            <div className={styles.medicalFormGroup}>
              <Label htmlFor="dropoff_access_notes">Accès destination</Label>
              <IsolatedTextInput
                id="dropoff_access_notes"
                type="text"
                name="dropoff_access_notes"
                externalValue={dropoffAccessNotesExternal}
                valueRef={dropoffAccessNotesRef}
                placeholder="Ex: entrée B, étage 2, service…, appeler secrétariat…"
              />
            </div>

            {/* Options chaise roulante */}
            <div className={styles.medicalFormGroup}>
              <Label>Chaise roulante</Label>
              <div className={styles.dateTimeRow}>
                <button
                  type="button"
                  className={`${styles.chip} ${wheelchairOptions.clientHasWheelchair ? styles.isActive : ''}`}
                  onClick={() => setWheelchairOptions({
                    clientHasWheelchair: !wheelchairOptions.clientHasWheelchair,
                    needWheelchair: false,
                  })}
                  aria-pressed={wheelchairOptions.clientHasWheelchair}
                >
                  ♿ En chaise
                </button>
                <button
                  type="button"
                  className={`${styles.chip} ${wheelchairOptions.needWheelchair ? styles.isActive : ''}`}
                  onClick={() => setWheelchairOptions({
                    needWheelchair: !wheelchairOptions.needWheelchair,
                    clientHasWheelchair: false,
                  })}
                  aria-pressed={wheelchairOptions.needWheelchair}
                >
                  🏥 Fournir chaise
                </button>
              </div>
            </div>
          </div>
        </div>
        </div>

        {/* Footer fixe en bas : résumé + Annuler à gauche, CTA à droite */}
        <div className={styles.footerActions} data-testid="footer-actions-sticky">
          <div className={styles.footerSummary} aria-live="polite">
            <span className={styles.footerSummaryText}>{footerSummary.summaryText}</span>
            <span className={styles.footerSummaryBadges}>
              {footerSummary.summaryBadges.map((badge) => (
                <span key={badge} className={styles.summaryBadge}>
                  {badge}
                </span>
              ))}
            </span>
          </div>
          <div className={styles.footerBody}>
            <div className={styles.footerLeft}>
              {onClose && (
                <button type="button" className={styles.footerCancel} onClick={onClose}>
                  Annuler
                </button>
              )}
            </div>
            <div className={styles.footerRight}>
              <button
                type="submit"
                data-tour-id="booking-submit"
                className={styles.submitButton}
                disabled={
                  bookingMutation.isLoading ||
                  (isMaterialDelivery && !deliveryDescription?.trim())
                }
              >
                {bookingMutation.isLoading ? '⏳ Création…' : 'Créer la réservation'}
              </button>
            </div>
          </div>
        </div>
      </form>

      {showClientModal && (
        <NewClientModal onClose={() => setShowClientModal(false)} onSave={handleSaveNewClient} />
      )}
    </div>
  );
}
