// frontend/src/pages/company/Dashboard/components/ManualBookingForm.jsx (fixed)

import React, { useState, useCallback, useEffect } from 'react';
import AsyncCreatableSelect from 'react-select/async-creatable';
import NewClientModal from '../../Clients/components/NewClientModal';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import {
  createManualBooking,
  searchClients,
  createClient,
} from '../../../../services/companyService';
import { Input } from './ui/Input';
import { Label } from './ui/Label';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import apiClient from '../../../../utils/apiClient';

// ⬇️ Assure-toi que ces chemins correspondent à ta structure réelle
import EstablishmentSelect from '../../../../components/common/EstablishmentSelect';
import ServiceSelect from '../../../../components/common/ServiceSelect';

import { extractMedicalServiceInfo } from '../../../../utils/medicalExtract';
import { toast } from 'sonner';
import styles from './ManualBookingForm.module.css';

export default function ManualBookingForm({ onSuccess }) {
  const queryClient = useQueryClient();

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

  // --- Client
  const [selectedClient, setSelectedClient] = useState(null);
  const [showClientModal, setShowClientModal] = useState(false);

  // --- Date/heure de l'aller
  const [scheduledTime, setScheduledTime] = useState('');

  // --- Aller-retour
  const [isRoundTrip, setIsRoundTrip] = useState(false);
  const [returnDate, setReturnDate] = useState(''); // Date de retour
  const [returnTime, setReturnTime] = useState(''); // Heure de retour (optionnel)

  // Pré-remplir automatiquement la date de retour avec la date de l'aller
  useEffect(() => {
    if (isRoundTrip && scheduledTime && !returnDate) {
      // Extraire la date de scheduledTime (format: YYYY-MM-DDTHH:mm)
      const datePart = scheduledTime.split('T')[0]; // YYYY-MM-DD
      setReturnDate(datePart);
    }
  }, [isRoundTrip, scheduledTime, returnDate]);

  // --- Récurrence
  const [isRecurring, setIsRecurring] = useState(false);
  const [recurrenceType, setRecurrenceType] = useState('weekly'); // daily, weekly, custom
  const [recurrenceEndDate, setRecurrenceEndDate] = useState('');
  const [selectedDays, setSelectedDays] = useState([]); // Pour la récurrence personnalisée
  const [occurrences, setOccurrences] = useState(4); // Nombre d'occurrences

  // --- Infos médicales libres (toujours disponibles)
  const [medicalFacility, setMedicalFacility] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [hospitalService, setHospitalService] = useState(''); // restera synchronisé avec serviceName
  const [notesMedical, setNotesMedical] = useState('');
  const [wheelchairOptions, setWheelchairOptions] = useState({
    clientHasWheelchair: false,
    needWheelchair: false,
  });

  // Calculer la durée estimée en temps réel avec OSRM (routes réelles)
  React.useEffect(() => {
    const calculateDuration = async () => {
      if (pickupCoords.lat && pickupCoords.lon && dropoffCoords.lat && dropoffCoords.lon) {
        try {
          // ✅ Appel à l'API OSRM pour obtenir la durée réelle du trajet
          const response = await apiClient.get('/osrm/route', {
            params: {
              pickup_lat: pickupCoords.lat,
              pickup_lon: pickupCoords.lon,
              dropoff_lat: dropoffCoords.lat,
              dropoff_lon: dropoffCoords.lon,
            },
          });

          if (response.data && response.data.duration) {
            // Convertir de secondes en minutes
            const durationMinutes = Math.round(response.data.duration / 60);
            const _distanceKm = (response.data.distance / 1000).toFixed(1);

            setEstimatedDuration(durationMinutes);
          } else {
            console.warn('⚠️ Pas de durée retournée par OSRM');
            setEstimatedDuration(null);
          }
        } catch (error) {
          console.error('❌ Erreur calcul durée OSRM:', error);

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
            console.warn(`⚠️ Fallback Haversine : ${durationMinutes} min (approximatif)`);
          } catch (fallbackError) {
            console.error('❌ Erreur fallback Haversine:', fallbackError);
            setEstimatedDuration(null);
          }
        }
      } else {
        setEstimatedDuration(null);
      }
    };

    calculateDuration();
  }, [pickupCoords.lat, pickupCoords.lon, dropoffCoords.lat, dropoffCoords.lon]);

  // Helper: min pour <input type="datetime-local"> au format local (pas UTC)
  const minLocalDatetime = (() => {
    const d = new Date(Date.now() + 5 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
      d.getHours()
    )}:${pad(d.getMinutes())}`;
  })();

  // === Handlers établissement / service ===
  const handleEstablishmentTextChange = (newText) => {
    setEstablishmentText(newText);
    // Si l'utilisateur modifie le texte, on efface l'objet sélectionné
    if (establishment && newText !== (establishment.label || establishment.display_name || '')) {
      setEstablishment(null);
      setServiceObj(null);
      setHospitalService('');
    }
  };

  const onPickEstablishment = (estab) => {
    console.log('🏥 ManualBookingForm.onPickEstablishment:', estab);
    setEstablishment(estab);
    setEstablishmentText(estab?.label || estab?.display_name || '');

    // Important: Effacer le service sélectionné
    setServiceObj(null);
    setHospitalService('');

    console.log('🏥 Establishment set, ID:', estab?.id, 'Type:', typeof estab?.id);
  };

  const onChangeService = useCallback(
    (srv) => {
      console.log('🏥 ManualBookingForm.onChangeService:', srv);
      setServiceObj(srv || null);
      setHospitalService(srv?.name || '');

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
          setNotesMedical((prevNotes) =>
            prevNotes ? `${prevNotes}\n${detailsText}` : detailsText
          );
        }
      }
    },
    [establishment]
  );

  // === State pour le montant (tarif préférentiel) ===
  const [amount, setAmount] = useState('');

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
          let label;
          if (c.is_institution && c.institution_name) {
            label = `🏥 ${c.institution_name}`;
          } else {
            // Pour les clients normaux, afficher nom + prénom
            label =
              `${c.user?.first_name ?? c.first_name ?? ''} ${
                c.user?.last_name ?? c.last_name ?? ''
              }`.trim() || `Client #${c.id}`;
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

  // === Clients ===
  const handleSelectClient = (clientObj) => {
    console.log('👤 Client sélectionné:', clientObj);
    setSelectedClient(clientObj);

    // 📍 Récupérer l'adresse exacte du client avec priorités
    let homeAddress = '';
    let homeGPS = { lat: null, lon: null };
    const client = clientObj?.raw;

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

    // 💰 Appliquer automatiquement le tarif préférentiel si disponible
    const preferentialRate = client?.preferential_rate;
    if (preferentialRate && preferentialRate > 0) {
      setAmount(preferentialRate.toString());
      console.log(`💰 Tarif préférentiel appliqué: ${preferentialRate} CHF`);
    } else {
      setAmount(''); // Pas de tarif préférentiel, laisser vide
    }
  };

  const loadClientOptions = useCallback(async (q) => {
    try {
      console.log('🔍 Recherche de clients avec query:', q);

      // Si pas de recherche, charger tous les clients (limité par le backend)
      const clients = q ? await searchClients(q) : await searchClients('');

      console.log('✅ Clients trouvés:', clients.length);

      const options = clients.map((c) => {
        // 🏥 Pour les institutions, afficher le nom de l'institution
        let label;
        if (c.is_institution && c.institution_name) {
          label = `🏥 ${c.institution_name}`;
        } else {
          // Pour les clients normaux, afficher nom + prénom
          label =
            `${c.user?.first_name ?? c.first_name ?? ''} ${
              c.user?.last_name ?? c.last_name ?? ''
            }`.trim() || `Client #${c.id}`;
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

  // === Notes médicales → extraction
  function handleNotesMedicalBlur(e) {
    const value = e.target.value;
    // ❌ Supprimé : setNotesMedical(value) car déjà géré par onChange

    const extracted = extractMedicalServiceInfo(value);

    if (extracted.medical_facility) setMedicalFacility(extracted.medical_facility);
    if (extracted.hospital_service) {
      setHospitalService(extracted.hospital_service);
      setServiceObj(null); // pas d'ID → on laisse vide ; l'utilisateur choisira dans la liste
    }
    if (extracted.doctor_name) setDoctorName(extracted.doctor_name);
    // building/floor -> concat dans notes
    let notes = value || '';
    if (extracted.building) notes += (notes ? '\n' : '') + extracted.building;
    if (extracted.floor) notes += (notes ? '\n' : '') + extracted.floor;
    if (notes !== value) setNotesMedical(notes);
  }

  // === Mutations API ===
  // Gestion de la création de client
  const handleSaveNewClient = async (clientData) => {
    try {
      const newClient = await createClient(clientData);
      queryClient.invalidateQueries(['clients']);
      setSelectedClient({
        value: newClient.id,
        label: `${newClient.user?.first_name ?? newClient.first_name ?? ''} ${
          newClient.user?.last_name ?? newClient.last_name ?? ''
        }`.trim(),
        raw: newClient,
      });
      setShowClientModal(false);
      toast.success('Client créé !');

      // Appliquer le tarif préférentiel et l'adresse si disponibles
      if (newClient.preferential_rate) {
        setAmount(newClient.preferential_rate.toString());
      }
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
      toast.error(err?.response?.data?.error || 'Erreur création client');
      throw err; // Pour que NewClientModal affiche l'erreur
    }
  };

  const bookingMutation = useMutation({
    mutationFn: createManualBooking,
    onSuccess: (data) => {
      toast.success('Réservation créée !');
      onSuccess?.(data);
    },
    onError: (err) => {
      console.error('createManualBooking error:', err?.response?.data || err);
      toast.error(
        err?.response?.data?.error ||
          err?.response?.data?.message ||
          `Erreur création réservation : ${err.message}`
      );
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

    // Vérifier que le montant est défini
    if (!amount || parseFloat(amount) <= 0) {
      toast.error('Veuillez saisir un montant valide pour la course');
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
      scheduled_time: scheduledTime,
      is_round_trip: !!isRoundTrip,
      amount: amount ? parseFloat(amount) : 0,

      // Champs médicaux (compat descendante)
      medical_facility:
        medicalFacility ||
        establishment?.display_name ||
        establishment?.label ||
        establishmentText || // ✅ fallback texte libre
        undefined,
      hospital_service: hospitalService || serviceObj?.name || undefined,
      doctor_name: doctorName || undefined,
      notes_medical: notesMedical || undefined,
      wheelchair_client_has: wheelchairOptions.clientHasWheelchair || undefined,
      wheelchair_need: wheelchairOptions.needWheelchair || undefined,

      // Nouveaux champs structurés
      establishment_id: establishment?.id ?? undefined,
      medical_service_id: serviceObj?.id ?? undefined,
      // Si aller-retour : toujours envoyer return_date, optionnellement return_time
      ...(isRoundTrip && returnDate
        ? {
            return_date: returnDate,
            return_time: returnTime || undefined, // Si vide, backend mettra 00:00
          }
        : {}),

      // Récurrence
      is_recurring: isRecurring,
      ...(isRecurring
        ? {
            recurrence_type: recurrenceType,
            recurrence_days: recurrenceType === 'custom' ? selectedDays : undefined,
            recurrence_end_date: recurrenceEndDate || undefined,
            occurrences: occurrences || undefined,
          }
        : {}),
    };

    console.log('[ManualBookingForm] payload:', payload);
    console.log('🔄 Récurrence activée:', isRecurring);
    console.log('📅 Type de récurrence:', recurrenceType);
    console.log('🗓️ Jours sélectionnés:', selectedDays);
    console.log("🔢 Nombre d'occurrences:", occurrences);

    bookingMutation.mutate(payload);
  };

  return (
    <div className={styles.formWrapper}>
      <form onSubmit={handleSubmit} className={styles.form}>
        {/* COLONNE GAUCHE */}
        <div className={styles.columnLeft}>
          {/* Client */}
          <div className={styles.formGroup}>
            <Label>Client *</Label>
            <AsyncCreatableSelect
              cacheOptions
              defaultOptions={defaultClientOptions}
              loadOptions={loadClientOptions}
              onChange={handleSelectClient}
              onCreateOption={(input) => {
                console.log("➕ Création d'un nouveau client:", input);
                setShowClientModal(true);
              }}
              value={selectedClient}
              placeholder="Rechercher un client…"
              formatCreateLabel={(i) => `➕ Créer "${i}"`}
              noOptionsMessage={({ inputValue }) =>
                inputValue
                  ? `Aucun client trouvé pour "${inputValue}"`
                  : 'Tapez pour rechercher un client'
              }
              loadingMessage={() => '🔍 Recherche en cours...'}
              menuPortalTarget={typeof window !== 'undefined' ? document.body : null}
              menuPosition="fixed"
              classNamePrefix="react-select"
            />
          </div>

          {/* Lieu de prise en charge */}
          <div className={styles.formGroup}>
            <Label>Lieu de prise en charge *</Label>
            <AddressAutocomplete
              name="pickup_location"
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
          </div>

          {/* Lieu de destination */}
          <div className={styles.formGroup}>
            <Label>Lieu de destination *</Label>
            <AddressAutocomplete
              name="dropoff_location"
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

                // ✅ Utiliser main_text pour Google Places (nom sans adresse), sinon label
                // main_text = "HUG Maternité" ou "Dr méd. Lakki Brahim"
                // label = "HUG Maternité, Boulevard de la Cluse, Genève, Suisse" (complet)
                let establishmentName = item.main_text || item.label || item.name || '';

                // Nettoyer le nom d'établissement pour retirer l'étage si présent
                const { floor: floorFromName, cleanedText: cleanedEstablishmentName } =
                  extractFloorInfo(establishmentName);
                establishmentName = cleanedEstablishmentName || establishmentName;

                // Construire l'adresse complète pour la destination
                let fullAddress = '';
                let floorInfo = floorFromName; // Étage extrait du nom

                if (item.address && (item.postcode || item.city)) {
                  // On a une vraie adresse structurée
                  // Nettoyer l'adresse pour retirer l'étage
                  const { floor: floorFromAddress, cleanedText: cleanedAddress } = extractFloorInfo(
                    item.address
                  );
                  floorInfo = floorInfo || floorFromAddress;

                  fullAddress = [cleanedAddress, item.postcode, item.city, item.country]
                    .filter(Boolean)
                    .join(' · ');
                } else {
                  // Pas d'adresse structurée, utiliser secondary_text ou label
                  const addressText = item.secondary_text || item.display_name || item.label || '';
                  const { floor: floorFromText, cleanedText: cleanedAddressText } =
                    extractFloorInfo(addressText);
                  floorInfo = floorInfo || floorFromText;
                  fullAddress = cleanedAddressText || addressText;
                }

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
                const looksLikeDoctor =
                  txt.includes('dr ') ||
                  txt.includes('dr.') ||
                  txt.includes('dr méd') ||
                  txt.includes('docteur') ||
                  txt.includes('cabinet') ||
                  txt.includes('médecin') ||
                  item.types?.includes('doctor');

                const looksLikeMedical =
                  txt.includes('hôpital') ||
                  txt.includes('hopital') ||
                  txt.includes('clinique') ||
                  txt.includes('centre médical') ||
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
                    setDoctorName(cleanName);
                    // NE PAS remplir l'établissement pour un docteur
                    setEstablishmentText('');
                    setMedicalFacility('');
                    // Ajouter l'étage dans les notes si présent
                    if (floorInfo) {
                      const floorNote = `🏢 ${floorInfo}`;
                      setNotesMedical((prevNotes) =>
                        prevNotes ? `${prevNotes}\n${floorNote}` : floorNote
                      );
                    }
                  } else {
                    // Pour Photon/autre, utiliser l'extraction
                    const extracted = extractMedicalServiceInfo(establishmentName);
                    setDoctorName(extracted.doctor_name || establishmentName.split(',')[0].trim());
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
                    // Ajouter l'étage dans les notes si présent
                    if (floorInfo) {
                      const floorNote = `🏢 ${floorInfo}`;
                      setNotesMedical((prevNotes) =>
                        prevNotes ? `${prevNotes}\n${floorNote}` : floorNote
                      );
                    }
                  } else {
                    // Pour Photon/autre, utiliser l'extraction
                    const extracted = extractMedicalServiceInfo(establishmentName);
                    setEstablishmentText(extracted.medical_facility || establishmentName);
                    setMedicalFacility(extracted.medical_facility || establishmentName);
                    if (extracted.doctor_name) setDoctorName(extracted.doctor_name);
                    if (extracted.hospital_service) setHospitalService(extracted.hospital_service);
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
                      // Ajouter l'étage si présent
                      if (floorInfo) {
                        locationNote += `\n🏢 ${floorInfo}`;
                      }
                      setNotesMedical((prevNotes) =>
                        prevNotes ? `${prevNotes}\n${locationNote}` : locationNote
                      );
                      // Ne PAS remplir les champs médicaux
                      setEstablishmentText('');
                      setMedicalFacility('');
                      setDoctorName('');
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
                      if (extracted.doctor_name) setDoctorName(extracted.doctor_name);
                      if (extracted.hospital_service)
                        setHospitalService(extracted.hospital_service || '');
                    }
                  }
                }

                // Ne pas appeler setHospitalService/setServiceObj ici pour éviter d'écraser
                setServiceObj(null);
              }}
              placeholder="Saisir ou choisir l'adresse"
            />
          </div>

          {/* Date & heure */}
          <div className={styles.formGroup}>
            <Label>Date &amp; heure *</Label>
            <Input
              type="datetime-local"
              name="scheduled_time"
              value={scheduledTime}
              onChange={(e) => setScheduledTime(e.target.value)}
              required
              min={minLocalDatetime}
            />

            <div className={styles.checkboxGroup}>
              <input
                type="checkbox"
                id="roundtrip"
                checked={isRoundTrip}
                onChange={(e) => setIsRoundTrip(e.target.checked)}
                className={styles.checkbox}
              />
              <Label htmlFor="roundtrip" className={styles.checkboxLabel}>
                Trajet aller-retour
              </Label>
            </div>

            {isRoundTrip && (
              <div className={styles.returnTimeGroup}>
                <Label>Date du retour</Label>
                <Input
                  type="date"
                  name="return_date"
                  value={returnDate}
                  onChange={(e) => setReturnDate(e.target.value)}
                  placeholder="Date du retour"
                />

                <Label style={{ marginTop: '12px' }}>Heure de retour (optionnel)</Label>
                <Input
                  type="time"
                  name="return_time"
                  value={returnTime}
                  onChange={(e) => setReturnTime(e.target.value)}
                  placeholder="Laisser vide pour « Heure à confirmer »"
                />
                <small
                  style={{
                    color: '#64748b',
                    fontSize: '0.85rem',
                    marginTop: '4px',
                    display: 'block',
                  }}
                >
                  💡 Si l'heure n'est pas définie, elle pourra être planifiée plus tard
                </small>
              </div>
            )}

            {/* Récurrence */}
            <div className={styles.checkboxGroup}>
              <input
                type="checkbox"
                id="recurring"
                checked={isRecurring}
                onChange={(e) => setIsRecurring(e.target.checked)}
                className={styles.checkbox}
              />
              <Label htmlFor="recurring" className={styles.checkboxLabel}>
                🔄 Réservation récurrente
              </Label>
            </div>

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
                  <Input
                    type="date"
                    value={recurrenceEndDate}
                    onChange={(e) => setRecurrenceEndDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Montant */}
          <div className={styles.formGroup}>
            <div className={styles.amountLabel}>
              <Label>Montant (optionnel)</Label>
              {amount && parseFloat(amount) > 0 && (
                <span className={styles.preferentialBadge}>💰 Tarif préférentiel</span>
              )}
            </div>
            <Input
              type="number"
              name="amount"
              step="0.01"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="Ex: 45.00"
            />
            {estimatedDuration && (
              <div className={styles.estimatedDurationBadge}>
                ⏱️ Durée estimée : <strong>{estimatedDuration} min</strong>
              </div>
            )}
          </div>
        </div>

        {/* COLONNE DROITE - Informations médicales */}
        <div className={styles.columnRight}>
          {/* Section médicale (optionnelle) */}
          <div className={styles.medicalSection}>
            <h3 className={styles.medicalSectionTitle}>🏥 Informations médicales (optionnel)</h3>

            {/* Établissement médical */}
            <div className={styles.medicalFormGroup}>
              <Label>Établissement médical</Label>
              <EstablishmentSelect
                value={establishmentText}
                onChange={handleEstablishmentTextChange}
                onPickEstablishment={onPickEstablishment}
                placeholder="HUG, Clinique La Colline, Grangettes, La Tour…"
              />
            </div>

            {/* Service médical (visible seulement si établissement sélectionné) */}
            {establishment?.id && (
              <div className={styles.medicalFormGroup}>
                <Label>Service / Département</Label>
                <ServiceSelect
                  key={establishment?.id}
                  establishmentId={establishment?.id}
                  value={serviceObj}
                  onChange={onChangeService}
                  placeholder="Ex: Urgences adultes, Cardiologie…"
                />
              </div>
            )}

            {/* Nom du médecin */}
            <div className={styles.medicalFormGroup}>
              <Label>Nom du médecin</Label>
              <Input
                type="text"
                name="doctor_name"
                value={doctorName}
                onChange={(e) => setDoctorName(e.target.value)}
                placeholder="Ex : Dr Dupont"
              />
            </div>

            {/* Notes médicales */}
            <div className={styles.medicalFormGroup}>
              <Label>Notes médicales</Label>
              <textarea
                name="notes_medical"
                value={notesMedical}
                onChange={(e) => setNotesMedical(e.target.value)}
                onBlur={handleNotesMedicalBlur}
                placeholder="Instructions particulières, bâtiment, étage…"
                rows={3}
                className={styles.textarea}
              />
            </div>

            {/* Options chaise roulante */}
            <div className={styles.medicalFormGroup}>
              <Label>Options chaise roulante</Label>

              <div className={styles.checkboxGroup}>
                <input
                  type="checkbox"
                  id="clientHasWheelchair"
                  checked={wheelchairOptions.clientHasWheelchair}
                  onChange={(e) =>
                    setWheelchairOptions({
                      clientHasWheelchair: e.target.checked,
                      needWheelchair: e.target.checked ? false : wheelchairOptions.needWheelchair,
                    })
                  }
                  className={styles.checkbox}
                />
                <Label htmlFor="clientHasWheelchair" className={styles.checkboxLabel}>
                  ♿ Le client est en chaise roulante
                </Label>
              </div>

              <div className={styles.checkboxGroup}>
                <input
                  type="checkbox"
                  id="needWheelchair"
                  checked={wheelchairOptions.needWheelchair}
                  onChange={(e) =>
                    setWheelchairOptions({
                      needWheelchair: e.target.checked,
                      clientHasWheelchair: e.target.checked
                        ? false
                        : wheelchairOptions.clientHasWheelchair,
                    })
                  }
                  className={styles.checkbox}
                />
                <Label htmlFor="needWheelchair" className={styles.checkboxLabel}>
                  🏥 Prendre une chaise roulante
                </Label>
              </div>
            </div>
          </div>
        </div>

        {/* BOUTON DE SOUMISSION - Sur toute la largeur */}
        <button
          type="submit"
          className={styles.submitButton}
          disabled={bookingMutation.isLoading}
          onClick={() => console.log('[ManualBookingForm] submit button clicked')}
        >
          {bookingMutation.isLoading ? '⏳ Création…' : 'Créer la réservation'}
        </button>
      </form>

      {showClientModal && (
        <NewClientModal onClose={() => setShowClientModal(false)} onSave={handleSaveNewClient} />
      )}
    </div>
  );
}
