// frontend/src/pages/company/Dashboard/components/ManualBookingForm.jsx (fixed)

import React, { useState, useCallback } from "react";
import AsyncCreatableSelect from "react-select/async-creatable";
import Modal from "../../../../components/common/Modal";
import NewClientForm from "./NewClientForm";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  createManualBooking,
  searchClients,
  createClient,
} from "../../../../services/companyService";
import { Input } from "./ui/Input";
import { Label } from "./ui/Label";
import AddressAutocomplete from "../../../../components/common/AddressAutocomplete";

// ⬇️ Assure-toi que ces chemins correspondent à ta structure réelle
import EstablishmentSelect from "../../../../components/common/EstablishmentSelect";
import ServiceSelect from "../../../../components/common/ServiceSelect";

import { extractMedicalServiceInfo } from "../../../../utils/medicalExtract";
import { toast } from "sonner";
import "./ManualBookingForm.css";

export default function ManualBookingForm({ onSuccess }) {
  const queryClient = useQueryClient();

  // --- Établissement (objet choisi) + champ texte (saisie en cours)
  const [establishment, setEstablishment] = React.useState(null); // { id, label?, display_name?, address, ... }
  const [establishmentText, setEstablishmentText] = useState(""); // texte tapé
  const [serviceObj, setServiceObj] = React.useState(null); // { id, name }

  // --- Adresses + coordonnées
  const [pickupLocation, setPickupLocation] = useState("");
  const [dropoffLocation, setDropoffLocation] = useState("");
  const [pickupCoords, setPickupCoords] = useState({ lat: null, lon: null });
  const [dropoffCoords, setDropoffCoords] = useState({ lat: null, lon: null });

  // --- Client
  const [selectedClient, setSelectedClient] = useState(null);
  const [showClientModal, setShowClientModal] = useState(false);
  const [newClientName, setNewClientName] = useState("");

  // --- Aller-retour
  const [isRoundTrip, setIsRoundTrip] = useState(false);
  const [returnTime, setReturnTime] = useState("");

  // --- Infos médicales libres (toujours disponibles)
  const [medicalFacility, setMedicalFacility] = useState("");
  const [doctorName, setDoctorName] = useState("");
  const [hospitalService, setHospitalService] = useState(""); // restera synchronisé avec serviceName
  const [notesMedical, setNotesMedical] = useState("");

  // Helper: min pour <input type="datetime-local"> au format local (pas UTC)
  const minLocalDatetime = (() => {
    const d = new Date(Date.now() + 5 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(
      d.getDate()
    )}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  })();

  // === Handlers établissement / service ===
  const handleEstablishmentTextChange = (newText) => {
    setEstablishmentText(newText);
    // Si l'utilisateur modifie le texte, on efface l'objet sélectionné
    if (
      establishment &&
      newText !== (establishment.label || establishment.display_name || "")
    ) {
      setEstablishment(null);
      setServiceObj(null);
      setHospitalService("");
    }
  };

  const onPickEstablishment = (estab) => {
    console.log("🏥 ManualBookingForm.onPickEstablishment:", estab);
    setEstablishment(estab);
    setEstablishmentText(estab?.label || estab?.display_name || "");

    // Important: Effacer le service sélectionné
    setServiceObj(null);
    setHospitalService("");

    console.log(
      "🏥 Establishment set, ID:",
      estab?.id,
      "Type:",
      typeof estab?.id
    );
  };

  const onChangeService = useCallback(
    (srv) => {
      console.log("🏥 ManualBookingForm.onChangeService:", srv);
      setServiceObj(srv || null);
      setHospitalService(srv?.name || "");

      // --- Logique pour la destination ---
      if (srv && establishment) {
        setDropoffLocation(establishment.address || "");
        setDropoffCoords({
          lat: establishment.lat ?? null,
          lon: establishment.lon ?? null,
        });
      }

      // --- NOUVEAU : Ajoute les détails du service dans les notes ---
      if (srv) {
        const details = [
          srv.building,
          srv.site_note,
          srv.floor,
          srv.phone,
        ].filter(Boolean); // Garde uniquement les valeurs qui existent

        if (details.length > 0) {
          const detailsText = details.join(" - ");
          // Ajoute les détails aux notes existantes (s'il y en a)
          setNotesMedical((prevNotes) =>
            prevNotes ? `${prevNotes}\n${detailsText}` : detailsText
          );
        }
      }
    },
    [establishment]
  );

  // === Clients ===
  const handleSelectClient = (clientObj) => {
    setSelectedClient(clientObj);
    const homeAddress =
      clientObj?.raw?.address || clientObj?.raw?.user?.address || "";
    if (homeAddress) {
      setPickupLocation(homeAddress);
      setPickupCoords({ lat: null, lon: null }); // pas de coords connues ici
    }
  };

  const loadClientOptions = useCallback(async (q) => {
    if (!q) return [];
    try {
      const clients = await searchClients(q);
      return clients.map((c) => ({
        value: c.id,
        label: `${c.user?.first_name ?? c.first_name ?? ""} ${
          c.user?.last_name ?? c.last_name ?? ""
        }`.trim(),
        raw: c,
      }));
    } catch (e) {
      console.error("searchClients error", e);
      return [];
    }
  }, []);

  // === Notes médicales → extraction
  function handleNotesMedicalBlur(e) {
    const value = e.target.value;
    setNotesMedical(value);

    const extracted = extractMedicalServiceInfo(value);

    if (extracted.medical_facility)
      setMedicalFacility(extracted.medical_facility);
    if (extracted.hospital_service) {
      setHospitalService(extracted.hospital_service);
      setServiceObj(null); // pas d'ID → on laisse vide ; l’utilisateur choisira dans la liste
    }
    if (extracted.doctor_name) setDoctorName(extracted.doctor_name);
    // building/floor -> concat dans notes
    let notes = value || "";
    if (extracted.building) notes += (notes ? "\n" : "") + extracted.building;
    if (extracted.floor) notes += (notes ? "\n" : "") + extracted.floor;
    if (notes !== value) setNotesMedical(notes);
  }

  // === Mutations API ===
  const clientMutation = useMutation({
    mutationFn: createClient,
    onSuccess: (client) => {
      queryClient.invalidateQueries(["clients"]);
      setSelectedClient({
        value: client.id,
        label: `${client.user?.first_name ?? client.first_name ?? ""} ${
          client.user?.last_name ?? client.last_name ?? ""
        }`.trim(),
        raw: client,
      });
      setShowClientModal(false);
      toast.success("Client créé !");
    },
    onError: (err) => {
      console.error("API createClient error:", err?.response?.data || err);
      toast.error(err?.response?.data?.error || "Erreur création client");
    },
  });

  const bookingMutation = useMutation({
    mutationFn: createManualBooking,
    onSuccess: (data) => {
      toast.success("Réservation créée !");
      onSuccess?.(data);
    },
    onError: (err) => {
      console.error("createManualBooking error:", err?.response?.data || err);
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
    console.log("[ManualBookingForm] submit click");

    if (!selectedClient) {
      toast.error("Veuillez sélectionner un client");
      return;
    }

    // 🔒 Sécurité : si un texte d’établissement est présent mais aucun établissement choisi (pas d’id)
    // On tolère le texte libre : on prévient juste
    if (establishmentText && !establishment?.id) {
      console.warn(
        "[ManualBookingForm] établissement en texte libre :",
        establishmentText
      );
    }

    // ⚠️ Toujours lire via FormData pour être indépendant des composants UI
    const formEl = e.currentTarget; // c'est <form>, garanti
    const fd = new FormData(formEl);
    const localDT = fd.get("scheduled_time"); // 'YYYY-MM-DDTHH:mm'
    if (!localDT) {
      toast.error("Veuillez sélectionner Date & heure");
      return;
    }

    const payload = {
      client_id: selectedClient.value,
      pickup_location: pickupLocation,
      dropoff_location: dropoffLocation,
      pickup_lat: pickupCoords.lat ?? undefined,
      pickup_lon: pickupCoords.lon ?? undefined,
      dropoff_lat: dropoffCoords.lat ?? undefined,
      dropoff_lon: dropoffCoords.lon ?? undefined,
      scheduled_time: String(localDT),
      is_round_trip: !!isRoundTrip,
      amount: fd.get("amount") ? parseFloat(fd.get("amount")) : 0,

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

      // Nouveaux champs structurés
      establishment_id: establishment?.id ?? undefined,
      medical_service_id: serviceObj?.id ?? undefined,
      ...(isRoundTrip && returnTime ? { return_time: returnTime } : {}),
    };

    console.log("[ManualBookingForm] payload:", payload);
    bookingMutation.mutate(payload);
  };

  return (
    <div className="manual-booking-form-wrapper">
      <form onSubmit={handleSubmit} className="manual-booking-form space-y-4">
        {/* Client */}
        <Label>Client</Label>
        <AsyncCreatableSelect
          cacheOptions
          defaultOptions
          loadOptions={loadClientOptions}
          onChange={handleSelectClient}
          onCreateOption={(input) => {
            setNewClientName(input);
            setShowClientModal(true);
          }}
          value={selectedClient}
          placeholder="Rechercher un client…"
          formatCreateLabel={(i) => `Créer “${i}”`}
          /* ✅ Menu cliquable dans un Modal (comme ServiceSelect) */
          menuPortalTarget={
            typeof window !== "undefined" ? document.body : null
          }
          menuPosition="fixed"
          styles={{
            menuPortal: (base) => ({ ...base, zIndex: 9999 }),
            menu: (base) => ({ ...base, zIndex: 9999 }),
          }}
        />

        {/* Prise en charge */}
        <Label>Lieu de prise en charge</Label>
        <AddressAutocomplete
          name="pickup_location"
          value={pickupLocation}
          onChange={(e) => {
            setPickupLocation(e.target.value);
            setPickupCoords({ lat: null, lon: null });
          }}
          onSelect={(item) => {
            setPickupLocation(item.label || item.address || "");
            setPickupCoords({ lat: item.lat ?? null, lon: item.lon ?? null });
          }}
          placeholder="Saisir ou choisir l'adresse"
          required
          className="w-full px-3 py-2 border rounded"
        />

        {/* Destination */}
        <Label>Lieu de destination</Label>
        <AddressAutocomplete
          name="dropoff_location"
          value={dropoffLocation}
          onChange={(e) => {
            setDropoffLocation(e.target.value);
            setDropoffCoords({ lat: null, lon: null });
          }}
          onSelect={(item) => {
            const mainText =
              item.label ||
              [item.address, item.postcode, item.city, item.country]
                .filter(Boolean)
                .join(" ") ||
              "";
            setDropoffLocation(mainText);
            setDropoffCoords({ lat: item.lat ?? null, lon: item.lon ?? null });

            const extracted = extractMedicalServiceInfo(mainText);

            const txt = (item.label || "").toLowerCase();
            const looksLikeDoctor =
              txt.includes("dr ") ||
              txt.includes("docteur") ||
              txt.includes("cabinet") ||
              txt.includes("médecin");

            if (looksLikeDoctor) {
              setDoctorName(extracted.doctor_name || item.label || "");
              setMedicalFacility(extracted.medical_facility || "");
            } else {
              setMedicalFacility(
                extracted.medical_facility || item.label || ""
              );
              if (extracted.doctor_name) setDoctorName(extracted.doctor_name);
            }

            setHospitalService(extracted.hospital_service || "");
            setServiceObj(null); // pas d'ID connu depuis ce champ libre

            let notes = notesMedical || "";
            if (extracted.building)
              notes += (notes ? "\n" : "") + extracted.building;
            if (extracted.floor) notes += (notes ? "\n" : "") + extracted.floor;
            setNotesMedical(notes);
          }}
          placeholder="Saisir ou choisir l'adresse"
        />

        {/* Établissement médical (structuré) */}
        <Label>Établissement médical</Label>
        <EstablishmentSelect
          value={establishmentText}
          onChange={handleEstablishmentTextChange}
          onPickEstablishment={onPickEstablishment}
          placeholder="HUG, Clinique La Colline, Grangettes, La Tour…"
        />

        {/* Service médical (lié à l’établissement) */}
        <ServiceSelect
          key={establishment?.id}
          establishmentId={establishment?.id}
          value={serviceObj}
          onChange={onChangeService}
          placeholder="Choisir le service (ex. Urgences adultes, Cardiologie…)"
        />

        {/* Champs complémentaires (libres) */}
        <Label>Nom du médecin (optionnel)</Label>
        <Input
          type="text"
          name="doctor_name"
          value={doctorName}
          onChange={(e) => setDoctorName(e.target.value)}
          placeholder="Ex : Dr Dupont"
        />

        <Label>Notes médicales (optionnel)</Label>
        {/* Remplacement d'un Input avec as="textarea" par un vrai textarea pour compat */}
        <textarea
          name="notes_medical"
          value={notesMedical}
          onChange={(e) => setNotesMedical(e.target.value)}
          onBlur={handleNotesMedicalBlur}
          placeholder="Ajouter un commentaire, instructions particulières…"
          rows={3}
          className="w-full px-3 py-2 border rounded"
        />

        {/* Date/heure & Aller-retour */}
        <Label>Date &amp; heure</Label>
        <Input
          type="datetime-local"
          name="scheduled_time"
          required
          min={minLocalDatetime}
        />

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="roundtrip"
            checked={isRoundTrip}
            onChange={(e) => setIsRoundTrip(e.target.checked)}
          />
          <Label htmlFor="roundtrip">Trajet aller-retour</Label>
        </div>

        {isRoundTrip && (
          <>
            <Label>Heure de retour (optionnel)</Label>
            <Input
              type="datetime-local"
              name="return_time"
              value={returnTime}
              onChange={(e) => setReturnTime(e.target.value)}
              min={minLocalDatetime}
            />
          </>
        )}

        <Label>Montant (optionnel)</Label>
        <Input type="number" name="amount" step="0.01" />

        <button
          type="submit"
          className="px-4 py-2 rounded bg-black text-white disabled:opacity-60"
          disabled={bookingMutation.isLoading}
          onClick={() =>
            console.log("[ManualBookingForm] submit button clicked")
          }
        >
          {bookingMutation.isLoading ? "Création…" : "Créer réservation"}{" "}
        </button>
      </form>

      {showClientModal && (
        <Modal onClose={() => setShowClientModal(false)}>
          <NewClientForm
            initialName={newClientName}
            onCancel={() => setShowClientModal(false)}
            onSubmit={(data) => clientMutation.mutate(data)}
          />
        </Modal>
      )}
    </div>
  );
}
