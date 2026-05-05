import { describe, expect, it } from "@jest/globals";
import {
  buildRideCreatePayload,
  parseMedicalHintsFromAddress,
  parseSimulationAmount,
} from "./rideCreateHelpers";

describe("rideCreateHelpers", () => {
  describe("parseSimulationAmount", () => {
    it("lit le montant racine", () => {
      expect(parseSimulationAmount({ amount: 45.5 })).toBe(45.5);
    });

    it("lit le montant dans pricing.amount", () => {
      expect(parseSimulationAmount({ pricing: { amount: "62.40" } })).toBe(62.4);
    });

    it("ignore les montants invalides", () => {
      expect(parseSimulationAmount({ amount: 0 })).toBeNull();
      expect(parseSimulationAmount({ amount: "abc" })).toBeNull();
      expect(parseSimulationAmount(null)).toBeNull();
    });
  });

  describe("parseMedicalHintsFromAddress", () => {
    it("détecte établissement + service urgences", () => {
      const hints = parseMedicalHintsFromAddress("HUG Urgences, Rue Gabrielle-Perret-Gentil 4");
      expect(hints.establishment).toBe("HUG Urgences");
      expect(hints.hospitalService).toBe("Urgences");
    });

    it("détecte un médecin", () => {
      const hints = parseMedicalHintsFromAddress("Dr Dupont, Clinique de Carouge");
      expect(hints.doctorName).toBe("Dr Dupont");
    });

    it("détecte l'étage dans les notes", () => {
      const hints = parseMedicalHintsFromAddress("Clinique Test, étage 2, Genève");
      expect(hints.notesMedical).toBe("Étage: étage 2");
    });
  });

  describe("buildRideCreatePayload scenario snapshots", () => {
    it("scénario 1: client domicile facturation patient", () => {
      const payload = buildRideCreatePayload({
        structuredPayloadEnabled: true,
        clientId: 1,
        pickup: "Rue du Rhône 1, Genève",
        dropoff: "Gare Cornavin, Genève",
        pickupAddress: { label: "Rue du Rhône 1, Genève", placeId: "p1", latitude: 46.2, longitude: 6.15 },
        dropoffAddress: { label: "Gare Cornavin, Genève", placeId: "p2", latitude: 46.21, longitude: 6.14 },
        scheduledTime: "2026-05-05T14:30:00",
        isRoundTrip: false,
        recurrence: "none",
        notesMedical: "",
        establishment: "",
        hospitalService: "",
        doctorName: "",
        pickupAccessNotes: "",
        dropoffAccessNotes: "",
        wheelchairClient: false,
        wheelchairProvide: false,
        internalNotes: "",
        notesMax: 500,
        amountInput: "45.00",
        amountSource: "manual",
        pricingProfileId: 11,
        pricingProfileVersionId: 22,
        isMaterialDelivery: false,
        deliveryDescription: "",
        returnScheduledAt: "",
        billToPatient: false,
        hasActiveStay: false,
        clinicBillingPartyId: null,
      });
      expect(payload).toMatchInlineSnapshot(`
{
  "amount": 45,
  "amount_source": "manual",
  "client_id": 1,
  "dropoff_address": {
    "label": "Gare Cornavin, Genève",
    "lat": 46.21,
    "lon": 6.14,
    "place_id": "p2",
  },
  "dropoff_lat": 46.21,
  "dropoff_lon": 6.14,
  "is_return": false,
  "notes_medical": null,
  "pickup_address": {
    "label": "Rue du Rhône 1, Genève",
    "lat": 46.2,
    "lon": 6.15,
    "place_id": "p1",
  },
  "pickup_lat": 46.2,
  "pickup_lon": 6.15,
  "pricing_profile_id": 11,
  "pricing_profile_version_id": 22,
  "scheduled_time": "2026-05-05T14:30:00",
}
`);
    });

    it("scénario 2: hospitalisé facturation clinique", () => {
      const payload = buildRideCreatePayload({
        structuredPayloadEnabled: true,
        clientId: 2,
        pickup: "Clinique Arcades, Genève",
        dropoff: "HUG, Genève",
        pickupAddress: { label: "Clinique Arcades, Genève", placeId: null, latitude: 46.22, longitude: 6.16 },
        dropoffAddress: { label: "HUG, Genève", placeId: null, latitude: 46.23, longitude: 6.17 },
        scheduledTime: "2026-05-06T09:00:00",
        isRoundTrip: false,
        recurrence: "none",
        notesMedical: "Patient fragile",
        establishment: "Clinique Arcades",
        hospitalService: "Cardiologie",
        doctorName: "",
        pickupAccessNotes: "Étage 2 · Chambre 203",
        dropoffAccessNotes: "",
        wheelchairClient: true,
        wheelchairProvide: false,
        internalNotes: "",
        notesMax: 500,
        amountInput: "62.40",
        amountSource: "preferential",
        pricingProfileId: 11,
        pricingProfileVersionId: 22,
        isMaterialDelivery: false,
        deliveryDescription: "",
        returnScheduledAt: "",
        billToPatient: false,
        hasActiveStay: true,
        clinicBillingPartyId: 99,
      });
      expect(payload).toMatchInlineSnapshot(`
{
  "amount": 62.4,
  "amount_source": "preferential",
  "billing_party_id": 99,
  "client_id": 2,
  "dropoff_address": {
    "label": "HUG, Genève",
    "lat": 46.23,
    "lon": 6.17,
    "place_id": null,
  },
  "dropoff_lat": 46.23,
  "dropoff_lon": 6.17,
  "hospital_service": "Cardiologie",
  "is_return": false,
  "medical_facility": "Clinique Arcades",
  "notes_medical": "Patient fragile",
  "pickup_access_notes": "Étage 2 · Chambre 203",
  "pickup_address": {
    "label": "Clinique Arcades, Genève",
    "lat": 46.22,
    "lon": 6.16,
    "place_id": null,
  },
  "pickup_lat": 46.22,
  "pickup_lon": 6.16,
  "pricing_profile_id": 11,
  "pricing_profile_version_id": 22,
  "scheduled_time": "2026-05-06T09:00:00",
  "wheelchair_client_has": true,
}
`);
    });

    it("scénario 3: hospitalisé override facturation patient", () => {
      const payload = buildRideCreatePayload({
        structuredPayloadEnabled: true,
        clientId: 3,
        pickup: "HUG, Genève",
        dropoff: "Domicile, Nyon",
        pickupAddress: { label: "HUG, Genève", placeId: null, latitude: 46.24, longitude: 6.18 },
        dropoffAddress: { label: "Domicile, Nyon", placeId: null, latitude: 46.38, longitude: 6.24 },
        scheduledTime: "2026-05-07T10:00:00",
        isRoundTrip: true,
        recurrence: "weekly",
        notesMedical: "",
        establishment: "HUG",
        hospitalService: "Urgences",
        doctorName: "Dr Dupont",
        pickupAccessNotes: "",
        dropoffAccessNotes: "",
        wheelchairClient: false,
        wheelchairProvide: true,
        internalNotes: "Retour domicile",
        notesMax: 500,
        amountInput: "88",
        amountSource: "simulated",
        pricingProfileId: 11,
        pricingProfileVersionId: 22,
        isMaterialDelivery: false,
        deliveryDescription: "",
        returnScheduledAt: "2026-05-07T12:00:00",
        billToPatient: true,
        hasActiveStay: true,
        clinicBillingPartyId: 101,
      });
      expect(payload).toMatchInlineSnapshot(`
{
  "amount": 88,
  "amount_source": "simulated",
  "bill_to_patient": true,
  "client_id": 3,
  "doctor_name": "Dr Dupont",
  "dropoff_address": {
    "label": "Domicile, Nyon",
    "lat": 46.38,
    "lon": 6.24,
    "place_id": null,
  },
  "dropoff_lat": 46.38,
  "dropoff_lon": 6.24,
  "hospital_service": "Urgences",
  "is_recurring": true,
  "is_return": true,
  "medical_facility": "HUG",
  "notes": "Retour domicile",
  "notes_medical": null,
  "pickup_address": {
    "label": "HUG, Genève",
    "lat": 46.24,
    "lon": 6.18,
    "place_id": null,
  },
  "pickup_lat": 46.24,
  "pickup_lon": 6.18,
  "pricing_profile_id": 11,
  "pricing_profile_version_id": 22,
  "recurrence_type": "weekly",
  "return_date": "2026-05-07",
  "return_time": "2026-05-07T12:00:00",
  "scheduled_time": "2026-05-07T10:00:00",
  "wheelchair_need": true,
}
`);
    });
  });
});
