import { trackClientKpiEvent } from "./statusEvents";

export const STATUS_DICTIONARY_VERSION = "1.0.0";

type StatusConfig = {
  label: string;
  description: string;
  ctas: string[];
  priority: number;
  terminal: boolean;
};

const STATUS_MAP: Record<string, StatusConfig> = {
  pending: {
    label: "Demande envoyee",
    description: "Demande recue, en attente de traitement.",
    ctas: ["voir", "annuler"],
    priority: 3,
    terminal: false,
  },
  awaiting_client_payment: {
    label: "Paiement requis",
    description: "Paiement necessaire avant progression.",
    ctas: ["payer", "annuler"],
    priority: 4,
    terminal: false,
  },
  accepted: {
    label: "Transport confirme",
    description: "Une entreprise a confirme votre demande.",
    ctas: ["voir", "contacter"],
    priority: 4,
    terminal: false,
  },
  in_progress: {
    label: "En cours",
    description: "Le transport est en execution.",
    ctas: ["suivre", "contacter"],
    priority: 3,
    terminal: false,
  },
  completed: {
    label: "Terminee",
    description: "Le transport est termine.",
    ctas: ["recommander"],
    priority: 2,
    terminal: true,
  },
  cancelled: {
    label: "Refusee ou indisponible",
    description: "La demande est annulee ou non executable.",
    ctas: ["reessayer", "support"],
    priority: 4,
    terminal: true,
  },
  rejected: {
    label: "Refusee ou indisponible",
    description: "La demande est annulee ou non executable.",
    ctas: ["reessayer", "support"],
    priority: 4,
    terminal: true,
  },
};

export function getClientStatusUx(rawStatus: string | null | undefined) {
  const normalized = String(rawStatus ?? "").trim().toLowerCase();
  const found = STATUS_MAP[normalized];
  if (found) return found;
  trackClientKpiEvent("status_dictionary_mismatch_event", {
    surface: "mobile",
    status: normalized || null,
    version: STATUS_DICTIONARY_VERSION,
  });
  return {
    label: "Statut en cours de mise a jour",
    description: "Le statut est en cours de synchronisation.",
    ctas: ["refresh"],
    priority: 1,
    terminal: false,
  };
}

