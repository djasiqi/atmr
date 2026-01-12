import { enterpriseApi, hasValidToken, retryWithBackoff } from "./enterpriseAuth";

const API_BASE = "/partnerships";

// ============================================================================
// TYPES
// ============================================================================

export interface Partnership {
  id: string;
  owner_company_id: string;
  partner_company_id: string;
  owner_company_name: string;
  partner_company_name: string;
  status: "PENDING" | "ACCEPTED" | "REJECTED";
  is_active: boolean;
  created_at: string;
  updated_at?: string;
}

export interface Transfer {
  id: string;
  partnership_id: string;
  booking_id: string;
  status: "PENDING" | "ACCEPTED" | "REJECTED";
  proposed_at: string;
  responded_at?: string;
  reason?: string;
  transfer_model?: string;
  // Informations sur la course (si incluses)
  booking?: {
    id: string;
    pickup_location: string;
    dropoff_location: string;
    scheduled_time: string;
    client_name?: string;
  };
  // Informations sur le partenariat
  partnership?: Partnership;
}

export interface TransferListFilters {
  partnership_id?: string;
  status?: "PENDING" | "ACCEPTED" | "REJECTED";
}

// ============================================================================
// API CALLS
// ============================================================================

/**
 * Récupère la liste des partenariats disponibles pour transférer une course
 * Retourne uniquement les partenariats acceptés et actifs où l'entreprise est propriétaire
 */
export const fetchPartnershipsForTransfer = async (): Promise<Partnership[]> => {
  try {
    console.log("[partnershipService] Récupération des partenariats pour transfert...");
    const response = await enterpriseApi.get<{ data: Partnership[] }>(`${API_BASE}/for-transfer`);
    const partnerships = response.data?.data || response.data;
    console.log(`[partnershipService] ${partnerships.length} partenariats disponibles`);
    return partnerships;
  } catch (error: any) {
    console.error("[partnershipService] Erreur lors de la récupération des partenariats:", error);
    console.error("[partnershipService] Détails:", error?.response?.data);
    throw error;
  }
};

/**
 * Propose le transfert d'une course à une entreprise partenaire
 * @param partnershipId ID du partenariat (entreprise partenaire)
 * @param bookingId ID de la course à transférer
 * @param transferModel (Optionnel) Modèle de transfert ("FULL" ou "PARTIAL")
 * @returns Les détails du transfert créé
 */
export const proposeTransfer = async (
  partnershipId: string,
  bookingId: string,
  transferModel?: string
): Promise<Transfer> => {
  // ✅ Guard Pattern : Vérifier qu'on a un token valide avant d'envoyer
  if (!(await hasValidToken())) {
    throw new Error("Aucun token valide. Veuillez vous reconnecter.");
  }

  try {
    console.log("[partnershipService] Proposition de transfert:", {
      partnershipId,
      bookingId,
      transferModel,
    });

    const payload: any = { booking_id: bookingId };
    if (transferModel) {
      payload.transfer_model = transferModel;
    }

    const response = await enterpriseApi.post<{ data: Transfer }>(
      `${API_BASE}/${partnershipId}/transfers`,
      payload
    );

    const transfer = response.data?.data || response.data;
    console.log("[partnershipService] Transfert proposé avec succès:", transfer.id);
    return transfer;
  } catch (error: any) {
    console.error("[partnershipService] Erreur lors de la proposition de transfert:", error);
    console.error("[partnershipService] Détails:", error?.response?.data);
    throw error;
  }
};

/**
 * Accepte un transfert de course proposé par une entreprise partenaire
 * @param transferId ID du transfert à accepter
 * @returns Les détails du transfert accepté
 */
export const acceptTransfer = async (transferId: string): Promise<Transfer> => {
  // 🔄 Retry avec backoff pour résilience réseau
  return retryWithBackoff(
    async () => {
      // ✅ Guard Pattern : Vérifier qu'on a un token valide avant d'envoyer
      if (!(await hasValidToken())) {
        throw new Error("Aucun token valide. Veuillez vous reconnecter.");
      }

      try {
        console.log("[partnershipService] Acceptation du transfert:", transferId);

        const response = await enterpriseApi.post<{ data: Transfer }>(
          `${API_BASE}/transfers/${transferId}/accept`
        );

        const transfer = response.data?.data || response.data;
        console.log("[partnershipService] Transfert accepté avec succès");
        return transfer;
      } catch (error: any) {
        console.error("[partnershipService] Erreur lors de l'acceptation du transfert:", error);
        console.error("[partnershipService] Détails:", error?.response?.data);
        throw error;
      }
    },
    {
      maxRetries: 1,
      baseDelay: 500,
      maxDelay: 2000,
      shouldRetry: (error) => {
        const status = error?.response?.status;
        return !status || status >= 500;
      },
    }
  );
};

/**
 * Refuse un transfert de course proposé par une entreprise partenaire
 * @param transferId ID du transfert à refuser
 * @param reason (Optionnel) Raison du refus
 * @returns Les détails du transfert refusé
 */
export const rejectTransfer = async (transferId: string, reason?: string): Promise<Transfer> => {
  // 🔄 Retry avec backoff pour résilience réseau
  return retryWithBackoff(
    async () => {
      // ✅ Guard Pattern : Vérifier qu'on a un token valide avant d'envoyer
      if (!(await hasValidToken())) {
        throw new Error("Aucun token valide. Veuillez vous reconnecter.");
      }

      try {
        console.log("[partnershipService] Refus du transfert:", transferId, "Raison:", reason);

        const payload: any = {};
        if (reason) {
          payload.reason = reason;
        }

        const response = await enterpriseApi.post<{ data: Transfer }>(
          `${API_BASE}/transfers/${transferId}/reject`,
          payload
        );

        const transfer = response.data?.data || response.data;
        console.log("[partnershipService] Transfert refusé avec succès");
        return transfer;
      } catch (error: any) {
        console.error("[partnershipService] Erreur lors du refus du transfert:", error);
        console.error("[partnershipService] Détails:", error?.response?.data);
        throw error;
      }
    },
    {
      maxRetries: 1,
      baseDelay: 500,
      maxDelay: 2000,
      shouldRetry: (error) => {
        const status = error?.response?.status;
        return !status || status >= 500;
      },
    }
  );
};

/**
 * Récupère la liste des transferts (entrants et sortants)
 * @param filters Filtres optionnels (partnership_id, status)
 * @returns Liste des transferts
 */
export const fetchPartnershipTransfers = async (
  filters?: TransferListFilters
): Promise<Transfer[]> => {
  try {
    console.log("[partnershipService] Récupération des transferts avec filtres:", filters);

    const params = new URLSearchParams();
    if (filters?.partnership_id) {
      params.append("partnership_id", filters.partnership_id);
    }
    if (filters?.status) {
      params.append("status", filters.status);
    }

    const queryString = params.toString();
    const url = queryString ? `${API_BASE}/transfers?${queryString}` : `${API_BASE}/transfers`;

    const response = await enterpriseApi.get<{ data: Transfer[] }>(url);

    const transfers = response.data?.data || response.data;
    console.log(`[partnershipService] ${transfers.length} transferts récupérés`);
    return transfers;
  } catch (error: any) {
    console.error("[partnershipService] Erreur lors de la récupération des transferts:", error);
    console.error("[partnershipService] Détails:", error?.response?.data);
    throw error;
  }
};

/**
 * Récupère uniquement les transferts entrants (proposés par des partenaires)
 * Avec status PENDING pour action
 */
export const fetchIncomingTransfers = async (): Promise<Transfer[]> => {
  return fetchPartnershipTransfers({ status: "PENDING" });
};

/**
 * Récupère uniquement les transferts sortants (proposés par cette entreprise)
 */
export const fetchOutgoingTransfers = async (): Promise<Transfer[]> => {
  return fetchPartnershipTransfers();
};
