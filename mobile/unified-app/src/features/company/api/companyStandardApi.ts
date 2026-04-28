import { apiClient } from "../../../core/api/client";

type CompanyStandardRequestOptions = {
  contextId: string;
};

function withContextHeaders(options: CompanyStandardRequestOptions) {
  return {
    headers: {
      "X-Active-Context-Id": options.contextId,
    },
  };
}

export type CompanyStandardClientPayload = {
  first_name: string;
  last_name: string;
  gender: "male" | "female";
  phone?: string | null;
  email?: string | null;
};

export type CompanyBillingParty = {
  id: number;
  display_name: string;
  type: string;
};

export async function createStandardCompanyClient(
  options: CompanyStandardRequestOptions & {
    payload: CompanyStandardClientPayload;
  }
) {
  const response = await apiClient.post(
    "/companies/me/clients",
    options.payload,
    withContextHeaders(options)
  );
  return response.data as Record<string, unknown>;
}

export async function createStandardCompanyClientStay(
  options: CompanyStandardRequestOptions & {
    clientId: number;
    payload: { company_id: number; start_date: string; end_date?: string | null; notes?: string | null };
  }
) {
  const response = await apiClient.post(
    `/clients/${options.clientId}/stays`,
    options.payload,
    withContextHeaders(options)
  );
  return response.data as Record<string, unknown>;
}

export async function fetchStandardCompanyBillingParties(
  options: CompanyStandardRequestOptions & { active?: boolean }
) {
  const response = await apiClient.get("/company-settings/billing/parties", {
    ...withContextHeaders(options),
    params: { active: options.active ?? true },
  });
  const payload = response.data as { data?: Record<string, unknown>[] };
  const rows = Array.isArray(payload.data) ? payload.data : [];
  return rows
    .map((row) => {
      const id = Number(row.id);
      if (!Number.isFinite(id)) return null;
      return {
        id,
        display_name:
          typeof row.display_name === "string" && row.display_name.trim().length > 0
            ? row.display_name
            : `Billing #${id}`,
        type: typeof row.type === "string" && row.type.trim().length > 0 ? row.type : "unknown",
      } satisfies CompanyBillingParty;
    })
    .filter((row): row is CompanyBillingParty => row !== null);
}

export async function linkStandardCompanyClientBillingParty(
  options: CompanyStandardRequestOptions & {
    clientId: number;
    payload: {
      billing_party_id: number;
      role?: string | null;
      is_default?: boolean;
      contact_name?: string | null;
      contact_email?: string | null;
      contact_phone?: string | null;
    };
  }
) {
  const response = await apiClient.post(
    `/clients/${options.clientId}/billing-parties`,
    options.payload,
    withContextHeaders(options)
  );
  return response.data as Record<string, unknown>;
}
