import { apiClient } from "../../../core/api/client";
import { normalizeError } from "../../../core/api/errors";

export type CompanyPushRegistrationPayload = {
  token: string;
  companyId: number;
  deviceId: string;
  platform?: "ios" | "android";
  provider?: "expo" | "fcm";
};

export async function registerCompanyPushToken(
  payload: CompanyPushRegistrationPayload
): Promise<void> {
  try {
    await apiClient.post("/companies/save-push-token", {
      token: payload.token,
      companyId: payload.companyId,
      device_id: payload.deviceId,
      platform: payload.platform,
      provider: payload.provider,
      client_auth_surface: "company",
    });
  } catch (error) {
    throw normalizeError(error);
  }
}

export type CompanyTestPushResult = {
  ok: boolean;
  results?: {
    token_preview?: string;
    platform?: string | null;
    ok?: boolean;
    error?: string;
  }[];
  tokens_count?: number;
  error?: string;
};

export async function sendCompanyTestPush(): Promise<CompanyTestPushResult> {
  try {
    const { data } = await apiClient.post<CompanyTestPushResult>("/companies/me/test-push");
    return data;
  } catch (error) {
    throw normalizeError(error);
  }
}
