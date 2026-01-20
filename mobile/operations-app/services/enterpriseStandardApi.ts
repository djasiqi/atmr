import axios, { AxiosHeaders } from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";

import { enterpriseApi, ENTERPRISE_SESSION_KEY } from "@/services/enterpriseAuth";
import { secureStorage, asyncStorage } from "@/services/storage";
import { AuthNotReadyError, isPublicEndpoint } from "@/services/authGuards";
import { waitForAuthReady } from "@/services/authSync";

// Client dédié pour les endpoints "standard API" appelés en mode enterprise
// (ex: /api/v1/companies/..., /api/v1/driver/company/...)
const baseURL = (enterpriseApi.defaults.baseURL || "").replace(
  "/api/v1/company_mobile",
  "/api/v1"
);

export const enterpriseStandardApi = axios.create({
  baseURL,
  timeout: 20000,
  headers: {
    "Content-Type": "application/json",
  },
});

enterpriseStandardApi.interceptors.request.use(async (config) => {
  const isPublic = isPublicEndpoint(config.url, "enterpriseStandard");

  if (!isPublic) {
    try {
      await waitForAuthReady(5000);
    } catch {
      throw new AuthNotReadyError({
        kind: "enterpriseStandard",
        reason: "auth_ready_timeout",
        url: config.url,
      });
    }
  }

  const headers =
    config.headers instanceof AxiosHeaders
      ? config.headers
      : new AxiosHeaders(config.headers || {});

  // Token enterprise requis pour standard API
  if (!headers.has("Authorization") && !isPublic) {
    const token = await secureStorage.getEnterpriseToken();
    if (!token) {
      throw new AuthNotReadyError({
        kind: "enterpriseStandard",
        reason: "missing_enterprise_token",
        url: config.url,
      });
    }
    headers.set("Authorization", `Bearer ${token}`);
  }

  // Ajouter X-Company-ID / X-Session-ID si disponibles
  if (!isPublic) {
    const sessionRaw = await AsyncStorage.getItem(ENTERPRISE_SESSION_KEY);
    if (sessionRaw) {
      try {
        const session = JSON.parse(sessionRaw);
        if (session?.company?.id && !headers.has("X-Company-ID")) {
          headers.set("X-Company-ID", String(session.company.id));
        }
        if (session?.sessionId && !headers.has("X-Session-ID")) {
          headers.set("X-Session-ID", String(session.sessionId));
        }
      } catch {
        // ignore
      }
    }
  }

  // Device correlation
  if (!headers.has("X-Device-ID") && Platform.OS !== "web") {
    try {
      const deviceId = await asyncStorage.getOrCreateDeviceId();
      headers.set("X-Device-ID", deviceId);
    } catch {
      // ignore
    }
  }

  config.headers = headers;
  return config;
});

