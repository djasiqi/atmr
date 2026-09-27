/**
 * Bootstrap DEBUG P0 — deep links + fichier commande si EXPO_PUBLIC_P0_VALIDATE=1.
 * lirie://p0-probe?cmd=status|delete_refresh|securestore_diag
 * Fichier (fallback Dev Client) : <filesDir>/p0_cmd.txt contenant "delete_refresh"
 */
import * as Linking from "expo-linking";
import { useEffect } from "react";
import { AppState, Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { createAndPersistInstallationId } from "./authCredentialStore";
import {
  isP0ValidateEnabled,
  p0ForceDeleteStrictRefreshOnly,
  probeCredentialStatuses,
  probeEvent,
} from "./p0AuthStatusProbe";

const DIAG_KEY = "atmr.auth.p0_diag_probe";
const P0_CMD_FILENAME = "p0_cmd.txt";

async function readP0CommandFile(): Promise<string | null> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const FS = require("expo-file-system/legacy") as typeof import("expo-file-system/legacy");
    const base = FS.documentDirectory;
    if (!base) return null;
    const path = `${base}${P0_CMD_FILENAME}`;
    const info = await FS.getInfoAsync(path);
    if (!info.exists) return null;
    const raw = (await FS.readAsStringAsync(path)).trim();
    await FS.deleteAsync(path, { idempotent: true });
    return raw || null;
  } catch {
    return null;
  }
}

async function pollP0CommandFile(): Promise<void> {
  const cmd = await readP0CommandFile();
  if (!cmd) return;
  probeEvent("p0_cmd_file", { cmd });
  if (cmd === "delete_refresh") {
    await p0ForceDeleteStrictRefreshOnly();
  } else if (cmd === "securestore_diag") {
    await runSecureStoreDiag();
  } else if (cmd === "status") {
    await probeCredentialStatuses("p0_cmd_status");
  }
}

async function runSecureStoreDiag(): Promise<void> {
  probeEvent("securestore_diag_start", { platform: Platform.OS });
  const cryptoObj = globalThis.crypto as Crypto | undefined;
  probeEvent("crypto_capability", {
    has_crypto: Boolean(cryptoObj),
    has_randomUUID: Boolean(cryptoObj && typeof cryptoObj.randomUUID === "function"),
    has_getRandomValues: Boolean(
      cryptoObj && typeof cryptoObj.getRandomValues === "function"
    ),
  });
  try {
    await SecureStore.setItemAsync(DIAG_KEY, "ok", { requireAuthentication: false });
    const readBack = await SecureStore.getItemAsync(DIAG_KEY, {
      requireAuthentication: false,
    });
    probeEvent("securestore_diag_rw", {
      write: "ok",
      read: readBack === "ok" ? "FOUND" : "MISMATCH",
    });
    await SecureStore.deleteItemAsync(DIAG_KEY, { requireAuthentication: false });
  } catch (err) {
    const cause = err instanceof Error ? err.message : String(err);
    // Ne jamais logger de secret — message d'erreur OS uniquement.
    probeEvent("securestore_diag_rw", { write: "FAIL", cause: cause.slice(0, 160) });
  }

  // Harness DEBUG uniquement : si CSPRNG randomUUID absent (Hermes), seed un installation_id
  // opaque fixe pour débloquer LOGIN FRAIS — pas un correctif prod / pas de Math.random.
  let created = await createAndPersistInstallationId();
  if (created.status !== "found") {
    const seed = "atmr-p0validate-00000000-0000-4000-8000-000000000001";
    try {
      await SecureStore.setItemAsync("atmr.auth.installation_id", seed, {
        requireAuthentication: false,
      });
      const readBack = await SecureStore.getItemAsync("atmr.auth.installation_id", {
        requireAuthentication: false,
      });
      probeEvent("installation_seed", {
        seeded: readBack === seed ? "YES" : "NO",
        prior_status: created.status.toUpperCase(),
        prior_cause: "cause" in created ? String(created.cause).slice(0, 80) : null,
      });
      created = await createAndPersistInstallationId();
    } catch (err) {
      const cause = err instanceof Error ? err.message : String(err);
      probeEvent("installation_seed", { seeded: "FAIL", cause: cause.slice(0, 160) });
    }
  }
  probeEvent("installation_create", {
    status: created.status.toUpperCase(),
    cause: "cause" in created ? String(created.cause).slice(0, 160) : null,
  });
  await probeCredentialStatuses("securestore_diag");
}

async function handleP0Url(url: string | null | undefined): Promise<void> {
  if (!url || !isP0ValidateEnabled()) return;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return;
  }
  const isP0 =
    parsed.protocol === "lirie:" &&
    (parsed.hostname === "p0-probe" || parsed.pathname.includes("p0-probe"));
  if (!isP0) return;

  const cmd =
    parsed.searchParams.get("cmd") ||
    (parsed.pathname.includes("delete_refresh") ? "delete_refresh" : "status");

  probeEvent("deeplink", { cmd });
  if (cmd === "delete_refresh") {
    await p0ForceDeleteStrictRefreshOnly();
    return;
  }
  if (cmd === "securestore_diag") {
    await runSecureStoreDiag();
    return;
  }
  await probeCredentialStatuses("deeplink_status");
}

export function P0ValidateBootstrap(): null {
  useEffect(() => {
    if (!isP0ValidateEnabled()) return;
    probeEvent("p0_validate_bootstrap_mounted");
    void runSecureStoreDiag().catch(() => undefined);
    void Linking.getInitialURL()
      .then((url) => handleP0Url(url))
      .catch(() => undefined);
    const sub = Linking.addEventListener("url", ({ url }) => {
      void handleP0Url(url);
    });
    // Fallback harness : Dev Client ne livre pas toujours lirie://p0-probe.
    void pollP0CommandFile().catch(() => undefined);
    const interval = setInterval(() => {
      void pollP0CommandFile().catch(() => undefined);
    }, 1500);
    const appSub = AppState.addEventListener("change", (state) => {
      if (state === "active") {
        void pollP0CommandFile().catch(() => undefined);
      }
    });
    return () => {
      sub.remove();
      clearInterval(interval);
      appSub.remove();
    };
  }, []);
  return null;
}
