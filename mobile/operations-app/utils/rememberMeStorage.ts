/**
 * Stockage "Se souvenir de moi" pour le login chauffeur.
 * App concernée : operations-app (Espace Chauffeur = driver app).
 * Utilise expo-secure-store (Keychain/Keystore) uniquement.
 * Ne jamais logger le mot de passe.
 */

import * as SecureStore from "expo-secure-store";

/** Lancée lorsque SecureStore échoue (ex. setItemAsync). Permet d’afficher un toast côté UI. */
export class RememberMeStorageError extends Error {
  constructor() {
    super("Impossible d'enregistrer sur cet appareil.");
    this.name = "RememberMeStorageError";
  }
}

export type RememberMeMode = "driver" | "enterprise";

const PREFIX = {
  driver: "driver",
  enterprise: "enterprise",
} as const;

export type RememberedCredentials = { email: string; password: string };

function keys(mode: RememberMeMode) {
  const p = PREFIX[mode];
  return {
    rememberMe: `${p}.rememberMe`,
    email: `${p}.rememberedEmail`,
    password: `${p}.rememberedPassword`,
  };
}

async function getItem(key: string): Promise<string | null> {
  try {
    return await SecureStore.getItemAsync(key);
  } catch {
    return null;
  }
}

async function setItem(key: string, value: string): Promise<void> {
  await SecureStore.setItemAsync(key, value);
}

async function deleteItem(key: string): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(key);
  } catch {
    // ignore
  }
}

/** Retourne true si "Se souvenir de moi" est activé. */
export async function getRememberMe(mode: RememberMeMode = "driver"): Promise<boolean> {
  const v = await getItem(keys(mode).rememberMe);
  return v === "true";
}

/** Persiste la préférence "Se souvenir de moi". */
export async function setRememberMe(
  value: boolean,
  mode: RememberMeMode = "driver"
): Promise<void> {
  const k = keys(mode);
  if (value) {
    await setItem(k.rememberMe, "true");
  } else {
    await deleteItem(k.rememberMe);
    await clearRememberedCredentials(mode);
  }
}

/** Retourne email + mot de passe mémorisés, ou null si incomplet / erreur. */
export async function getRememberedCredentials(
  mode: RememberMeMode = "driver"
): Promise<RememberedCredentials | null> {
  try {
    const k = keys(mode);
    const [email, password] = await Promise.all([
      getItem(k.email),
      getItem(k.password),
    ]);
    if (email?.trim() && password != null && password.length > 0) {
      return { email: email.trim(), password };
    }
    return null;
  } catch {
    return null;
  }
}

/** Enregistre email + mot de passe dans SecureStore. Ne pas logger le password. */
export async function setRememberedCredentials(
  email: string,
  password: string,
  mode: RememberMeMode = "driver"
): Promise<void> {
  const k = keys(mode);
  await Promise.all([
    setItem(k.email, email.trim()),
    setItem(k.password, password),
  ]);
}

/** Supprime email et mot de passe mémorisés (rememberMe reste inchangé). */
export async function clearRememberedCredentials(
  mode: RememberMeMode = "driver"
): Promise<void> {
  const k = keys(mode);
  await Promise.all([deleteItem(k.email), deleteItem(k.password)]);
}
