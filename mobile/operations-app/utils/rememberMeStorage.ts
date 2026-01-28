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

const KEY_REMEMBER_ME = "driver.rememberMe";
const KEY_EMAIL = "driver.rememberedEmail";
const KEY_PASSWORD = "driver.rememberedPassword";

export type RememberedCredentials = { email: string; password: string };

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
export async function getRememberMe(): Promise<boolean> {
  const v = await getItem(KEY_REMEMBER_ME);
  return v === "true";
}

/** Persiste la préférence "Se souvenir de moi". */
export async function setRememberMe(value: boolean): Promise<void> {
  if (value) {
    await setItem(KEY_REMEMBER_ME, "true");
  } else {
    await deleteItem(KEY_REMEMBER_ME);
    await clearRememberedCredentials();
  }
}

/** Retourne email + mot de passe mémorisés, ou null si incomplet / erreur. */
export async function getRememberedCredentials(): Promise<RememberedCredentials | null> {
  try {
    const [email, password] = await Promise.all([
      getItem(KEY_EMAIL),
      getItem(KEY_PASSWORD),
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
  password: string
): Promise<void> {
  await Promise.all([
    setItem(KEY_EMAIL, email.trim()),
    setItem(KEY_PASSWORD, password),
  ]);
}

/** Supprime email et mot de passe mémorisés (rememberMe reste inchangé). */
export async function clearRememberedCredentials(): Promise<void> {
  await Promise.all([
    deleteItem(KEY_EMAIL),
    deleteItem(KEY_PASSWORD),
  ]);
}
