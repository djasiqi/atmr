import { AxiosError } from "axios";

/**
 * Forme commune des erreurs API mobile (normalisée depuis Axios).
 *
 * Aligné sur le pattern privé déjà utilisé dans `features/driver/api.ts`
 * et `features/client/api.ts`. Centralisé ici pour permettre aux nouveaux
 * modules (ex. company push) d'importer une fonction partagée sans
 * dupliquer la logique.
 */
export type ApiError = {
  status: number | null;
  code: string;
  message: string;
  retryable?: boolean;
};

type ApiErrorPayload = {
  error?: string;
  code?: string;
  message?: string;
  error_code?: string;
  error_message?: string;
  retryable?: boolean;
};

const UNKNOWN_CODE = "UNKNOWN_ERROR";
const UNKNOWN_MESSAGE = "Erreur inconnue";

export function normalizeError(error: unknown): ApiError {
  const axiosError = error as AxiosError<ApiErrorPayload>;
  const payload: ApiErrorPayload = axiosError?.response?.data ?? {};

  const code =
    payload.error_code ??
    payload.code ??
    payload.error ??
    UNKNOWN_CODE;

  const message =
    payload.error_message ??
    payload.message ??
    payload.error ??
    axiosError?.message ??
    UNKNOWN_MESSAGE;

  const retryable =
    typeof payload.retryable === "boolean" ? payload.retryable : undefined;

  return {
    status: axiosError?.response?.status ?? null,
    code,
    message,
    retryable,
  };
}
