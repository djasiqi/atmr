export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'Une erreur inconnue est survenue';
}

export function logError(context: string, error: unknown): void {
  const message = getErrorMessage(error);
  console.error(`❌ ${context}:`, message);
}