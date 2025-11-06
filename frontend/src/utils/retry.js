/**
 * Utilitaire pour réessayer une fonction avec exponential backoff
 *
 * @param {Function} fn - La fonction à exécuter
 * @param {number} retries - Nombre de tentatives (défaut: 3)
 * @param {number} delay - Délai initial en ms (défaut: 1000)
 * @param {Function} onRetry - Callback appelé à chaque retry (optionnel)
 * @returns {Promise} - Résultat de la fonction ou erreur
 *
 * @example
 * const data = await retryWithBackoff(
 *   () => fetch('/api/v1/data'),
 *   3,
 *   1000,
 *   (attempt) => console.log(`Tentative ${attempt}...`)
 * );
 */
export async function retryWithBackoff(fn, retries = 3, delay = 1000, onRetry = null) {
  let lastError;

  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Si c'est la dernière tentative, on lance l'erreur
      if (i === retries - 1) {
        throw error;
      }

      // Calculer le délai avec exponential backoff
      const backoffDelay = delay * Math.pow(2, i);

      // Appeler le callback onRetry si fourni
      if (onRetry && typeof onRetry === 'function') {
        onRetry(i + 1, retries, backoffDelay);
      }

      if (process.env.NODE_ENV === 'development') {
        console.log(
          `[Retry] Tentative ${i + 1}/${retries} échouée, retry dans ${backoffDelay}ms...`
        );
      }

      // Attendre avant de réessayer
      await new Promise((resolve) => setTimeout(resolve, backoffDelay));
    }
  }

  throw lastError;
}

/**
 * Variante de retry pour les requêtes HTTP avec gestion des codes d'erreur
 *
 * @param {Function} fn - La fonction qui retourne une promesse de réponse HTTP
 * @param {Object} options - Options de configuration
 * @returns {Promise} - Résultat de la requête
 *
 * @example
 * const response = await retryHttpRequest(
 *   () => apiClient.get('/data'),
 *   {
 *     retries: 3,
 *     delay: 1000,
 *     retryableStatuses: [408, 429, 500, 502, 503, 504]
 *   }
 * );
 */
export async function retryHttpRequest(fn, options = {}) {
  const {
    retries = 3,
    delay = 1000,
    retryableStatuses = [408, 429, 500, 502, 503, 504],
    onRetry = null,
  } = options;

  let lastError;

  for (let i = 0; i < retries; i++) {
    try {
      const response = await fn();

      // Si la réponse est OK, on la retourne
      return response;
    } catch (error) {
      lastError = error;

      // Vérifier si l'erreur est "retryable"
      const isRetryable = error.response && retryableStatuses.includes(error.response.status);

      // Si c'est la dernière tentative OU l'erreur n'est pas retryable, on lance l'erreur
      if (i === retries - 1 || !isRetryable) {
        throw error;
      }

      // Calculer le délai avec exponential backoff
      const backoffDelay = delay * Math.pow(2, i);

      // Gestion spéciale pour 429 (Rate Limit) - utiliser Retry-After header si disponible
      if (error.response && error.response.status === 429) {
        const retryAfter = error.response.headers['retry-after'];
        if (retryAfter) {
          const retryAfterMs = parseInt(retryAfter) * 1000;
          if (!isNaN(retryAfterMs)) {
            await new Promise((resolve) => setTimeout(resolve, retryAfterMs));
            continue;
          }
        }
      }

      // Appeler le callback onRetry si fourni
      if (onRetry && typeof onRetry === 'function') {
        onRetry(i + 1, retries, backoffDelay, error);
      }

      if (process.env.NODE_ENV === 'development') {
        console.log(
          `[HTTP Retry] Tentative ${i + 1}/${retries} échouée (status: ${
            error.response?.status
          }), retry dans ${backoffDelay}ms...`
        );
      }

      // Attendre avant de réessayer
      await new Promise((resolve) => setTimeout(resolve, backoffDelay));
    }
  }

  throw lastError;
}

/**
 * Wrapper pour faciliter l'utilisation du retry avec les hooks React
 *
 * @param {Function} fn - La fonction à exécuter
 * @param {Object} options - Options de configuration
 * @returns {Promise} - Résultat de la fonction
 *
 * @example
 * const loadData = async () => {
 *   try {
 *     const data = await retryAsync(() => fetchData(), {
 *       retries: 3,
 *       onRetry: (attempt, total) => {
 *         toast.loading(`Tentative ${attempt}/${total}...`);
 *       }
 *     });
 *     setData(data);
 *   } catch (error) {
 *     setError('Échec du chargement après 3 tentatives');
 *   }
 * };
 */
export async function retryAsync(fn, options = {}) {
  return retryWithBackoff(fn, options.retries, options.delay, options.onRetry);
}

const retryUtils = {
  retryWithBackoff,
  retryHttpRequest,
  retryAsync,
};

export default retryUtils;
