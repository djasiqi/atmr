import AsyncStorage from "@react-native-async-storage/async-storage";

export type OfflineMutationAction = {
  id: string;
  queuedAt: number;
  retryCount: number;
};

export type OfflineMutationFlushResult = {
  sent: number;
  dropped: number;
  failed: number;
};

type OfflineMutationQueueOptions<TAction extends OfflineMutationAction> = {
  storageKey: string;
  maxRetries: number;
  replayWindowMs: number;
  maxReplayAttemptsPerAction?: number;
  backoffBaseMs: number;
  backoffMaxMs: number;
  execute: (action: TAction) => Promise<void>;
  /**
   * Classification des erreurs d'exécution : une erreur permanente (ex. 4xx
   * métier `retryable: false`) ne sera JAMAIS résolue par un retry — l'action
   * est retirée immédiatement et les suivantes continuent.
   */
  isPermanentError?: (error: unknown) => boolean;
  /** L'action doit-elle être rejouée dans le contexte actif ? (multi-session) */
  shouldReplay?: (action: TAction) => boolean;
  onExpired?: (action: TAction) => void;
  onRetry?: (action: TAction, retryCount: number) => void;
  onFailure?: (action: TAction, retryCount: number) => void;
  onPermanentFailure?: (action: TAction, error: unknown) => void;
  onSkipped?: (action: TAction) => void;
  onSuccess?: (action: TAction) => void;
};

export class OfflineMutationQueue<TAction extends OfflineMutationAction> {
  private loaded = false;
  private actions: TAction[] = [];
  private isFlushing = false;

  constructor(private readonly options: OfflineMutationQueueOptions<TAction>) {}

  private async load() {
    if (this.loaded) return;
    const raw = await AsyncStorage.getItem(this.options.storageKey);
    this.actions = raw ? (JSON.parse(raw) as TAction[]) : [];
    this.loaded = true;
  }

  private async save() {
    await AsyncStorage.setItem(this.options.storageKey, JSON.stringify(this.actions));
  }

  private isExpired(action: TAction) {
    return Date.now() - action.queuedAt > this.options.replayWindowMs;
  }

  private backoffDelayMs(retryCount: number) {
    return Math.min(
      this.options.backoffBaseMs * 2 ** retryCount,
      this.options.backoffMaxMs
    );
  }

  async enqueue(action: TAction) {
    await this.load();
    this.actions.push(action);
    await this.save();
    return action;
  }

  async removeWhere(predicate: (action: TAction) => boolean) {
    await this.load();
    this.actions = this.actions.filter((action) => !predicate(action));
    await this.save();
  }

  async removeById(actionId: string) {
    await this.removeWhere((action) => action.id === actionId);
  }

  async count() {
    await this.load();
    return this.actions.length;
  }

  async oldestAgeMs() {
    await this.load();
    if (this.actions.length === 0) return 0;
    const oldestQueuedAt = this.actions.reduce((min, action) => Math.min(min, action.queuedAt), Date.now());
    return Math.max(0, Date.now() - oldestQueuedAt);
  }

  async flush(): Promise<OfflineMutationFlushResult> {
    await this.load();
    if (this.isFlushing) return { sent: 0, dropped: 0, failed: 0 };

    this.isFlushing = true;
    try {
      this.actions.sort((a, b) => a.queuedAt - b.queuedAt);
      let sent = 0;
      let dropped = 0;
      let failed = 0;
      const nextActions: TAction[] = [];

      for (let index = 0; index < this.actions.length; index += 1) {
        const action = this.actions[index];
        if (this.isExpired(action)) {
          dropped += 1;
          this.options.onExpired?.(action);
          continue;
        }
        if (this.options.shouldReplay && !this.options.shouldReplay(action)) {
          // Contexte différent (autre chauffeur / autre session) :
          // conservée mais jamais rejouée hors de son contexte.
          this.options.onSkipped?.(action);
          nextActions.push(action);
          continue;
        }

        try {
          await this.options.execute(action);
          sent += 1;
          this.options.onSuccess?.(action);
        } catch (error) {
          if (this.options.isPermanentError?.(error)) {
            // Erreur définitive (ex. transition périmée 409) : retirer cette
            // action et CONTINUER — les suivantes restent éligibles.
            dropped += 1;
            this.options.onPermanentFailure?.(action, error);
            continue;
          }

          const retryCount = action.retryCount + 1;
          this.options.onRetry?.(action, retryCount);

          if (
            retryCount >= this.options.maxRetries ||
            retryCount >= (this.options.maxReplayAttemptsPerAction ?? Number.POSITIVE_INFINITY)
          ) {
            failed += 1;
            this.options.onFailure?.(action, retryCount);
            nextActions.push(...this.actions.slice(index + 1));
            continue;
          }

          failed += 1;
          await new Promise((resolve) =>
            setTimeout(resolve, this.backoffDelayMs(retryCount))
          );
          nextActions.push({
            ...action,
            retryCount,
          });
          nextActions.push(...this.actions.slice(index + 1));
          break;
        }
      }

      this.actions = nextActions;
      await this.save();
      return { sent, dropped, failed };
    } finally {
      this.isFlushing = false;
    }
  }
}
