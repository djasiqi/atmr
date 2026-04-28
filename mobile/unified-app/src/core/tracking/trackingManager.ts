import { AppState, AppStateStatus } from "react-native";

export type TrackingMode = "mission_live" | "availability_presence" | "observability_only";
export type TrackingTickResult = "success" | "failed" | "skipped";

type TrackingTickContext = {
  mode: TrackingMode;
  appState: AppStateStatus;
};

type TrackingFailureContext = TrackingTickContext & {
  consecutiveFailures: number;
  backoffMs: number;
};

type TrackingRecoveredContext = TrackingTickContext & {
  previousFailures: number;
};

type TrackingManagerOptions = {
  foregroundIntervalMs: number;
  backgroundIntervalMs: number;
  maxBackoffMs: number;
  onTick: (context: TrackingTickContext) => Promise<TrackingTickResult>;
  onFailure?: (context: TrackingFailureContext) => void;
  onRecovered?: (context: TrackingRecoveredContext) => void;
};

type TrackingManagerSnapshot = {
  mode: TrackingMode | null;
  appState: AppStateStatus;
  isRunning: boolean;
  consecutiveFailures: number;
  backoffUntilMs: number;
  lastAttemptAt: string | null;
};

export class TrackingManager {
  private mode: TrackingMode | null = null;
  private appState: AppStateStatus = AppState.currentState;
  private timer: ReturnType<typeof setInterval> | null = null;
  private foregroundIntervalMs: number;
  private backgroundIntervalMs: number;
  private consecutiveFailures = 0;
  private backoffUntilMs = 0;
  private lastAttemptAt: string | null = null;

  private readonly appStateSubscription = AppState.addEventListener("change", (next) => {
    this.appState = next;
    if (this.timer) {
      this.restartLoop();
    }
  });

  constructor(private readonly options: TrackingManagerOptions) {
    this.foregroundIntervalMs = options.foregroundIntervalMs;
    this.backgroundIntervalMs = options.backgroundIntervalMs;
  }

  private intervalMs() {
    return this.appState === "active" ? this.foregroundIntervalMs : this.backgroundIntervalMs;
  }

  private backoffDelayMs(retryCount: number) {
    return Math.min(2 ** retryCount * 1000, this.options.maxBackoffMs);
  }

  private restartLoop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (!this.mode) return;
    this.timer = setInterval(() => {
      void this.tick();
    }, this.intervalMs());
  }

  async tick() {
    if (!this.mode) return;
    const now = Date.now();
    if (now < this.backoffUntilMs) return;

    this.lastAttemptAt = new Date(now).toISOString();
    const result = await this.options.onTick({ mode: this.mode, appState: this.appState });
    if (result === "failed") {
      this.consecutiveFailures += 1;
      const backoffMs = this.backoffDelayMs(this.consecutiveFailures);
      this.backoffUntilMs = Date.now() + backoffMs;
      this.options.onFailure?.({
        mode: this.mode,
        appState: this.appState,
        consecutiveFailures: this.consecutiveFailures,
        backoffMs,
      });
      return;
    }
    if (result === "success") {
      if (this.consecutiveFailures > 0) {
        this.options.onRecovered?.({
          mode: this.mode,
          appState: this.appState,
          previousFailures: this.consecutiveFailures,
        });
      }
      this.consecutiveFailures = 0;
      this.backoffUntilMs = 0;
    }
  }

  start(mode: TrackingMode) {
    this.mode = mode;
    this.restartLoop();
    void this.tick();
  }

  updateMode(mode: TrackingMode) {
    this.mode = mode;
    if (this.timer) {
      this.restartLoop();
    }
  }

  setIntervals(intervals: { foregroundIntervalMs: number; backgroundIntervalMs: number }) {
    const nextForeground = Math.max(1000, Math.floor(intervals.foregroundIntervalMs));
    const nextBackground = Math.max(1000, Math.floor(intervals.backgroundIntervalMs));
    const changed =
      this.foregroundIntervalMs !== nextForeground || this.backgroundIntervalMs !== nextBackground;
    this.foregroundIntervalMs = nextForeground;
    this.backgroundIntervalMs = nextBackground;
    if (changed && this.timer) {
      this.restartLoop();
    }
  }

  stop() {
    this.mode = null;
    this.consecutiveFailures = 0;
    this.backoffUntilMs = 0;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  getSnapshot(): TrackingManagerSnapshot {
    return {
      mode: this.mode,
      appState: this.appState,
      isRunning: Boolean(this.timer),
      consecutiveFailures: this.consecutiveFailures,
      backoffUntilMs: this.backoffUntilMs,
      lastAttemptAt: this.lastAttemptAt,
    };
  }

  dispose() {
    this.stop();
    this.appStateSubscription.remove();
  }
}
