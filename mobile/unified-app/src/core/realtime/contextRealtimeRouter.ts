type ContextRealtimeListener = (event: unknown) => void;
type ContextType = "client" | "driver" | "company" | "institution";

type DispatchOptions = {
  contextType?: ContextType | null;
};

class ContextRealtimeRouter {
  private listenersByContext = new Map<string, Set<ContextRealtimeListener>>();
  private activeContextType: ContextType | null = null;

  subscribe(contextId: string, listener: ContextRealtimeListener) {
    const listeners = this.listenersByContext.get(contextId) ?? new Set<ContextRealtimeListener>();
    listeners.add(listener);
    this.listenersByContext.set(contextId, listeners);
    return () => {
      const current = this.listenersByContext.get(contextId);
      if (!current) return;
      current.delete(listener);
      if (current.size === 0) {
        this.listenersByContext.delete(contextId);
      }
    };
  }

  setActiveContext(contextType: ContextType | null) {
    this.activeContextType = contextType;
  }

  private resolveEventContextType(
    event: unknown,
    options?: DispatchOptions
  ): ContextType | null {
    if (options?.contextType) {
      return options.contextType;
    }
    if (!event || typeof event !== "object") {
      return null;
    }
    const candidate = event as { context_type?: unknown };
    if (
      candidate.context_type === "client" ||
      candidate.context_type === "driver" ||
      candidate.context_type === "company" ||
      candidate.context_type === "institution"
    ) {
      return candidate.context_type;
    }
    return null;
  }

  dispatch(contextId: string, event: unknown, options?: DispatchOptions) {
    const eventContextType = this.resolveEventContextType(event, options);
    if (
      this.activeContextType &&
      eventContextType &&
      this.activeContextType !== eventContextType
    ) {
      return;
    }
    const listeners = this.listenersByContext.get(contextId);
    if (!listeners) return;
    listeners.forEach((listener) => {
      listener(event);
    });
  }
}

export const contextRealtimeRouter = new ContextRealtimeRouter();
