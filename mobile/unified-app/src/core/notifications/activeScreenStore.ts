export type ActiveScreenRole = "driver" | "company" | null;

export type ActiveScreenState = {
  currentRoute: string | null;
  currentMissionId: number | null;
  currentThreadId: string | null;
  role: ActiveScreenRole;
};

const initialState: ActiveScreenState = {
  currentRoute: null,
  currentMissionId: null,
  currentThreadId: null,
  role: null,
};

let state: ActiveScreenState = { ...initialState };

export function getActiveScreenState(): ActiveScreenState {
  return state;
}

export function setActiveScreenState(patch: Partial<ActiveScreenState>): void {
  state = { ...state, ...patch };
}

export function clearActiveScreenState(role?: ActiveScreenRole): void {
  if (role && state.role !== role) return;
  state = { ...initialState };
}

export function setActiveMissionScreen(missionId: number, route?: string): void {
  setActiveScreenState({
    role: "driver",
    currentMissionId: missionId,
    currentThreadId: null,
    currentRoute: route ?? `/(app)/(driver)/missions/${missionId}`,
  });
}

export function setActiveThreadScreen(threadId: string, role: ActiveScreenRole, route?: string): void {
  setActiveScreenState({
    role,
    currentThreadId: threadId,
    currentMissionId: null,
    currentRoute: route ?? null,
  });
}

export function clearActiveMissionScreen(missionId: number): void {
  if (state.currentMissionId !== missionId) return;
  setActiveScreenState({
    currentMissionId: null,
    currentRoute: state.currentThreadId ? state.currentRoute : null,
  });
}

export function clearActiveThreadScreen(threadId: string): void {
  if (state.currentThreadId !== threadId) return;
  setActiveScreenState({
    currentThreadId: null,
    currentRoute: state.currentMissionId ? state.currentRoute : null,
  });
}
