export type PerfRole = "driver" | "company" | "client" | "institution" | "unknown";

export type PerfActiveContext = {
  role: PerfRole;
  screen: string;
};

let active: PerfActiveContext = {
  role: "unknown",
  screen: "app.root",
};

export function getPerfActiveContext(): PerfActiveContext {
  return active;
}

export function setPerfActiveContext(partial: Partial<PerfActiveContext>): void {
  active = { ...active, ...partial };
}

export function setPerfScreen(screen: string): void {
  active = { ...active, screen };
}

export function setPerfRole(role: PerfRole): void {
  active = { ...active, role };
}
