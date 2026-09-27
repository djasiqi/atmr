export type AssignDriverPickerOption = { id: number; label: string };

export type DriverAvatarTone = { background: string; foreground: string };

const AVATAR_TONES: DriverAvatarTone[] = [
  { background: "#CCFBF1", foreground: "#0F766E" },
  { background: "#E0E7FF", foreground: "#3730A3" },
  { background: "#FCE7F3", foreground: "#9D174D" },
  { background: "#FEF3C7", foreground: "#92400E" },
  { background: "#DBEAFE", foreground: "#1D4ED8" },
  { background: "#DCFCE7", foreground: "#166534" },
];

function nameParts(label: string): string[] {
  return label
    .trim()
    .split(/[\s_]+/)
    .filter((part) => part.length > 0);
}

function initialsFromParts(parts: string[], extraLastLetters: number): string {
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2 + extraLastLetters).toLocaleUpperCase("fr");
  const first = parts[0][0] ?? "";
  const last = parts[parts.length - 1] ?? "";
  return `${first}${last.slice(0, 1 + extraLastLetters)}`.toLocaleUpperCase("fr");
}

export function driverInitials(label: string): string {
  return initialsFromParts(nameParts(label), 0);
}

/** Évite « LM » pour Léa Martin et Lucas Morel en allongeant seulement les noms en conflit. */
export function uniqueDriverInitials(drivers: AssignDriverPickerOption[]): Map<number, string> {
  const assigned = new Map<number, string>();
  const groups = new Map<string, AssignDriverPickerOption[]>();
  for (const driver of drivers) {
    const base = driverInitials(driver.label);
    const group = groups.get(base) ?? [];
    group.push(driver);
    groups.set(base, group);
  }

  for (const group of groups.values()) {
    if (group.length === 1) {
      assigned.set(group[0].id, driverInitials(group[0].label));
      continue;
    }
    let resolved = false;
    for (let extra = 1; extra < 8 && !resolved; extra += 1) {
      const local = new Map<number, string>();
      const seen = new Map<string, number>();
      for (const driver of group) {
        const initials = initialsFromParts(nameParts(driver.label), extra);
        local.set(driver.id, initials);
        seen.set(initials, (seen.get(initials) ?? 0) + 1);
      }
      if ([...seen.values()].every((count) => count === 1)) {
        for (const [id, initials] of local) assigned.set(id, initials);
        resolved = true;
      }
    }
    if (!resolved) {
      for (const driver of group) assigned.set(driver.id, driverInitials(driver.label));
    }
  }
  return assigned;
}

export function driverAvatarTone(label: string): DriverAvatarTone {
  let hash = 0;
  for (const char of label) {
    hash = (hash * 31 + (char.codePointAt(0) ?? 0)) >>> 0;
  }
  return AVATAR_TONES[hash % AVATAR_TONES.length] ?? AVATAR_TONES[0];
}

export function filterDriverOptions<T extends { label: string }>(drivers: T[], query: string): T[] {
  const needle = query.trim().toLocaleLowerCase("fr");
  if (!needle) return drivers;
  return drivers.filter((driver) => driver.label.toLocaleLowerCase("fr").includes(needle));
}

export function driverPickerSummary(total: number, visible: number, query: string): string {
  if (query.trim().length > 0) {
    return visible === 1 ? "1 résultat" : `${visible} résultats`;
  }
  return total === 1 ? "1 chauffeur disponible" : `${total} chauffeurs disponibles`;
}
