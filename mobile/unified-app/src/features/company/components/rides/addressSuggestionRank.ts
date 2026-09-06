import type { RideAddressOption } from "../../useRideForms";

export const ADDRESS_IGNORE_QUERIES = new Set(["non spécifié", "non specifie", "n/a", "na"]);

export function splitAddressLabel(label: string) {
  const parts = label
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length <= 1) {
    return { primary: label.trim(), secondary: "" };
  }
  const [primary, ...rest] = parts;
  return { primary, secondary: rest.join(", ") };
}

export function isAliasSuggestion(item: RideAddressOption): boolean {
  return item.source === "alias";
}

export function isGoogleLikeSuggestion(item: RideAddressOption): boolean {
  return item.source === "google_places" || item.source === "google";
}

export function looksLikePoi(item: RideAddressOption): boolean {
  return (
    isGoogleLikeSuggestion(item) &&
    Array.isArray(item.types) &&
    item.types.some((t) => t !== "geocode" && t !== "route" && t !== "street_address")
  );
}

export function rankAddressSuggestion(item: RideAddressOption, normalizedQuery: string): number {
  if (isAliasSuggestion(item)) return 500;
  const label = (item.label || "").toLowerCase();
  const mainText = (item.mainText || "").toLowerCase();
  const startsWithQuery = label.startsWith(normalizedQuery) || mainText.startsWith(normalizedQuery);
  if (startsWithQuery && looksLikePoi(item)) return 400;
  if (startsWithQuery && isGoogleLikeSuggestion(item)) return 350;
  if (looksLikePoi(item)) return 300;
  if (isGoogleLikeSuggestion(item)) return 250;
  if (startsWithQuery) return 200;
  return 100;
}

export function sortAddressSuggestions(
  rows: RideAddressOption[],
  normalizedQuery: string
): RideAddressOption[] {
  return [...rows].sort(
    (a, b) => rankAddressSuggestion(b, normalizedQuery) - rankAddressSuggestion(a, normalizedQuery)
  );
}
