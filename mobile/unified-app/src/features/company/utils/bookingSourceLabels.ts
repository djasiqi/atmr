/** Libellés identity.source.type — alignés web `bookingSourceLabels.js`. */
export const BOOKING_SOURCE_LABELS: Record<string, { label: string }> = {
  institution: { label: "Institution" },
  partner_company: { label: "Partenaire" },
  company_client: { label: "Portefeuille" },
  company_account: { label: "Compte entreprise" },
  lirie_client: { label: "Plateforme LIRIE" },
  lirie_guest: { label: "Invité LIRIE" },
  legacy: { label: "Course" },
};

export function getBookingSourceMeta(sourceType: string | null | undefined): { label: string } {
  const key = String(sourceType || "legacy").toLowerCase();
  return BOOKING_SOURCE_LABELS[key] ?? BOOKING_SOURCE_LABELS.legacy;
}
