/**
 * Préremplissage e-mail destinataire pour SendEmailModal (Direct patient / PORTAL).
 * Miroir frontend de `resolve_invoice_recipient_email` (API expose aussi
 * `default_recipient_email`).
 */

export function resolveDefaultRecipientEmail(inv) {
  if (!inv) return '';
  if (inv.default_recipient_email) return String(inv.default_recipient_email).trim();
  if (inv.meta?.last_recipient_email) return String(inv.meta.last_recipient_email).trim();
  if (inv.meta?.recipient_email) return String(inv.meta.recipient_email).trim();
  if (inv.billing_party?.contact_email) return inv.billing_party.contact_email;
  if (inv.bill_to_client?.contact_email) return inv.bill_to_client.contact_email;
  if (inv.billed_to_company?.billing_email) return inv.billed_to_company.billing_email;
  if (inv.billed_to_company?.contact_email) return inv.billed_to_company.contact_email;
  if (inv.client?.email) return inv.client.email;
  if (inv.client?.contact_email) return inv.client.contact_email;
  return '';
}
