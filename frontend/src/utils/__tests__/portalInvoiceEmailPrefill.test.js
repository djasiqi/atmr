import { resolveDefaultRecipientEmail } from '../portalInvoiceRecipientEmail';

describe('resolveDefaultRecipientEmail (PORTAL Direct patient)', () => {
  it('préremplit via default_recipient_email API', () => {
    expect(
      resolveDefaultRecipientEmail({
        default_recipient_email: 'osmani.mirjete@gmail.com',
        client: { contact_email: null },
      })
    ).toBe('osmani.mirjete@gmail.com');
  });

  it('utilise client.email (compte PORTAL) avant contact_email vide', () => {
    expect(
      resolveDefaultRecipientEmail({
        client: {
          email: 'portal@user.ch',
          contact_email: null,
        },
      })
    ).toBe('portal@user.ch');
  });

  it('fallback billing_party.contact_email', () => {
    expect(
      resolveDefaultRecipientEmail({
        billing_party: { contact_email: 'bp@patient.ch' },
        client: { contact_email: null },
      })
    ).toBe('bp@patient.ch');
  });

  it('retourne vide si aucune source', () => {
    expect(resolveDefaultRecipientEmail({ client: {} })).toBe('');
  });
});
