import {
  buildCardMeta,
  resolveStatusDisplay,
  STATUS_TONES,
  statusToneToBadgeClass,
} from '../statusColors';
import { EXTERNAL_STATUSES } from '../../../../utils/requestStatus';

describe('statusColors', () => {
  const resolveBookingStatusKey = (summary) => String(summary?.status || '').toUpperCase();

  it('mappe Confirmée sur success', () => {
    const st = resolveStatusDisplay({ status: 'CONVERTED' }, resolveBookingStatusKey);
    expect(st.statusTone).toBe(STATUS_TONES.success);
    expect(st.badgeClass).toBe('badgeStatusSuccess');
  });

  it('mappe externe affecté sur warning', () => {
    const st = resolveStatusDisplay(
      { status: EXTERNAL_STATUSES.ASSIGNED, carrier_source: 'external' },
      resolveBookingStatusKey,
    );
    expect(st.statusTone).toBe(STATUS_TONES.warning);
  });

  it('compose metaCarrier et metaDetails sur 2 niveaux', () => {
    const meta = buildCardMeta({
      req: { status: 'CONVERTED', billing_intent: 'patient' },
      companyName: 'LIRIE Transport',
      carrierModeLabel: 'LIRIE',
      isExternal: false,
      tripTypeLabel: 'A/R',
      billingLabel: 'Facturé patient',
      timeTypeLabel: null,
    });
    expect(meta.carrierLine).toBe('LIRIE Transport');
    expect(meta.detailsLine).toBe('A/R · Facturé patient');
  });

  it('limite metaDetails à 2 fragments', () => {
    const meta = buildCardMeta({
      req: { status: 'CONVERTED' },
      companyName: 'LIRIE',
      carrierModeLabel: 'LIRIE',
      isExternal: false,
      tripTypeLabel: 'A/R',
      billingLabel: 'Facturé institution',
      timeTypeLabel: 'RDV',
    });
    expect(meta.detailsLine).toBe('A/R · Facturé institution');
  });

  it('statusToneToBadgeClass retourne neutral par défaut', () => {
    expect(statusToneToBadgeClass('unknown')).toBe('badgeStatusNeutral');
  });

  describe('micro STOP GATE PR1.5 — 3 cartes les plus chargées', () => {
    it('Annulée + Externe + Retard : statut error, transporteur niveau 2, pas de mur de texte', () => {
      const st = resolveStatusDisplay(
        { status: 'CANCELLED', carrier_source: 'external' },
        resolveBookingStatusKey,
      );
      const meta = buildCardMeta({
        req: { status: 'CANCELLED' },
        companyName: null,
        carrierModeLabel: 'Transporteur externe',
        isExternal: true,
        tripTypeLabel: null,
        billingLabel: null,
        timeTypeLabel: null,
      });

      expect(st.statusTone).toBe(STATUS_TONES.error);
      expect(st.badgeClass).toBe('badgeStatusError');
      expect(meta.carrierLine).toBe('Transporteur externe');
      expect(meta.detailsLine).toBeNull();
      expect(`${meta.carrierLine}${meta.detailsLine || ''}`).not.toMatch(/·/);
    });

    it('Confirmée + A/R + Facturation : 3 niveaux distincts (badges / transporteur / détails)', () => {
      const st = resolveStatusDisplay({ status: 'CONVERTED' }, resolveBookingStatusKey);
      const meta = buildCardMeta({
        req: { status: 'CONVERTED', billing_intent: 'clinic' },
        companyName: 'LIRIE Transport',
        carrierModeLabel: 'LIRIE',
        isExternal: false,
        tripTypeLabel: 'A/R',
        billingLabel: 'Facturé institution',
        timeTypeLabel: null,
      });

      expect(st.statusTone).toBe(STATUS_TONES.success);
      expect(meta.carrierLine).toBe('LIRIE Transport');
      expect(meta.detailsLine).toBe('A/R · Facturé institution');
      expect(meta.detailsLine.split(' · ').length).toBeLessThanOrEqual(2);
      expect(meta.carrierLine).not.toContain('A/R');
    });

    it('Externe affecté + Transporteur + Retour : warning, transporteur niveau 2, retour niveau 3', () => {
      const st = resolveStatusDisplay(
        { status: EXTERNAL_STATUSES.ASSIGNED, carrier_source: 'external' },
        resolveBookingStatusKey,
      );
      const meta = buildCardMeta({
        req: { status: EXTERNAL_STATUSES.ASSIGNED },
        companyName: 'Taxi Genève SA',
        carrierModeLabel: 'Transporteur externe',
        isExternal: true,
        tripTypeLabel: 'A/R',
        billingLabel: null,
        timeTypeLabel: null,
      });

      expect(st.statusTone).toBe(STATUS_TONES.warning);
      expect(meta.carrierLine).toBe('Transporteur : Taxi Genève SA');
      expect(meta.detailsLine).toBe('A/R');
    });
  });
});
