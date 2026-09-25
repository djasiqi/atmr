/**
 * @jest-environment jsdom
 */
import {
  isPortalDoubleValidationUiReady,
  isPortalOfferConfirmable,
  isPortalDoubleValidationFlow,
  PORTAL_DV_COPY,
  firstClickToastMessage,
  portalReservationAmountDisplay,
  portalCarrierFacingAmountDisplay,
  portalCarrierAcceptOfferedAmount,
  portalCarrierAcceptButtonLabel,
  formatPortalCancellationPolicyForDisplay,
  formatPortalOfferChf,
} from '../../utils/portalDoubleValidationUi';

describe('portalDoubleValidationUi', () => {
  it('distingue estimation et plafond', () => {
    const state = isPortalDoubleValidationUiReady({
      enabled: true,
      estimate: 85,
      maximum: 95,
    });
    expect(state.ok).toBe(true);
    expect(state.estimateEqualsMaximum).toBe(false);
  });

  it('refuse un plafond manquant ou sous l’estimation', () => {
    expect(
      isPortalDoubleValidationUiReady({ enabled: true, estimate: 85, maximum: null }).ok
    ).toBe(false);
    expect(
      isPortalDoubleValidationUiReady({ enabled: true, estimate: 85, maximum: 80 }).ok
    ).toBe(false);
  });

  it('ignore la règle si le flag est OFF', () => {
    expect(
      isPortalDoubleValidationUiReady({ enabled: false, estimate: 85, maximum: null }).ok
    ).toBe(true);
  });

  it('happy path — offre confirmable avec transporteur, prix, plafond, policy', () => {
    const offer = {
      id: 1,
      company_name: 'Trans SA',
      offered_amount: 82,
      maximum_accepted_amount: 95,
      cancellation_policy_text: 'Annulation >24h gratuite',
      offer_content_hash: 'abc',
    };
    expect(isPortalOfferConfirmable(offer)).toBe(true);
  });

  it('above maximum — jamais confirmable', () => {
    expect(
      isPortalOfferConfirmable({
        id: 2,
        offered_amount: 120,
        maximum_accepted_amount: 95,
        offer_content_hash: 'abc',
      })
    ).toBe(false);
  });

  it('stale / sans hash — non confirmable', () => {
    expect(
      isPortalOfferConfirmable({
        id: 3,
        offered_amount: 80,
        maximum_accepted_amount: 95,
      })
    ).toBe(false);
  });

  it('wording contractuel 1er / 2e clic', () => {
    expect(firstClickToastMessage({ doubleValidationEnabled: true })).toBe(
      PORTAL_DV_COPY.firstClick
    );
    expect(PORTAL_DV_COPY.firstClick.toLowerCase()).not.toContain('transport est confirmé');
    expect(PORTAL_DV_COPY.carrierOfferedTitle.toLowerCase()).toContain('proposition');
    expect(PORTAL_DV_COPY.transportConfirmed.toLowerCase()).toContain('confirmé');
  });

  it('détecte le flux double_validation_v2', () => {
    expect(
      isPortalDoubleValidationFlow({ portal_contract_flow: 'double_validation_v2' })
    ).toBe(true);
    expect(isPortalDoubleValidationFlow({ portal_contract_flow: 'legacy' })).toBe(false);
  });

  it('carte réservation DV : uniquement le plafond (pas l’estimation à côté)', () => {
    const d = portalReservationAmountDisplay({
      portal_contract_flow: 'double_validation_v2',
      amount: 50,
      estimated_amount_snapshot: 50,
      maximum_accepted_amount: 52,
    });
    expect(d.isCeiling).toBe(true);
    expect(d.label).toBe('Prix maximum de la demande');
    expect(d.amount).toBe(52);
    expect(d.ceiling).toBe(52);
    expect(d.estimate).toBe(50);
    expect(d.lines).toHaveLength(1);
    expect(d.lines[0].label).toMatch(/Prix maximum accepté/);
    expect(d.lines[0].amount).toBe(52);
    expect(d.lines[0].primary).toBe(true);
  });

  it('carte réservation DV confirmée : montant contractuel', () => {
    const d = portalReservationAmountDisplay({
      portal_contract_flow: 'double_validation_v2',
      amount: 50,
      maximum_accepted_amount: 52,
      contractual_amount: 40,
    });
    expect(d.isCeiling).toBe(false);
    expect(d.label).toBe('Montant confirmé');
    expect(d.amount).toBe(40);
    expect(d.lines).toEqual([
      expect.objectContaining({
        key: 'contractual',
        label: 'Montant confirmé',
        amount: 40,
        primary: true,
      }),
    ]);
  });

  it('entreprise : affiche le tarif grille, jamais l’estimation client 50', () => {
    const open = portalCarrierFacingAmountDisplay({
      portal_contract_flow: 'double_validation_v2',
      amount: 50,
      company_id: null,
      company_suggested_amount: 40,
    });
    expect(open.mode).toBe('company_quote');
    expect(open.amount).toBe(40);
    expect(open.label).toBe('Votre tarif (grille)');
    expect(
      portalCarrierAcceptOfferedAmount({
        portal_contract_flow: 'double_validation_v2',
        company_suggested_amount: 45,
        amount: 50,
      })
    ).toBe(45);
    expect(
      portalCarrierAcceptButtonLabel({
        portal_contract_flow: 'double_validation_v2',
        company_suggested_amount: 45,
        amount: 50,
      })
    ).toBe('Accepter cette course à CHF 45.00');
  });

  it('entreprise conditional_order_v1 : tarif grille, jamais estimation 50', () => {
    const open = portalCarrierFacingAmountDisplay({
      portal_contract_flow: 'conditional_order_v1',
      amount: 50,
      company_id: null,
      carrier_quote: 40,
    });
    expect(open.mode).toBe('company_quote');
    expect(open.amount).toBe(40);
    expect(open.label).toBe('Votre tarif (grille)');
    expect(
      portalCarrierAcceptButtonLabel({
        portal_contract_flow: 'conditional_order_v1',
        company_suggested_amount: 40,
        amount: 50,
      })
    ).toBe('Accepter cette course à CHF 40.00');
  });

  it('formate le prix CHF et retire la réf. technique des conditions', () => {
    expect(formatPortalOfferChf(40)).toBe('CHF 40.00');
    expect(formatPortalOfferChf('52.5')).toBe('CHF 52.50');
    expect(
      formatPortalCancellationPolicyForDisplay(
        "Conditions d'annulation\n\nAucun frais.\n\n[Réf. config abc123]"
      )
    ).toBe("Conditions d'annulation\n\nAucun frais.");
  });
});
