import { trackSeoEvent, trackPublicSeoPageView } from './seoAnalytics';

describe('seoAnalytics', () => {
  const original = window.__LIRIE_SEO_ANALYTICS__;

  afterEach(() => {
    window.__LIRIE_SEO_ANALYTICS__ = original;
  });

  it('délègue au collector global sans données sensibles', () => {
    const calls = [];
    window.__LIRIE_SEO_ANALYTICS__ = (event, props) => {
      calls.push({ event, props });
    };

    trackSeoEvent('seo_contact_click', {
      landing_page: '/contact',
      patient_id: 'secret',
      email: 'patient@example.com',
      source: 'footer',
    });

    expect(calls).toHaveLength(1);
    expect(calls[0].event).toBe('seo_contact_click');
    expect(calls[0].props.source).toBe('footer');
    expect(calls[0].props.patient_id).toBeUndefined();
    expect(calls[0].props.email).toBeUndefined();
  });

  it('ignore les événements hors liste', () => {
    const calls = [];
    window.__LIRIE_SEO_ANALYTICS__ = (event, props) => {
      calls.push({ event, props });
    };
    trackSeoEvent('seo_unknown_event', {});
    expect(calls).toHaveLength(0);
  });

  it('trackPublicSeoPageView émet seo_public_page_view', () => {
    const calls = [];
    window.__LIRIE_SEO_ANALYTICS__ = (event) => {
      calls.push(event);
    };
    trackPublicSeoPageView('/professionnel');
    expect(calls).toEqual(['seo_public_page_view']);
  });
});
