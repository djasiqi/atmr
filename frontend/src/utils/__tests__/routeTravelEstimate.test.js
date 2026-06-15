jest.mock('../apiClient', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
  },
}));

import apiClient from '../apiClient';
import {
  resolveOutboundRoute,
  resolveOutboundRouteEndpoints,
  formatOutboundRouteLabel,
  estimateTravelMinutesHaversine,
  haversineKm,
  fetchRouteTravelMinutes,
  toCoord,
} from '../routeTravelEstimate';

describe('routeTravelEstimate', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('rejette les coordonnées nulles (0)', () => {
    expect(toCoord(0)).toBeNull();
    expect(toCoord('0')).toBeNull();
    expect(toCoord(46.2)).toBe(46.2);
  });

  it('utilise la première étape pour les missions multi-legs', () => {
    const req = {
      pickup_location: 'Institution',
      dropoff_location: 'Retour institution',
      pickup_lat: 46.1,
      pickup_lng: 6.1,
      dropoff_lat: 46.2,
      dropoff_lng: 6.2,
      legs: [
        {
          sequence_index: 0,
          pickup_location: 'Anières',
          pickup_lat: 46.276,
          pickup_lng: 6.223,
          dropoff_location: 'HUG Genève',
          dropoff_lat: 46.192,
          dropoff_lng: 6.149,
        },
        {
          sequence_index: 1,
          dropoff_location: 'Retour Anières',
          dropoff_lat: 46.276,
          dropoff_lng: 6.223,
        },
      ],
    };

    expect(resolveOutboundRoute(req)).toMatchObject({
      pickup_lat: 46.276,
      pickup_lng: 6.223,
      dropoff_lat: 46.192,
      dropoff_lng: 6.149,
      pickup_address: 'Anières',
      dropoff_address: 'HUG Genève',
    });
    expect(resolveOutboundRouteEndpoints(req)).toEqual({
      pickup_lat: 46.276,
      pickup_lng: 6.223,
      dropoff_lat: 46.192,
      dropoff_lng: 6.149,
    });
    expect(formatOutboundRouteLabel(req)).toBe('Anières → HUG Genève');
  });

  it('estime un trajet Anières → HUG au-delà du plancher 15 min (repli haversine)', () => {
    const km = haversineKm(46.276, 6.223, 46.192, 6.149);
    expect(km).toBeGreaterThan(5);

    const minutes = estimateTravelMinutesHaversine({
      pickup_lat: 46.276,
      pickup_lng: 6.223,
      dropoff_lat: 46.192,
      dropoff_lng: 6.149,
    });
    expect(minutes).toBeGreaterThanOrEqual(18);
    expect(minutes).toBeLessThanOrEqual(35);
  });

  it('priorise l\'estimation serveur pour une offre institution', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/company/request-offers/42/travel-estimate') {
        return Promise.resolve({ data: { travel_minutes: 31, source: 'google_directions' } });
      }
      return Promise.reject(new Error('unexpected get'));
    });

    const minutes = await fetchRouteTravelMinutes({ id: 1002 }, 42);
    expect(minutes).toBe(31);
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it('géocode les adresses institution sans GPS puis interroge Google Directions', async () => {
    apiClient.get.mockImplementation((url, config) => {
      if (url === '/geocode/aliases') {
        return Promise.resolve({ data: [] });
      }
      if (url === '/geocode/geocode') {
        const address = String(config?.params?.address || '');
        if (address.includes('Anières')) {
          return Promise.resolve({ data: { lat: 46.276, lon: 6.223 } });
        }
        if (address.includes('HUG')) {
          return Promise.resolve({ data: { lat: 46.192, lon: 6.149 } });
        }
      }
      return Promise.reject(new Error('unexpected get'));
    });

    apiClient.post.mockImplementation((url) => {
      if (url === '/directions') {
        return Promise.resolve({
          data: {
            status: 'OK',
            duration_seconds: 28 * 60,
            duration_in_traffic_seconds: 31 * 60,
          },
        });
      }
      return Promise.reject(new Error('unexpected post'));
    });

    const req = {
      pickup_location: 'Chemin des Courbes 9, Anières',
      dropoff_location: 'HUG Genève',
      scheduled_time: '2026-06-16T09:30:00',
      legs: [
        {
          sequence_index: 0,
          pickup_location: 'Chemin des Courbes 9, Anières',
          dropoff_location: 'HUG Genève',
        },
      ],
    };

    const minutes = await fetchRouteTravelMinutes(req);
    expect(minutes).toBe(31);
    expect(apiClient.post).toHaveBeenCalledWith(
      '/directions',
      expect.objectContaining({
        origin: { lat: 46.276, lng: 6.223 },
        destination: { lat: 46.192, lng: 6.149 },
        departure_time: expect.any(Number),
      }),
      expect.any(Object),
    );
  });
});
