import {
  resolveDriverClusteringEnabled,
  resolveDriverClusteringThreshold,
} from '../driverMapClustering';

describe('driverMapClustering', () => {
  it('force true ignore le seuil', () => {
    expect(
      resolveDriverClusteringEnabled(10, { REACT_APP_ENABLE_DRIVER_CLUSTERING: 'true' })
    ).toBe(true);
  });

  it('force false même au-delà du seuil', () => {
    expect(
      resolveDriverClusteringEnabled(100, { REACT_APP_ENABLE_DRIVER_CLUSTERING: 'false' })
    ).toBe(false);
  });

  it('auto active au-delà du seuil par défaut (50)', () => {
    expect(resolveDriverClusteringEnabled(51, {})).toBe(true);
    expect(resolveDriverClusteringEnabled(50, {})).toBe(false);
  });

  it('seuil personnalisable via REACT_APP_DRIVER_CLUSTERING_THRESHOLD', () => {
    const env = { REACT_APP_DRIVER_CLUSTERING_THRESHOLD: '30' };
    expect(resolveDriverClusteringThreshold(env)).toBe(30);
    expect(resolveDriverClusteringEnabled(31, env)).toBe(true);
    expect(resolveDriverClusteringEnabled(30, env)).toBe(false);
  });
});
