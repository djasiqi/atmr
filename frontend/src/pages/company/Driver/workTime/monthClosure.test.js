import React from 'react';
import { render, screen } from '@testing-library/react';
import { monthClosureState } from './monthClosure';
import { PeriodChrome } from './WorkTimeViews';

const october = { from: '2026-10-01', to: '2026-10-31' };
const september = { from: '2026-09-01', to: '2026-09-30' };

describe('monthClosureState', () => {
  it('refuse octobre le 30 et le 31, et l’autorise le 1er novembre', () => {
    const on30 = monthClosureState('month', october, new Date('2026-10-30T10:00:00Z'));
    const on31 = monthClosureState('month', october, new Date('2026-10-31T22:00:00Z'));
    const on1 = monthClosureState('month', october, new Date('2026-10-31T23:30:00Z'));
    expect(on30).toMatchObject({ visible: true, closable: false, availableOn: '2026-11-01' });
    expect(on31).toMatchObject({ visible: true, closable: false });
    expect(on1).toMatchObject({ visible: true, closable: true, monthName: 'octobre' });
  });

  it('autorise septembre dès le 1er octobre et novembre dès le 1er décembre', () => {
    expect(
      monthClosureState('month', september, new Date('2026-09-30T22:30:00Z')).closable,
    ).toBe(true);
    expect(
      monthClosureState('month', { from: '2026-11-01', to: '2026-11-30' }, new Date('2026-11-30T23:30:00Z'))
        .closable,
    ).toBe(true);
  });

  it('cache la clôture hors de la vue mensuelle, même pour un mois civil complet', () => {
    const now = new Date('2026-10-01T08:00:00Z');
    expect(monthClosureState('today', { from: '2026-09-30', to: '2026-09-30' }, now).visible).toBe(
      false,
    );
    expect(monthClosureState('week', { from: '2026-09-28', to: '2026-10-04' }, now).visible).toBe(
      false,
    );
    expect(monthClosureState('custom', september, now).visible).toBe(false);
    expect(
      monthClosureState('month', { from: '2026-09-15', to: '2026-09-30' }, now).visible,
    ).toBe(false);
  });
});

function chrome(closure, finalized = false) {
  return render(
    <PeriodChrome
      presets={[{ id: 'month', label: 'Ce mois' }]}
      preset="month"
      heading="Octobre 2026"
      periodLabel="01.10.2026 → 31.10.2026"
      displayMode="flat"
      onPreset={() => {}}
      onShift={() => {}}
      onMode={() => {}}
      showClosure
      closure={closure}
      finalized={finalized}
      onClosePeriod={() => {}}
      onReopenPeriod={() => {}}
      onAdd={() => {}}
      ready={false}
      showSummary={false}
    />,
  );
}

test('le mois en cours annonce la date, un mois terminé se clôture, un mois figé se réouvre', () => {
  const pending = monthClosureState('month', october, new Date('2026-10-30T12:00:00Z'));
  chrome(pending);
  expect(
    screen.getByText((_, element) =>
      element?.tagName === 'P' &&
      /Clôture disponible à partir du 1 novembre/.test(element.textContent || ''),
    ),
  ).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Clôturer le mois' })).toBeNull();

  chrome(monthClosureState('month', september, new Date('2026-10-01T08:00:00Z')));
  expect(screen.getByRole('button', { name: 'Clôturer le mois' })).toBeTruthy();

  chrome(
    { visible: true, closable: false, availableOn: '2026-10-01', monthName: 'septembre' },
    true,
  );
  expect(screen.getByText('Clôturé')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Réouvrir septembre' })).toBeTruthy();
});
