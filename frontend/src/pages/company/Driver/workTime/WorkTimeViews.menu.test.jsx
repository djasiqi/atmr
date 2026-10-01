import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { DriverSummary } from './WorkTimeViews';

const day = {
  date: '2026-09-03',
  entries: [
    {
      kind: 'transport',
      booking_id: 12,
      work_time_status: 'pending_validation',
      pickup_label: 'Rue de l’Ancien Lavoir',
      dropoff_label: 'Clinique les Hauts d’Anières',
      proposed_worked_minutes: 20,
    },
  ],
};

function renderDay() {
  return render(
    <DriverSummary
      name="Salomon"
      mode="real"
      days={[day]}
      finalized={false}
      loading={false}
      openDays={{ '2026-09-03': true }}
      onToggleDay={jest.fn()}
      onOpenBooking={jest.fn()}
      onExplain={jest.fn()}
      onAdjust={jest.fn()}
      onDuration={jest.fn()}
      onValidate={jest.fn()}
      validatingId={null}
      onCancelManual={jest.fn()}
    />
  );
}

test('le menu Rectifier / Pourquoi s’affiche hors du tableau', () => {
  const onDuration = jest.fn();
  const onExplain = jest.fn();
  render(
    <DriverSummary
      name="Salomon"
      mode="real"
      days={[day]}
      finalized={false}
      loading={false}
      openDays={{ '2026-09-03': true }}
      onToggleDay={jest.fn()}
      onOpenBooking={jest.fn()}
      onExplain={onExplain}
      onAdjust={jest.fn()}
      onDuration={onDuration}
      onValidate={jest.fn()}
      validatingId={null}
      onCancelManual={jest.fn()}
    />
  );

  fireEvent.click(screen.getByRole('button', { name: 'Autres actions' }));
  const menu = screen.getByRole('menu', { name: 'Autres actions' });
  expect(menu).toBeTruthy();
  expect(menu.parentElement).toBe(document.body);
  expect(screen.getByRole('menuitem', { name: 'Rectifier' })).toBeTruthy();
  expect(screen.getByRole('menuitem', { name: 'Pourquoi' })).toBeTruthy();

  fireEvent.click(screen.getByRole('menuitem', { name: 'Pourquoi' }));
  expect(onExplain).toHaveBeenCalledWith(12);
  expect(screen.queryByRole('menu')).toBeNull();
});

test('le bouton ··· reste visible même si le menu est fermé', () => {
  renderDay();
  expect(screen.getByRole('button', { name: 'Autres actions' })).toBeTruthy();
  expect(screen.queryByRole('menu')).toBeNull();
});
