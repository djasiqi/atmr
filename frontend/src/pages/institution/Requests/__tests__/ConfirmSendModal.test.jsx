import React from 'react';
import { render, screen } from '@testing-library/react';
import ConfirmSendModal from '../ConfirmSendModal';

describe('ConfirmSendModal récap livraison', () => {
  it('affiche type et description avant envoi', () => {
    render(
      <ConfirmSendModal
        onClose={() => {}}
        onConfirm={() => {}}
        missionTypeLabel="Livraison"
        deliveryDescription="Livraison des effets personnels de M. Basset."
      />
    );
    expect(screen.getByTestId('confirm-send-mission-recap')).toBeInTheDocument();
    expect(screen.getByText('Type de mission')).toBeInTheDocument();
    expect(screen.getByText('Livraison')).toBeInTheDocument();
    expect(screen.getByText('Description')).toBeInTheDocument();
    expect(screen.getByText('Livraison des effets personnels de M. Basset.')).toBeInTheDocument();
  });

  it('n’affiche pas le récap livraison pour un transport de personne sans props', () => {
    render(<ConfirmSendModal onClose={() => {}} onConfirm={() => {}} />);
    expect(screen.queryByTestId('confirm-send-mission-recap')).not.toBeInTheDocument();
  });
});
