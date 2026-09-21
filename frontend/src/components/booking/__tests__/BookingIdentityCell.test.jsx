import React from 'react';
import { render, screen } from '@testing-library/react';
import BookingIdentityCell from '../BookingIdentityCell';

describe('BookingIdentityCell livraison', () => {
  it('met LIVRAISON et l’origine en signal principal', () => {
    render(
      <BookingIdentityCell
        booking={{
          mission_type: 'material_delivery',
          delivery_description: 'Livraison des effets personnels de M. Basset.',
          identity: {
            primary_label: 'Michel BASSET',
            secondary_label: "Clinique les Hauts d'Anières",
          },
        }}
      />
    );
    expect(screen.getByTestId('booking-identity-delivery-badge')).toHaveTextContent('LIVRAISON');
    expect(screen.getByText("Clinique les Hauts d'Anières")).toBeInTheDocument();
    expect(screen.getByText('Bénéficiaire · Michel BASSET')).toBeInTheDocument();
  });

  it('laisse un transport de personne inchangé', () => {
    render(
      <BookingIdentityCell
        booking={{
          mission_type: 'patient_transport',
          identity: {
            primary_label: 'Michel BASSET',
            secondary_label: "Clinique les Hauts d'Anières",
          },
        }}
      />
    );
    expect(screen.queryByTestId('booking-identity-delivery-badge')).not.toBeInTheDocument();
    expect(screen.getByText('Michel BASSET')).toBeInTheDocument();
    expect(screen.getByText("Clinique les Hauts d'Anières")).toBeInTheDocument();
  });

  it('n’affiche pas LIVRAISON si delivery_description est présente sans mission_type livraison', () => {
    render(
      <BookingIdentityCell
        booking={{
          mission_type: 'patient_transport',
          delivery_description: 'Livraison de documents',
          identity: {
            primary_label: 'Michel BASSET',
            secondary_label: "Clinique les Hauts d'Anières",
          },
        }}
      />
    );
    expect(screen.queryByTestId('booking-identity-delivery-badge')).not.toBeInTheDocument();
    expect(screen.queryByText('Bénéficiaire · Michel BASSET')).not.toBeInTheDocument();
    expect(screen.getByText('Michel BASSET')).toBeInTheDocument();
  });
});
