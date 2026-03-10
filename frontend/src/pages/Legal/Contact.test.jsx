import { fireEvent, render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

import Contact from './Contact';

const renderPage = () =>
  render(
    <BrowserRouter>
      <Contact />
    </BrowserRouter>
  );

describe('Contact page', () => {
  it("affiche le hero et la liste des categories", () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /contact/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /selectionnez la nature de votre demande/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /01 support technique/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /02 institution \/ integration/i })).toBeInTheDocument();
  });

  it('rend chaque ligne de categorie cliquable vers la bonne route', () => {
    renderPage();
    const item = screen.getByRole('link', { name: /01 support technique/i });
    fireEvent.mouseEnter(item);

    expect(screen.getByText(/assistance liee a l'utilisation de la plateforme/i)).toBeInTheDocument();
    expect(item).toHaveAttribute('href', '/contact/support');
  });
});
