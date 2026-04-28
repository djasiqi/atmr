import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import Modal from './Modal';

describe('Modal accessible', () => {
  it('expose un role dialog et ferme avec Escape', () => {
    const onClose = jest.fn();
    render(
      <Modal onClose={onClose}>
        <button type="button">Action</button>
      </Modal>
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
