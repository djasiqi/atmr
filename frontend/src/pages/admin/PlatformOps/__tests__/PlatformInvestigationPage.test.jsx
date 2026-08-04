import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';

jest.mock('../../../../services/adminService', () => ({
  postPlatformSearch: jest.fn(() => Promise.resolve({ ok: true })),
  fetchPlatformAuditReplay: jest.fn(),
}));

const PlatformInvestigationPage = require('../PlatformInvestigationPage').default;
const { postPlatformSearch } = require('../../../../services/adminService');

describe('PlatformInvestigationPage deep-link', () => {
  beforeEach(() => {
    postPlatformSearch.mockClear();
  });

  it('préremplit et lance la recherche avec booking_id', async () => {
    render(
      <MemoryRouter
        initialEntries={['/dashboard/admin/pub-1/advanced/platform/investigation?booking_id=3']}
      >
        <Routes>
          <Route
            path="/dashboard/admin/:public_id/advanced/platform/investigation"
            element={<PlatformInvestigationPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByDisplayValue('3')).toBeInTheDocument();
    expect(postPlatformSearch).toHaveBeenCalledWith({ query: '3' });
  });
});
