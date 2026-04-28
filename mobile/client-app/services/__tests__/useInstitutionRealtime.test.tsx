/* eslint-disable import/first */
import React from 'react';
import renderer, { act } from 'react-test-renderer';

jest.mock('@tanstack/react-query', () => ({
  useQuery: jest.fn(),
  useQueryClient: jest.fn(),
}));

jest.mock('@/services/institutionRealtimeBridge', () => ({
  joinInstitutionRealtime: jest.fn(),
  subscribeInstitutionEvents: jest.fn(() => jest.fn()),
  disconnectInstitutionRealtime: jest.fn(),
}));

import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  disconnectInstitutionRealtime,
  joinInstitutionRealtime,
  subscribeInstitutionEvents,
} from '@/services/institutionRealtimeBridge';
import { useInstitutionRealtime } from '../useInstitutionRealtime';

const mockedUseQuery = useQuery as jest.Mock;
const mockedUseQueryClient = useQueryClient as jest.Mock;

describe('useInstitutionRealtime', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedUseQueryClient.mockReturnValue({
      invalidateQueries: jest.fn().mockResolvedValue(undefined),
    });
  });

  function mount(enabled: boolean, institutionId?: number) {
    mockedUseQuery.mockReturnValue({
      data: institutionId ? { id: institutionId } : undefined,
    });

    function Probe() {
      useInstitutionRealtime(enabled);
      return null;
    }

    let tree: renderer.ReactTestRenderer;
    act(() => {
      tree = renderer.create(<Probe />);
    });
    return tree!;
  }

  it('does not connect when disabled', () => {
    mount(false, 10);
    expect(joinInstitutionRealtime).not.toHaveBeenCalled();
    expect(subscribeInstitutionEvents).not.toHaveBeenCalled();
  });

  it('connects and subscribes when enabled', () => {
    const tree = mount(true, 22);
    expect(joinInstitutionRealtime).toHaveBeenCalledWith(22);
    expect(subscribeInstitutionEvents).toHaveBeenCalledTimes(1);
    act(() => tree.unmount());
    expect(disconnectInstitutionRealtime).toHaveBeenCalledTimes(1);
  });
});
