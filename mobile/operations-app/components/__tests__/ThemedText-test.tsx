import * as React from 'react';
import renderer from 'react-test-renderer';
import { act } from 'react-test-renderer';

import { ThemedText } from '../ThemedText';

// Mock useColorScheme pour retourner 'light' par défaut dans les tests
jest.mock('@/hooks/useColorScheme', () => ({
  useColorScheme: jest.fn(() => 'light'),
}));

it(`renders correctly`, () => {
  let tree: renderer.ReactTestRenderer | undefined;

  act(() => {
    tree = renderer.create(<ThemedText>Snapshot test!</ThemedText>);
  });

  const json = tree!.toJSON();
  expect(json).toMatchSnapshot();
});
