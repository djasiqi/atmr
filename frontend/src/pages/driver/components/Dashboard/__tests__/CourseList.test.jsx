/**
 * Tests d'intégration pour CourseList (wrapper)
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import CourseList from '../CourseList';
import VirtualizedCourseList from 'components/virtualized/VirtualizedCourseList';
import { mockAssignments } from 'components/virtualized/__tests__/fixtures/courseListData';

// Mock VirtualizedCourseList pour vérifier que les props sont passées
// Utiliser le chemin relatif depuis le fichier de test
jest.mock('../../../../../components/virtualized/VirtualizedCourseList', () => {
  const MockComponent = jest.fn(({ assignments, onRowClick }) => (
    <div data-testid="virtualized-course-list">
      <div data-testid="assignments-count">{assignments?.length || 0}</div>
      <div data-testid="has-on-row-click">{onRowClick ? 'yes' : 'no'}</div>
    </div>
  ));
  return {
    __esModule: true,
    default: MockComponent,
  };
});

describe('CourseList (Wrapper)', () => {
  const mockOnRowClick = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render VirtualizedCourseList', () => {
    render(<CourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
    
    expect(screen.getByTestId('virtualized-course-list')).toBeInTheDocument();
  });

  it('should pass assignments prop correctly', () => {
    render(<CourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
    
    expect(screen.getByTestId('assignments-count')).toHaveTextContent('3');
    expect(VirtualizedCourseList).toHaveBeenCalledWith(
      expect.objectContaining({
        assignments: mockAssignments,
      }),
      expect.anything()
    );
  });

  it('should pass onRowClick prop correctly', () => {
    render(<CourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
    
    expect(screen.getByTestId('has-on-row-click')).toHaveTextContent('yes');
    expect(VirtualizedCourseList).toHaveBeenCalledWith(
      expect.objectContaining({
        onRowClick: mockOnRowClick,
      }),
      expect.anything()
    );
  });

  it('should handle empty assignments', () => {
    render(<CourseList assignments={[]} onRowClick={mockOnRowClick} />);
    
    expect(screen.getByTestId('assignments-count')).toHaveTextContent('0');
  });

  it('should handle missing onRowClick', () => {
    render(<CourseList assignments={mockAssignments} />);
    
    expect(screen.getByTestId('has-on-row-click')).toHaveTextContent('no');
  });
});

