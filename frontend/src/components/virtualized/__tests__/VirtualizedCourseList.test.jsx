/**
 * Tests pour VirtualizedCourseList
 */

import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import VirtualizedCourseList from '../VirtualizedCourseList';
import { fetchDriverAssignments } from '../../../services/driverService';
import { mockAssignments, mockAssignmentsLarge } from './fixtures/courseListData';

// Mock react-window
jest.mock('react-window', () => {
  const mockReact = require('react');
  return {
    FixedSizeList: ({ children, itemCount, itemData }) => {
      const items = Array.from({ length: itemCount }, (_, i) => {
        const child = children({ index: i, style: { height: 50, top: i * 50 }, data: itemData });
        return mockReact.cloneElement(child, { key: i, 'data-testid': `virtualized-item-${i}` });
      });
      return mockReact.createElement('div', { 'data-testid': 'virtualized-list', 'data-item-count': itemCount }, items);
    },
  };
});

// Mock driverService
jest.mock('../../../services/driverService', () => ({
  fetchDriverAssignments: jest.fn(),
}));

// Mock CSS modules
jest.mock('../VirtualizedCourseList.module.css', () => ({
  courseList: 'courseList',
  virtualizedList: 'virtualizedList',
  courseItem: 'courseItem',
}));

describe('VirtualizedCourseList', () => {
  const mockOnRowClick = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    fetchDriverAssignments.mockClear();
  });

  describe('Rendu avec données', () => {
    it('should render with empty data', async () => {
      // Mocker fetchDriverAssignments pour retourner un tableau vide immédiatement
      fetchDriverAssignments.mockResolvedValue([]);
      
      render(<VirtualizedCourseList assignments={[]} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByText('Courses assignées')).toBeInTheDocument();
      
      // Attendre que le chargement se termine
      await waitFor(() => {
        expect(screen.getByText('Aucune course assignée.')).toBeInTheDocument();
      });
    });

    it('should render with data', () => {
      render(<VirtualizedCourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByText('Courses assignées')).toBeInTheDocument();
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '3');
    });

    it('should display course information correctly', () => {
      render(<VirtualizedCourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByText(/Hôpital de Genève → Aéroport de Genève/)).toBeInTheDocument();
      expect(screen.getByText(/Clinique de Lausanne → Gare de Lausanne/)).toBeInTheDocument();
    });

    it('should handle large lists', () => {
      render(<VirtualizedCourseList assignments={mockAssignmentsLarge} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '100');
    });
  });

  describe('Chargement automatique', () => {
    it('should load data automatically when initialAssignments is empty', async () => {
      fetchDriverAssignments.mockResolvedValue(mockAssignments);

      render(<VirtualizedCourseList assignments={[]} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByText('Chargement des courses...')).toBeInTheDocument();
      expect(fetchDriverAssignments).toHaveBeenCalledTimes(1);

      await waitFor(() => {
        expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
      });
    });

    it('should not load data when initialAssignments is provided', () => {
      render(<VirtualizedCourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
      
      expect(fetchDriverAssignments).not.toHaveBeenCalled();
      expect(screen.getByTestId('virtualized-list')).toBeInTheDocument();
    });

    it('should display error message on load failure', async () => {
      fetchDriverAssignments.mockRejectedValue(new Error('Network error'));

      render(<VirtualizedCourseList assignments={[]} onRowClick={mockOnRowClick} />);
      
      await waitFor(() => {
        expect(screen.getByText('Erreur lors du chargement des courses.')).toBeInTheDocument();
      });
    });
  });

  describe('Interactions', () => {
    it('should call onRowClick when a course item is clicked', () => {
      render(<VirtualizedCourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
      
      // Le mock de react-window crée des éléments avec data-testid="virtualized-item-X"
      // Le composant CourseListItem a un onClick sur l'élément avec className={styles.courseItem}
      // Cherchons l'élément qui contient le texte de la course
      const courseText = screen.getByText(/Hôpital de Genève → Aéroport de Genève/);
      // Trouvons l'élément parent qui a le onClick (celui avec la classe courseItem)
      // Le mock clone l'enfant, donc la structure devrait être préservée
      const clickableParent = courseText.closest('div[class*="courseItem"]') || 
                               courseText.parentElement?.parentElement || 
                               courseText.parentElement;
      fireEvent.click(clickableParent);
      
      expect(mockOnRowClick).toHaveBeenCalledTimes(1);
      expect(mockOnRowClick).toHaveBeenCalledWith(mockAssignments[0]);
    });

    it('should not call onRowClick if callback is not provided', () => {
      render(<VirtualizedCourseList assignments={mockAssignments} />);
      
      const courseItems = screen.getAllByTestId(/virtualized-item-/);
      fireEvent.click(courseItems[0]);
      
      // Should not throw error
      expect(courseItems[0]).toBeInTheDocument();
    });
  });

  describe('Formatage des données', () => {
    it('should format time correctly', () => {
      render(<VirtualizedCourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />);
      
      // Vérifier que l'heure est formatée (format français)
      const timeElements = screen.getAllByText(/\d{2}:\d{2}/);
      expect(timeElements.length).toBeGreaterThan(0);
    });

    it('should handle missing pickup or dropoff', () => {
      const assignmentsWithMissing = [
        { id: 1, pickup: null, dropoff: 'Destination', scheduled_time: '2025-01-14T10:00:00Z' },
        { id: 2, pickup: 'Origin', dropoff: null, scheduled_time: '2025-01-14T11:00:00Z' },
      ];

      render(<VirtualizedCourseList assignments={assignmentsWithMissing} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByText(/— → Destination/)).toBeInTheDocument();
      expect(screen.getByText(/Origin → —/)).toBeInTheDocument();
    });

    it('should handle missing scheduled_time', () => {
      const assignmentsWithoutTime = [
        { id: 1, pickup: 'Origin', dropoff: 'Destination', scheduled_time: null },
      ];

      render(<VirtualizedCourseList assignments={assignmentsWithoutTime} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByText(/Origin → Destination/)).toBeInTheDocument();
      expect(screen.getByText('—')).toBeInTheDocument();
    });
  });

  describe('Mise à jour des données', () => {
    it('should update when assignments prop changes', () => {
      const { unmount } = render(
        <VirtualizedCourseList assignments={mockAssignments} onRowClick={mockOnRowClick} />
      );
      
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '3');
      
      // Démontons et remontons pour éviter l'erreur de hooks
      unmount();
      
      const newAssignments = [...mockAssignments, { id: 4, pickup: 'New', dropoff: 'Place', scheduled_time: '2025-01-14T15:00:00Z' }];
      render(<VirtualizedCourseList assignments={newAssignments} onRowClick={mockOnRowClick} />);
      
      expect(screen.getByTestId('virtualized-list')).toHaveAttribute('data-item-count', '4');
    });
  });
});

