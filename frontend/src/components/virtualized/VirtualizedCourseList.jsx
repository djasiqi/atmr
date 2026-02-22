/**
 * VirtualizedCourseList.jsx
 *
 * Composant virtualisé pour CourseList utilisant react-window.
 *
 * Ce composant virtualise le rendu de la liste des courses pour améliorer
 * les performances avec de grandes listes.
 *
 * @module components/virtualized/VirtualizedCourseList
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import { List } from 'react-window';
import { fetchDriverAssignments } from '../../services/driverService';
import styles from './VirtualizedCourseList.module.css';

/**
 * Composant pour une ligne de course
 */
const CourseListItem = ({ index, style, data }) => {
  const { assignments, onRowClick } = data;
  const course = assignments[index];

  const handleClick = useCallback(() => {
    if (onRowClick && course) {
      onRowClick(course);
    }
  }, [course, onRowClick]);

  if (!course) return null;

  return (
    <div style={style}>
      <div onClick={handleClick} className={styles.courseItem}>
        <span>
          {course.pickup || '—'} → {course.dropoff || '—'}
        </span>
        <span>
          {course.scheduled_time
            ? new Date(course.scheduled_time).toLocaleTimeString('fr-FR', {
                hour: '2-digit',
                minute: '2-digit',
              })
            : '—'}
        </span>
      </div>
    </div>
  );
};

CourseListItem.propTypes = {
  index: PropTypes.number.isRequired,
  style: PropTypes.object.isRequired,
  data: PropTypes.shape({
    assignments: PropTypes.array.isRequired,
    onRowClick: PropTypes.func,
  }).isRequired,
};

/**
 * Composant VirtualizedCourseList
 *
 * Version virtualisée de CourseList qui n'affiche que les éléments visibles
 * dans le viewport.
 *
 * @param {Object} props - Props du composant
 * @param {Array} props.assignments - Liste des assignments/courses (optionnel, chargé automatiquement si vide)
 * @param {Function} props.onRowClick - Callback pour clic sur une course
 *
 * @returns {JSX.Element} Composant virtualisé
 */
const VirtualizedCourseList = ({ assignments: initialAssignments, onRowClick }) => {
  const [assignments, setAssignments] = useState(initialAssignments || []);
  const [loading, setLoading] = useState(!initialAssignments || initialAssignments.length === 0);
  const [error, setError] = useState(null);

  // Chargement automatique si initialAssignments vide
  useEffect(() => {
    if (!initialAssignments || initialAssignments.length === 0) {
      const loadAssignments = async () => {
        try {
          setLoading(true);
          const data = await fetchDriverAssignments();
          setAssignments(data);
        } catch (err) {
          console.error('Erreur lors du chargement des courses:', err);
          setError('Erreur lors du chargement des courses.');
        } finally {
          setLoading(false);
        }
      };
      loadAssignments();
    } else {
      setAssignments(initialAssignments);
      setLoading(false);
    }
  }, [initialAssignments]);

  // Mettre à jour assignments si initialAssignments change
  useEffect(() => {
    if (initialAssignments && initialAssignments.length > 0) {
      setAssignments(initialAssignments);
    }
  }, [initialAssignments]);

  // ✅ Données pour react-window - TOUJOURS un objet valide (jamais null/undefined/array)
  const itemData = useMemo(
    () => ({
      assignments: Array.isArray(assignments) ? assignments : [],
      onRowClick: typeof onRowClick === 'function' ? onRowClick : () => {},
    }),
    [assignments, onRowClick]
  );

  // Hauteur fixe par ligne (basé sur le CSS : padding + contenu)
  const ITEM_HEIGHT = 50;
  // Hauteur du conteneur (affiche ~8 lignes)
  const CONTAINER_HEIGHT = 400;

  if (loading) {
    return (
      <div className={styles.courseList}>
        <h2>Courses assignées</h2>
        <p>Chargement des courses...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.courseList}>
        <h2>Courses assignées</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className={styles.courseList}>
      <h2>Courses assignées</h2>
      {assignments.length === 0 ? (
        <p>Aucune course assignée.</p>
      ) : (
        <List
          height={Math.min(CONTAINER_HEIGHT, assignments.length * ITEM_HEIGHT)}
          itemCount={assignments.length}
          itemSize={ITEM_HEIGHT}
          width="100%"
          itemData={itemData}
          className={styles.virtualizedList}
        >
          {CourseListItem}
        </List>
      )}
    </div>
  );
};

VirtualizedCourseList.propTypes = {
  assignments: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      pickup: PropTypes.string,
      dropoff: PropTypes.string,
      scheduled_time: PropTypes.string,
    })
  ),
  onRowClick: PropTypes.func,
};

export default VirtualizedCourseList;
