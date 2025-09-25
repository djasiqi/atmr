// src/components/CourseList.jsx
import React, { useState, useEffect } from "react";
import styles from "./CourseList.module.css";
import { fetchDriverAssignments } from "../../../../services/driverService";

const CourseList = ({ assignments: initialAssignments, onRowClick }) => {
  const [assignments, setAssignments] = useState(initialAssignments || []);
  const [loading, setLoading] = useState(
    !initialAssignments || initialAssignments.length === 0
  );
  const [error, setError] = useState(null);

  useEffect(() => {
    // Si aucune donnée n'est passée via les props, on charge les courses via l'API.
    if (!initialAssignments || initialAssignments.length === 0) {
      const loadAssignments = async () => {
        try {
          setLoading(true);
          const data = await fetchDriverAssignments();
          setAssignments(data);
        } catch (err) {
          console.error("Erreur lors du chargement des courses:", err);
          setError("Erreur lors du chargement des courses.");
        } finally {
          setLoading(false);
        }
      };
      loadAssignments();
    }
  }, [initialAssignments]);

  if (loading) {
    return <p>Chargement des courses...</p>;
  }

  if (error) {
    return <p>{error}</p>;
  }

  return (
    <div className={styles.courseList}>
      <h2>Courses assignées</h2>
      {assignments.length === 0 ? (
        <p>Aucune course assignée.</p>
      ) : (
        <ul>
          {assignments.map((course) => (
            <li
              key={course.id}
              onClick={() => onRowClick(course)}
              className={styles.courseItem}
            >
              <span>
                {course.pickup} → {course.dropoff}
              </span>
              <span>
                {new Date(course.scheduled_time).toLocaleTimeString("fr-FR")}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default CourseList;
