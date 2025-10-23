import React from 'react';
import './DispatchTableSkeleton.css';

/**
 * Skeleton loader pour le tableau de dispatch
 * Affiche un loading state pendant le chargement des données
 */
const DispatchTableSkeleton = ({ rows = 5 }) => {
  return (
    <div className="dispatch-table-skeleton">
      <div className="skeleton-header">
        <div className="skeleton-cell skeleton-cell-header"></div>
        <div className="skeleton-cell skeleton-cell-header"></div>
        <div className="skeleton-cell skeleton-cell-header"></div>
        <div className="skeleton-cell skeleton-cell-header"></div>
        <div className="skeleton-cell skeleton-cell-header"></div>
      </div>

      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="skeleton-row">
          <div className="skeleton-cell skeleton-cell-time"></div>
          <div className="skeleton-cell skeleton-cell-client"></div>
          <div className="skeleton-cell skeleton-cell-address"></div>
          <div className="skeleton-cell skeleton-cell-driver"></div>
          <div className="skeleton-cell skeleton-cell-status"></div>
        </div>
      ))}
    </div>
  );
};

export default DispatchTableSkeleton;
