import React from 'react';
import './DispatchCardSkeleton.css';

/**
 * Skeleton loader pour les cartes de dispatch
 * Utilisé dans les vues en mode carte
 */
const DispatchCardSkeleton = ({ count = 3 }) => {
  return (
    <div className="dispatch-card-skeleton-container">
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="dispatch-card-skeleton">
          <div className="card-skeleton-header">
            <div className="skeleton-avatar"></div>
            <div className="skeleton-header-text">
              <div className="skeleton-title"></div>
              <div className="skeleton-subtitle"></div>
            </div>
          </div>

          <div className="card-skeleton-body">
            <div className="skeleton-line skeleton-line-full"></div>
            <div className="skeleton-line skeleton-line-medium"></div>
            <div className="skeleton-line skeleton-line-short"></div>
          </div>

          <div className="card-skeleton-footer">
            <div className="skeleton-badge"></div>
            <div className="skeleton-badge"></div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default DispatchCardSkeleton;
