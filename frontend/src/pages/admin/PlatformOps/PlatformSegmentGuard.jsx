import React from 'react';
import { usePlatformCapabilities } from '../../../hooks/usePlatformCapabilities';
import PlatformAccessDenied from './PlatformAccessDenied';
import styles from './AdminPlatformOps.module.css';

export default function PlatformSegmentGuard({ segment, children }) {
  const { canAccess, isLoading } = usePlatformCapabilities();

  if (isLoading) {
    return <div className={styles.loading}>Chargement…</div>;
  }

  if (!canAccess(segment)) {
    return <PlatformAccessDenied />;
  }

  return children;
}
