import React from 'react';
import {
  getAdminRelativePath,
  resolveActiveWorkspace,
} from '../navigation/adminNavRegistry';
import styles from './AdminTopbar.module.css';

/**
 * Barre supérieure admin — titre du workspace actif uniquement (pas de stubs).
 * @param {{ publicId: string, pathname: string }} props
 */
export default function AdminTopbar({ publicId, pathname }) {
  const relative = getAdminRelativePath(pathname, publicId);
  const workspace = resolveActiveWorkspace(relative);

  return (
    <header className={styles.topbar} role="banner">
      <h1 className={styles.title}>{workspace?.label || 'Administration'}</h1>
      <div className={styles.meta} aria-hidden="true" />
    </header>
  );
}
