import React, { useMemo } from 'react';
import { NavLink, useLocation, useParams } from 'react-router-dom';
import { usePlatformCapabilities } from '../../../hooks/usePlatformCapabilities';
import { useAdminCapabilities } from '../../../hooks/useAdminCapabilities';
import {
  getAdminRelativePath,
  resolveActiveWorkspace,
} from '../navigation/adminNavRegistry';
import { adminBasePath } from '../routing/adminRoutePaths';
import styles from './AdminWorkspaceNav.module.css';

/**
 * Sous-navigation du workspace actif — alimentée par le registre.
 * - platformCapability → usePlatformCapabilities
 * - adminCapability → useAdminCapabilities (PR2bis, labs)
 */
export default function AdminWorkspaceNav() {
  const { public_id: publicId } = useParams();
  const location = useLocation();
  const { canAccess, isLoading } = usePlatformCapabilities();
  const { can: canAdmin, isLoading: adminCapsLoading } = useAdminCapabilities();

  const relative = getAdminRelativePath(location.pathname, publicId);
  const workspace = resolveActiveWorkspace(relative);
  const base = adminBasePath(publicId);

  const children = useMemo(() => {
    const list = workspace?.children || [];
    if (!list.length) return [];
    return list.filter((child) => {
      if (child.platformCapability) {
        if (isLoading) return true;
        if (!canAccess(child.platformCapability)) return false;
      }
      if (child.adminCapability) {
        if (adminCapsLoading) return true;
        if (!canAdmin(child.adminCapability)) return false;
      }
      return true;
    });
  }, [workspace, canAccess, isLoading, canAdmin, adminCapsLoading]);

  if (!children.length) {
    return null;
  }

  const needsPlatformFilter = (workspace?.children || []).some((c) => c.platformCapability);

  return (
    <nav className={styles.workspaceNav} aria-label={`Sections ${workspace.label}`}>
      {needsPlatformFilter && isLoading ? (
        <span className={styles.loading} role="status">
          Chargement des accès…
        </span>
      ) : null}
      {children.map((child) => {
        const href = child.path ? `${base}/${child.path}` : base;
        return (
          <NavLink
            key={child.id}
            to={href}
            end={child.end ?? false}
            className={({ isActive }) =>
              `${styles.tab} ${isActive ? styles.tabActive : ''}`
            }
          >
            {child.label}
          </NavLink>
        );
      })}
    </nav>
  );
}
