// src/components/layout/Header/CompanyHeader.jsx
/**
 * Header professionnel — Espace Entreprise
 *
 * Aligné sur le design InstitutionLayout header :
 * - Zone gauche : logo entreprise + nom
 * - Zone droite : indicateurs (retards, socket), aujourd'hui
 */

import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams, useLocation } from 'react-router-dom';
import { FiAlertTriangle, FiCalendar } from 'react-icons/fi';
import styles from './CompanyHeader.module.css';

import useCompanyData from '../../../hooks/useCompanyData';
import useCompanyAuthToken from '../../../hooks/useCompanyAuthToken';
import useDispatchDelays from '../../../hooks/useDispatchDelays';
import resolveLogoUrl from '../../../utils/resolveLogoUrl';
import SocketStatusBadge from '../../common/SocketStatusBadge';
import CompanyNotificationBell from './CompanyNotificationBell';

function getInitials(name = '') {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || '').join('') || 'CO';
}

function formatToday() {
  return new Date().toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  });
}

const CompanyHeader = () => {
  const params = useParams();
  const location = useLocation();

  const routePublicId =
    params.public_id ||
    (() => {
      const match = location.pathname.match(/\/dashboard\/company\/([^/]+)/);
      return match ? match[1] : null;
    })();

  const companyData = useCompanyData() || {};
  const company = companyData.company || null;

  const [logoError, setLogoError] = useState(false);
  const name = company?.name || 'Entreprise';

  const logoSrc = useMemo(() => {
    const abs = resolveLogoUrl(company?.logo_url);
    if (!abs) return '';
    if (abs && !abs.startsWith('http') && !abs.startsWith('data:') && !abs.startsWith('blob:')) {
      return '';
    }
    if (abs.startsWith('http://') || abs.startsWith('https://')) {
      const sep = abs.includes('?') ? '&' : '?';
      const cacheBuster = Math.floor(Date.now() / 1000);
      return `${abs}${sep}v=${cacheBuster}`;
    }
    return abs;
  }, [company?.logo_url]);

  useEffect(() => {
    setLogoError(false);
  }, [company?.logo_url]);

  const homeHref = routePublicId ? `/dashboard/company/${routePublicId}` : '/dashboard/company';

  const { isCompanyAuthReady } = useCompanyAuthToken();
  const { delayCount, hasCriticalDelays } = useDispatchDelays(null, 120000, isCompanyAuthReady);

  const todayLabel = useMemo(() => formatToday(), []);

  return (
    <header className={styles.header} role="banner">
      {/* ── Zone gauche — Identité entreprise ── */}
      <div className={styles.headerLeft}>
        <Link to={homeHref} className={styles.brand} aria-label="Tableau de bord entreprise">
          <div className={styles.logoWrap}>
            {logoSrc && !logoError ? (
              <img
                src={logoSrc}
                alt=""
                className={styles.logoImg}
                onError={() => setLogoError(true)}
                onLoad={() => setLogoError(false)}
                loading="eager"
              />
            ) : (
              <div className={styles.logoFallback} aria-hidden="true">
                {getInitials(name)}
              </div>
            )}
          </div>
          <span className={styles.companyName}>{name}</span>
        </Link>
      </div>

      {/* ── Zone droite — Indicateurs ── */}
      <div className={styles.headerRight}>
        {/* Date du jour */}
        <div className={styles.indicator}>
          <FiCalendar className={styles.indicatorIcon} />
          <span className={styles.indicatorText}>{todayLabel}</span>
        </div>

        {/* Retards */}
        {delayCount > 0 && (
          <>
            <div className={styles.headerDivider} />
            <Link
              to={`/dashboard/company/${routePublicId}/dispatch/monitor`}
              className={`${styles.delayIndicator} ${hasCriticalDelays ? styles.delayIndicatorCritical : ''}`}
              title={`${delayCount} retard(s) détecté(s)`}
            >
              <FiAlertTriangle className={styles.indicatorIcon} />
              <span className={styles.indicatorText}>
                {delayCount} retard{delayCount > 1 ? 's' : ''}
              </span>
            </Link>
          </>
        )}

        <div className={styles.headerDivider} />

        {/* Notifications */}
        <CompanyNotificationBell />

        <div className={styles.headerDivider} />

        {/* Socket.IO */}
        <SocketStatusBadge />
      </div>
    </header>
  );
};

export default CompanyHeader;
