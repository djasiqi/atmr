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

import { useQuery } from '@tanstack/react-query';
import { useLirieCompany } from '../../../hooks/useLirieCompany';
import useCompanyAuthToken from '../../../hooks/useCompanyAuthToken';
import { fetchDispatchDelays } from '../../../services/companyService';
import { lirieKeys } from '../../../queryKeys/lirie';
import resolveLogoUrl from '../../../utils/resolveLogoUrl';
import SocketStatusBadge from '../../common/SocketStatusBadge';
import CompanyNotificationBell from './CompanyNotificationBell';
import { getAuthEnv } from '../../../utils/webAuthSession';

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

/** Clé jour dispatch (YYYY-MM-DD, UTC) — alignée sur les autres écrans entreprise. */
function todayYmd() {
  return new Date().toISOString().slice(0, 10);
}

const CompanyHeader = () => {
  const params = useParams();
  const location = useLocation();

  const routePublicId =
    params.public_id ||
    (() => {
      const match = location.pathname.match(/(?:\/demo)?\/dashboard\/company\/([^/]+)/);
      return match?.[1] || null;
    })();
  const isDemoEnv =
    location.pathname.startsWith('/demo/') ||
    getAuthEnv() === 'demo';
  const dashboardRoot = isDemoEnv ? '/demo/dashboard' : '/dashboard';

  const { company } = useLirieCompany();

  const [logoError, setLogoError] = useState(false);
  const name = company?.name || 'Entreprise';

  const logoSrc = useMemo(() => {
    const abs = resolveLogoUrl(company?.logo_url);
    if (!abs) return '';
    if (abs && !abs.startsWith('http') && !abs.startsWith('data:') && !abs.startsWith('blob:')) {
      return '';
    }
    return abs;
  }, [company?.logo_url]);

  useEffect(() => {
    setLogoError(false);
  }, [company?.logo_url]);

  const homeHref = routePublicId
    ? `${dashboardRoot}/company/${routePublicId}`
    : `${dashboardRoot}/company`;

  const { isCompanyAuthReady } = useCompanyAuthToken();
  const todayKey = todayYmd();
  const { data: headerDelays = [] } = useQuery({
    queryKey: lirieKeys.dispatchDelays(todayKey),
    queryFn: () => fetchDispatchDelays(todayKey),
    staleTime: 20_000,
    enabled: isCompanyAuthReady,
  });
  const delayCount = useMemo(() => {
    const seen = new Set();
    for (const row of headerDelays || []) {
      if (!row?.booking_id || !(Number(row.delay_minutes) > 0)) continue;
      seen.add(row.booking_id);
    }
    return seen.size;
  }, [headerDelays]);
  const hasCriticalDelays = useMemo(
    () => (headerDelays || []).some((d) => Number(d?.delay_minutes || 0) >= 30),
    [headerDelays]
  );

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
                width="36"
                height="36"
                loading="eager"
                decoding="async"
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
              to={`${dashboardRoot}/company/${routePublicId}/dispatch/monitor`}
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
