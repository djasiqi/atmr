import React, { Suspense, useEffect, useRef, Component } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import CompanyShellProvider from '../../providers/CompanyShellProvider';
import CompanyHeader from './Header/CompanyHeader';
import CompanySidebar from './Sidebar/CompanySidebar/CompanySidebar';
import shellStyles from '../../pages/company/Dashboard/CompanyDashboard.module.css';

/** Fallback léger pour les routes entreprise lazy — shell déjà visible. */
function CompanyOutletFallback() {
  return (
    <main className={shellStyles.content} aria-busy="true">
      <div
        style={{
          width: '100%',
          height: '18px',
          borderRadius: '8px',
          marginBottom: '14px',
          background: 'linear-gradient(90deg, #eef2f7 0%, #f8fafc 50%, #eef2f7 100%)',
        }}
      />
      <div
        style={{
          width: '100%',
          height: '140px',
          borderRadius: '12px',
          background: 'linear-gradient(90deg, #eef2f7 0%, #f8fafc 50%, #eef2f7 100%)',
        }}
      />
      <div style={{ fontSize: '15px', color: '#64748b', marginTop: '16px' }}>
        Chargement du module…
      </div>
    </main>
  );
}

class CompanyOutletErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    if (process.env.NODE_ENV === 'development') {
      console.error('[CompanyEnterpriseLayout] erreur contenu central', error?.message || error);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className={shellStyles.content} role="alert">
          <h1 style={{ fontSize: '1.25rem', marginBottom: 8 }}>Impossible d’afficher cette page</h1>
          <p style={{ color: '#64748b' }}>
            Une erreur est survenue. Le menu reste disponible — essayez une autre section ou rechargez.
          </p>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false })}
            style={{
              marginTop: 12,
              padding: '8px 14px',
              borderRadius: 8,
              border: '1px solid #cbd5e1',
              background: '#fff',
              cursor: 'pointer',
            }}
          >
            Réessayer
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}

/**
 * Prefetch chunk intentionnel (pointerenter / focus) — respect saveData / 2G.
 */
export function prefetchCompanyRouteChunk(loader) {
  if (typeof loader !== 'function') return;
  try {
    const conn = typeof navigator !== 'undefined' ? navigator.connection : null;
    if (conn?.saveData) return;
    if (conn?.effectiveType && /2g/i.test(conn.effectiveType)) return;
  } catch {
    // ignore
  }
  void loader().catch(() => {});
}

/**
 * Layout shell persistant : Header + Sidebar uniques pour les 13 routes + /demo.
 * Suspense + Error Boundary autour du contenu central uniquement.
 * Pas de prefetch Maps ni factures au cold start (Lot 2).
 */
export default function CompanyEnterpriseLayout() {
  const location = useLocation();
  const announceRef = useRef(null);

  useEffect(() => {
    const el = announceRef.current;
    if (!el) return;
    const path = location.pathname || '';
    el.textContent = `Page chargée : ${path}`;
  }, [location.pathname]);

  return (
    <CompanyShellProvider>
      <div className={shellStyles.companyContainer}>
        <CompanyHeader />
        <div className={shellStyles.dashboard}>
          <CompanySidebar />
          <div
            ref={announceRef}
            style={{
              position: 'absolute',
              width: 1,
              height: 1,
              padding: 0,
              margin: -1,
              overflow: 'hidden',
              clip: 'rect(0, 0, 0, 0)',
              whiteSpace: 'nowrap',
              border: 0,
            }}
            aria-live="polite"
            aria-atomic="true"
          />
          <CompanyOutletErrorBoundary key={location.pathname}>
            <Suspense fallback={<CompanyOutletFallback />}>
              <Outlet />
            </Suspense>
          </CompanyOutletErrorBoundary>
        </div>
      </div>
    </CompanyShellProvider>
  );
}
