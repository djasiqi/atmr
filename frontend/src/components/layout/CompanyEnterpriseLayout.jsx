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

/** Chunk webpack introuvable après déploiement (hash obsolète en cache navigateur). */
function isChunkLoadError(error) {
  if (!error) return false;
  if (error.name === 'ChunkLoadError') return true;
  const msg = String(error.message || error || '');
  return (
    /Loading chunk [\w.-]+ failed/i.test(msg) ||
    /Failed to fetch dynamically imported module/i.test(msg) ||
    /Importing a module script failed/i.test(msg)
  );
}

function reloadOnceFromChunkError() {
  if (typeof window === 'undefined') return;
  if (window.__RELOADING_FROM_CHUNK_ERROR__) return;
  window.__RELOADING_FROM_CHUNK_ERROR__ = true;
  window.location.reload();
}

class CompanyOutletErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Toujours logger : nécessaire pour diagnostiquer le fallback « Impossible d’afficher ».
    console.error(
      '[CompanyEnterpriseLayout] erreur contenu central',
      error?.message || error,
      errorInfo?.componentStack || '',
    );
    // lazy() rejette dans React → ce boundary masque le reload global de index.js.
    if (isChunkLoadError(error)) {
      reloadOnceFromChunkError();
    }
  }

  render() {
    if (this.state.hasError) {
      const chunkStale = isChunkLoadError(this.state.error);
      const detail =
        this.state.error?.message ||
        (typeof this.state.error === 'string' ? this.state.error : null);
      return (
        <main className={shellStyles.content} role="alert">
          <h1 style={{ fontSize: '1.25rem', marginBottom: 8 }}>
            {chunkStale
              ? 'Nouvelle version disponible'
              : 'Impossible d’afficher cette page'}
          </h1>
          <p style={{ color: '#64748b' }}>
            {chunkStale
              ? 'L’application a été mise à jour. Rechargez pour charger les derniers modules.'
              : 'Une erreur est survenue. Le menu reste disponible — essayez une autre section ou rechargez.'}
          </p>
          {!chunkStale && detail ? (
            <pre
              style={{
                marginTop: 12,
                padding: 12,
                borderRadius: 8,
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                color: '#334155',
                fontSize: 12,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                maxWidth: '48rem',
              }}
            >
              {detail}
            </pre>
          ) : null}
          <button
            type="button"
            onClick={() => {
              if (chunkStale) {
                reloadOnceFromChunkError();
                return;
              }
              this.setState({ hasError: false, error: null });
            }}
            style={{
              marginTop: 12,
              padding: '8px 14px',
              borderRadius: 8,
              border: '1px solid #cbd5e1',
              background: '#fff',
              cursor: 'pointer',
            }}
          >
            {chunkStale ? 'Recharger la page' : 'Réessayer'}
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
