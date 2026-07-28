import React from 'react';
import { useLocation } from 'react-router-dom';
import Header from '../../components/layout/Header/Header';
import Footer from '../../components/layout/Footer/Footer';
import { normalizePublicSeoPath } from '../../config/publicSeo';

const DefaultLayout = ({ children, compactMain = false, hideAuthEntry = false }) => {
  const { pathname } = useLocation();
  const isPublicSeoPage = Boolean(normalizePublicSeoPath(pathname));

  return (
    <div style={styles.container}>
      <Header hideAuthEntry={hideAuthEntry} />
      <main
        style={compactMain ? styles.mainCompact : styles.main}
        {...(isPublicSeoPage ? { 'data-seo-ready': 'true' } : {})}
      >
        {children}
      </main>
      <Footer />
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
  },
  main: {
    flex: '1',
    backgroundColor: '#ffffff',
  },
  mainCompact: {
    flex: '0 0 auto',
  },
};

export default DefaultLayout;
