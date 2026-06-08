import React from 'react';
import Header from '../../components/layout/Header/Header';
import Footer from '../../components/layout/Footer/Footer';

const DefaultLayout = ({ children, compactMain = false, hideAuthEntry = false }) => {
  return (
    <div style={styles.container}>
      <Header hideAuthEntry={hideAuthEntry} />
      <main style={compactMain ? styles.mainCompact : styles.main}>{children}</main>
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
