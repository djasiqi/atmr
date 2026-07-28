import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getPublicSeoForPath } from '../../config/publicSeo';
import PublicSeo from './PublicSeo';
import PrivateSeo from './PrivateSeo';
import { trackPublicSeoPageView } from '../../utils/seoAnalytics';

/**
 * Applique PublicSeo ou PrivateSeo selon la route courante.
 * Doit être monté sous BrowserRouter.
 */
export default function RouteSeoManager() {
  const { pathname } = useLocation();
  const publicSeo = getPublicSeoForPath(pathname);

  useEffect(() => {
    if (publicSeo?.path) {
      trackPublicSeoPageView(publicSeo.path);
    }
  }, [publicSeo?.path]);

  if (publicSeo) {
    return (
      <PublicSeo
        title={publicSeo.title}
        description={publicSeo.description}
        path={publicSeo.path}
        canonicalUrl={publicSeo.canonicalUrl}
        image={publicSeo.image}
        structuredData={publicSeo.structuredData}
      />
    );
  }

  return <PrivateSeo />;
}
