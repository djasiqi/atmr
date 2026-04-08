import { useEffect, useRef, useMemo } from 'react';
import { useGoogleMap } from '@react-google-maps/api';
import { iconAnchorToAdvancedMarkerCss, GOOGLE_MAPS_USE_JS_STYLES } from '../../utils/mapUtils';

/**
 * Ancrage CSS depuis `icon` (AdvancedMarkerElement uniquement).
 */
function iconToAnchorCss(icon) {
  if (!icon?.scaledSize || !icon?.anchor) {
    return { anchorLeft: '-50%', anchorTop: '-50%' };
  }
  const w = icon.scaledSize.width;
  const h = icon.scaledSize.height;
  const ax = typeof icon.anchor.x === 'function' ? icon.anchor.x() : icon.anchor.x;
  const ay = typeof icon.anchor.y === 'function' ? icon.anchor.y() : icon.anchor.y;
  return iconAnchorToAdvancedMarkerCss(ax, ay, w, h);
}

function buildLegacyIcon(icon) {
  if (!icon?.url || !window.google?.maps) return undefined;
  return {
    url: icon.url,
    scaledSize: icon.scaledSize,
    anchor: icon.anchor,
  };
}

/**
 * Marqueur carte : AdvancedMarkerElement en mode `cloud`, ou `google.maps.Marker` en mode `js` (style Lirie en JS).
 */
export default function GoogleMapsAdvancedMarker({
  position,
  icon,
  title,
  zIndex,
  onClick,
}) {
  const map = useGoogleMap();
  const markerRef = useRef(null);
  const imgRef = useRef(null);
  const onClickRef = useRef(onClick);
  onClickRef.current = onClick;

  const anchors = useMemo(() => iconToAnchorCss(icon), [icon]);

  useEffect(() => {
    if (!map || !icon?.url || !window.google?.maps) return;

    if (GOOGLE_MAPS_USE_JS_STYLES) {
      const marker = new window.google.maps.Marker({
        map,
        position,
        icon: buildLegacyIcon(icon),
        zIndex: zIndex ?? 0,
        title: title || undefined,
        optimized: true,
      });
      markerRef.current = marker;
      const listener = marker.addListener('click', () => {
        onClickRef.current?.();
      });
      return () => {
        if (listener && window.google?.maps?.event?.removeListener) {
          window.google.maps.event.removeListener(listener);
        }
        marker.setMap(null);
        markerRef.current = null;
      };
    }

    if (!window.google?.maps?.marker?.AdvancedMarkerElement) return;

    const img = document.createElement('img');
    img.src = icon.url;
    img.alt = title || '';
    if (title) img.title = title;
    if (icon.scaledSize) {
      img.width = icon.scaledSize.width;
      img.height = icon.scaledSize.height;
    }
    img.style.display = 'block';
    img.draggable = false;

    const marker = new window.google.maps.marker.AdvancedMarkerElement({
      map,
      position,
      content: img,
      anchorLeft: anchors.anchorLeft,
      anchorTop: anchors.anchorTop,
      zIndex: zIndex ?? 0,
      title: title || undefined,
      gmpClickable: true,
    });

    imgRef.current = img;
    markerRef.current = marker;

    const listener = marker.addListener('click', () => {
      onClickRef.current?.();
    });

    return () => {
      if (listener && window.google?.maps?.event?.removeListener) {
        window.google.maps.event.removeListener(listener);
      }
      marker.map = null;
      markerRef.current = null;
      imgRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- cycle de vie du marqueur ; position/title/zIndex via effets dédiés
  }, [map, icon?.url, anchors.anchorLeft, anchors.anchorTop]);

  useEffect(() => {
    const m = markerRef.current;
    if (!m || !position) return;
    if (GOOGLE_MAPS_USE_JS_STYLES) {
      m.setPosition(position);
    } else {
      m.position = position;
    }
  }, [position]);

  useEffect(() => {
    if (GOOGLE_MAPS_USE_JS_STYLES) return;
    const img = imgRef.current;
    if (img && title != null) {
      img.title = title;
      img.alt = title;
    }
  }, [title]);

  useEffect(() => {
    if (GOOGLE_MAPS_USE_JS_STYLES) {
      const m = markerRef.current;
      if (m && icon?.url) {
        m.setIcon(buildLegacyIcon(icon));
      }
      if (m && zIndex != null) m.setZIndex(zIndex);
      return;
    }
    const img = imgRef.current;
    const m = markerRef.current;
    if (img && icon?.url) {
      img.src = icon.url;
      if (icon.scaledSize) {
        img.width = icon.scaledSize.width;
        img.height = icon.scaledSize.height;
      }
    }
    if (m) {
      m.anchorLeft = anchors.anchorLeft;
      m.anchorTop = anchors.anchorTop;
      if (zIndex != null) m.zIndex = zIndex;
    }
  }, [icon, anchors, zIndex]);

  useEffect(() => {
    if (!GOOGLE_MAPS_USE_JS_STYLES) return;
    const m = markerRef.current;
    if (m && title != null) m.setTitle(title);
  }, [title]);

  return null;
}
