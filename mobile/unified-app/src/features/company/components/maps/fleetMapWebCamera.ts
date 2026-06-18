import type { Region } from "react-native";

import type { FleetMapMarker } from "./fleetMapTypes";



const FIT_BOUNDS_PADDING_PX = 52;

/** Marge minimale si tous les chauffeurs sont au même point (~1 km). */

const MIN_BOUNDS_SPAN_DEG = 0.012;



export type FleetMapLatLng = { latitude: number; longitude: number };



/** Toutes les positions chauffeur visibles (y compris membres d’un cluster). */

export function collectFleetMarkerPositions(markers: FleetMapMarker[]): FleetMapLatLng[] {

  const out: FleetMapLatLng[] = [];

  for (const m of markers) {

    if (m.kind === "driver") {

      out.push({ latitude: m.driver.latitude, longitude: m.driver.longitude });

    } else {

      for (const d of m.drivers) {

        out.push({ latitude: d.latitude, longitude: d.longitude });

      }

    }

  }

  return out;

}



export function collectValidFleetPositions(

  points: FleetMapLatLng[]

): FleetMapLatLng[] {

  return points.filter(

    (p) =>

      Number.isFinite(p.latitude) &&

      Number.isFinite(p.longitude) &&

      Math.abs(p.latitude) <= 90 &&

      Math.abs(p.longitude) <= 180

  );

}



/** Étend les bornes si les points sont confondus (fitBounds sinon sans effet visible). */

export function expandFleetBoundsSpan(points: FleetMapLatLng[]): FleetMapLatLng[] {

  if (points.length === 0) return points;

  const lats = points.map((p) => p.latitude);

  const lngs = points.map((p) => p.longitude);

  const minLat = Math.min(...lats);

  const maxLat = Math.max(...lats);

  const minLng = Math.min(...lngs);

  const maxLng = Math.max(...lngs);

  const latSpan = maxLat - minLat;

  const lngSpan = maxLng - minLng;

  if (latSpan >= MIN_BOUNDS_SPAN_DEG && lngSpan >= MIN_BOUNDS_SPAN_DEG) {

    return points;

  }

  const latPad = Math.max(MIN_BOUNDS_SPAN_DEG, latSpan) / 2;

  const lngPad = Math.max(MIN_BOUNDS_SPAN_DEG, lngSpan) / 2;

  const cLat = (minLat + maxLat) / 2;

  const cLng = (minLng + maxLng) / 2;

  return [

    { latitude: cLat + latPad, longitude: cLng + lngPad },

    { latitude: cLat - latPad, longitude: cLng - lngPad },

  ];

}



type GmapsLatLng = new (lat: number, lng: number) => unknown;

type GmapsBounds = new () => { extend: (p: unknown) => void };



export type FleetWebMapCamera = {

  panTo?: (center: unknown) => void;

  setCenter: (center: unknown) => void;

  setZoom: (z: number) => void;

  fitBounds: (bounds: unknown, padding?: number | Record<string, number>) => void;

  getZoom?: () => number;

};



export type FleetCameraPadding = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

const DEFAULT_FIT_PADDING: FleetCameraPadding = {
  top: FIT_BOUNDS_PADDING_PX,
  right: FIT_BOUNDS_PADDING_PX,
  bottom: FIT_BOUNDS_PADDING_PX,
  left: FIT_BOUNDS_PADDING_PX,
};

export function resolveFleetFitPadding(
  padding?: Partial<FleetCameraPadding> | number
): FleetCameraPadding | number {
  if (padding == null) return DEFAULT_FIT_PADDING;
  if (typeof padding === "number") return padding;
  return {
    top: padding.top ?? DEFAULT_FIT_PADDING.top,
    right: padding.right ?? DEFAULT_FIT_PADDING.right,
    bottom: padding.bottom ?? DEFAULT_FIT_PADDING.bottom,
    left: padding.left ?? DEFAULT_FIT_PADDING.left,
  };
}



function fitMapToPositions(

  map: FleetWebMapCamera,

  LatLng: GmapsLatLng,

  LatLngBounds: GmapsBounds,

  positions: FleetMapLatLng[],

  fitPadding: FleetCameraPadding | number = DEFAULT_FIT_PADDING

): void {

  const valid = collectValidFleetPositions(positions);

  if (valid.length === 0) return;



  if (valid.length === 1) {

    const d = valid[0];

    const center = new LatLng(d.latitude, d.longitude);

    if (map.panTo) map.panTo(center);

    else map.setCenter(center);

    map.setZoom(14);

    return;

  }



  const bounds = new LatLngBounds();

  for (const p of expandFleetBoundsSpan(valid)) {

    bounds.extend(new LatLng(p.latitude, p.longitude));

  }

  map.fitBounds(bounds, fitPadding);

}



/** Cadre tous les chauffeurs visibles sur la carte (bouton « Recentrer »). */

export function applyFleetFitAllDrivers(

  map: FleetWebMapCamera,

  LatLng: GmapsLatLng,

  LatLngBounds: GmapsBounds,

  drivers: FleetMapLatLng[],

  markers: FleetMapMarker[],

  defaultCenter: FleetMapLatLng,

  fitPadding?: Partial<FleetCameraPadding> | number

): void {

  const padding = resolveFleetFitPadding(fitPadding);

  const fromDrivers = collectValidFleetPositions(drivers);

  const fromMarkers = collectValidFleetPositions(collectFleetMarkerPositions(markers));

  const positions = fromDrivers.length > 0 ? fromDrivers : fromMarkers;



  if (positions.length === 0) {

    const center = new LatLng(defaultCenter.latitude, defaultCenter.longitude);

    if (map.panTo) map.panTo(center);

    else map.setCenter(center);

    map.setZoom(11);

    return;

  }



  fitMapToPositions(map, LatLng, LatLngBounds, positions, padding);

}



export function regionToZoom(latitudeDelta: number): number {

  const delta = Math.max(latitudeDelta, 0.008);

  return Math.min(16, Math.max(10, Math.round(Math.log2(360 / delta))));

}



/** Recentrage ciblé (chauffeur sélectionné / urgent). */

export function applyFleetWebCamera(

  map: FleetWebMapCamera,

  LatLng: GmapsLatLng,

  LatLngBounds: GmapsBounds,

  options: {

    recenterRegion: Region | null | undefined;

    markers: FleetMapMarker[];

    defaultCenter: FleetMapLatLng;

    fitPadding?: Partial<FleetCameraPadding> | number;

  }

): void {

  const { recenterRegion, markers, defaultCenter, fitPadding } = options;



  if (recenterRegion) {

    const center = new LatLng(recenterRegion.latitude, recenterRegion.longitude);

    const targetZoom = regionToZoom(recenterRegion.latitudeDelta ?? 0.024);

    if (map.panTo) {

      map.panTo(center);

    } else {

      map.setCenter(center);

    }

    const current = map.getZoom?.() ?? targetZoom;

    if (Math.abs(current - targetZoom) > 0.5) {

      map.setZoom(targetZoom);

    }

    return;

  }



  applyFleetFitAllDrivers(map, LatLng, LatLngBounds, [], markers, defaultCenter, fitPadding);

}


