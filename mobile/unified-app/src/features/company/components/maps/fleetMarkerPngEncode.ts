type Rgba = { r: number; g: number; b: number; a: number };

const uriCache = new Map<string, string>();

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const PNG_IHDR = new Uint8Array([73, 72, 68, 82]); // IHDR
const PNG_IDAT = new Uint8Array([73, 68, 65, 84]); // IDAT
const PNG_IEND = new Uint8Array([73, 69, 78, 68]); // IEND

function writeUint32Be(target: Uint8Array, offset: number, value: number): void {
  target[offset] = (value >>> 24) & 0xff;
  target[offset + 1] = (value >>> 16) & 0xff;
  target[offset + 2] = (value >>> 8) & 0xff;
  target[offset + 3] = value & 0xff;
}

function adler32(data: Uint8Array): number {
  let a = 1;
  let b = 0;
  const mod = 65521;
  for (let i = 0; i < data.length; i += 1) {
    a = (a + data[i]) % mod;
    b = (b + a) % mod;
  }
  return ((b << 16) | a) >>> 0;
}

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(parts: Uint8Array[]): number {
  let c = 0xffffffff;
  for (const part of parts) {
    for (let i = 0; i < part.length; i += 1) {
      c = CRC32_TABLE[(c ^ part[i]) & 0xff] ^ (c >>> 8);
    }
  }
  return (c ^ 0xffffffff) >>> 0;
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((sum, p) => sum + p.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

function makePngChunk(type: Uint8Array, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(4 + 4 + data.length + 4);
  writeUint32Be(out, 0, data.length);
  out.set(type, 4);
  out.set(data, 8);
  writeUint32Be(out, 8 + data.length, crc32([type, data]));
  return out;
}

/**
 * Deflate "stored blocks" (sans compression) encapsule en zlib.
 * Compatible PNG sans dependance Node.
 */
function zlibStore(raw: Uint8Array): Uint8Array {
  const chunks: Uint8Array[] = [];
  const zlibHeader = new Uint8Array([0x78, 0x01]); // deflate, no compression/fastest
  chunks.push(zlibHeader);

  let offset = 0;
  while (offset < raw.length) {
    const remaining = raw.length - offset;
    const blockLen = Math.min(65535, remaining);
    const finalFlag = offset + blockLen >= raw.length ? 1 : 0;

    const header = new Uint8Array(5);
    header[0] = finalFlag; // BFINAL + BTYPE=00
    header[1] = blockLen & 0xff;
    header[2] = (blockLen >>> 8) & 0xff;
    const nlen = (~blockLen) & 0xffff;
    header[3] = nlen & 0xff;
    header[4] = (nlen >>> 8) & 0xff;
    chunks.push(header);
    chunks.push(raw.subarray(offset, offset + blockLen));
    offset += blockLen;
  }

  const ad = adler32(raw);
  const trailer = new Uint8Array(4);
  writeUint32Be(trailer, 0, ad);
  chunks.push(trailer);
  return concatBytes(chunks);
}

function base64Encode(bytes: Uint8Array): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i] ?? 0;
    const b1 = bytes[i + 1] ?? 0;
    const b2 = bytes[i + 2] ?? 0;
    const triple = (b0 << 16) | (b1 << 8) | b2;
    const hasB1 = i + 1 < bytes.length;
    const hasB2 = i + 2 < bytes.length;

    out += alphabet[(triple >>> 18) & 0x3f];
    out += alphabet[(triple >>> 12) & 0x3f];
    out += hasB1 ? alphabet[(triple >>> 6) & 0x3f] : "=";
    out += hasB2 ? alphabet[triple & 0x3f] : "=";
  }
  return out;
}

function encodeRgbaToPngBase64(width: number, height: number, rgba: Uint8Array): string {
  const bytesPerRow = width * 4;
  const raw = new Uint8Array(height * (1 + bytesPerRow));
  for (let y = 0; y < height; y += 1) {
    const rawOffset = y * (1 + bytesPerRow);
    const rowOffset = y * bytesPerRow;
    raw[rawOffset] = 0; // filter none
    raw.set(rgba.subarray(rowOffset, rowOffset + bytesPerRow), rawOffset + 1);
  }

  const ihdr = new Uint8Array(13);
  writeUint32Be(ihdr, 0, width);
  writeUint32Be(ihdr, 4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const compressed = zlibStore(raw);
  const pngBytes = concatBytes([
    PNG_SIGNATURE,
    makePngChunk(PNG_IHDR, ihdr),
    makePngChunk(PNG_IDAT, compressed),
    makePngChunk(PNG_IEND, new Uint8Array(0)),
  ]);

  return base64Encode(pngBytes);
}

function parseHexColor(hex: string, alpha = 255): Rgba {
  const normalized = hex.replace("#", "");
  const full =
    normalized.length === 3
      ? normalized
          .split("")
          .map((c) => c + c)
          .join("")
      : normalized;
  const value = Number.parseInt(full, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
    a: alpha,
  };
}

function setPixel(data: Uint8Array, width: number, x: number, y: number, color: Rgba): void {
  if (x < 0 || y < 0) return;
  const idx = (y * width + x) << 2;
  if (idx < 0 || idx + 3 >= data.length) return;
  const srcA = color.a / 255;
  const dstA = data[idx + 3] / 255;
  const outA = srcA + dstA * (1 - srcA);
  if (outA <= 0) return;
  data[idx] = Math.round((color.r * srcA + data[idx] * dstA * (1 - srcA)) / outA);
  data[idx + 1] = Math.round((color.g * srcA + data[idx + 1] * dstA * (1 - srcA)) / outA);
  data[idx + 2] = Math.round((color.b * srcA + data[idx + 2] * dstA * (1 - srcA)) / outA);
  data[idx + 3] = Math.round(outA * 255);
}

function fillCircle(
  data: Uint8Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
  color: Rgba
): void {
  const r2 = radius * radius;
  const minX = Math.max(0, Math.floor(cx - radius));
  const maxX = Math.min(width - 1, Math.ceil(cx + radius));
  const minY = Math.max(0, Math.floor(cy - radius));
  const maxY = Math.min(height - 1, Math.ceil(cy + radius));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        setPixel(data, width, x, y, color);
      }
    }
  }
}

function strokeCircle(
  data: Uint8Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
  strokePx: number,
  color: Rgba
): void {
  const outer = radius + strokePx / 2;
  const inner = Math.max(0, radius - strokePx / 2);
  const outer2 = outer * outer;
  const inner2 = inner * inner;
  const minX = Math.max(0, Math.floor(cx - outer));
  const maxX = Math.min(width - 1, Math.ceil(cx + outer));
  const minY = Math.max(0, Math.floor(cy - outer));
  const maxY = Math.min(height - 1, Math.ceil(cy + outer));
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 <= outer2 && d2 >= inner2) {
        setPixel(data, width, x, y, color);
      }
    }
  }
}

function fillRoundRect(
  data: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  w: number,
  h: number,
  radius: number,
  color: Rgba
): void {
  const cx1 = x + radius;
  const cy1 = y + radius;
  const cx2 = x + w - radius;
  const cy2 = y + h - radius;
  for (let py = y; py < y + h; py += 1) {
    for (let px = x; px < x + w; px += 1) {
      const inRect =
        (px >= x + radius && px < x + w - radius) ||
        (py >= y + radius && py < y + h - radius) ||
        (px - cx1) ** 2 + (py - cy1) ** 2 <= radius ** 2 ||
        (px - cx2) ** 2 + (py - cy1) ** 2 <= radius ** 2 ||
        (px - cx1) ** 2 + (py - cy2) ** 2 <= radius ** 2 ||
        (px - cx2) ** 2 + (py - cy2) ** 2 <= radius ** 2;
      if (inRect) setPixel(data, width, px, py, color);
    }
  }
}

/** Police bitmap 5×5 (chiffres, symboles ETA, initiales chauffeur a–z). */
const DIGIT_FONT: Record<string, number[]> = {
  "0": [0x1c, 0x22, 0x22, 0x22, 0x1c],
  "1": [0x08, 0x18, 0x08, 0x08, 0x1c],
  "2": [0x1c, 0x02, 0x1c, 0x20, 0x1e],
  "3": [0x1e, 0x02, 0x0c, 0x02, 0x1c],
  "4": [0x12, 0x12, 0x1e, 0x02, 0x02],
  "5": [0x1e, 0x20, 0x1c, 0x02, 0x1c],
  "6": [0x0c, 0x10, 0x1c, 0x22, 0x1c],
  "7": [0x1e, 0x02, 0x04, 0x08, 0x08],
  "8": [0x1c, 0x22, 0x1c, 0x22, 0x1c],
  "9": [0x1c, 0x22, 0x1e, 0x02, 0x0c],
  "+": [0x00, 0x08, 0x1c, 0x08, 0x00],
  "~": [0x00, 0x00, 0x12, 0x0c, 0x00],
  " ": [0x00, 0x00, 0x00, 0x00, 0x00],
  a: [0x0e, 0x11, 0x1f, 0x11, 0x11],
  b: [0x1e, 0x10, 0x1e, 0x10, 0x1e],
  c: [0x0e, 0x11, 0x11, 0x11, 0x0e],
  d: [0x0e, 0x11, 0x11, 0x11, 0x1e],
  e: [0x1f, 0x10, 0x1e, 0x10, 0x1f],
  f: [0x1e, 0x10, 0x1c, 0x10, 0x10],
  g: [0x0e, 0x11, 0x11, 0x0e, 0x01],
  h: [0x10, 0x10, 0x1e, 0x11, 0x11],
  i: [0x04, 0x00, 0x0c, 0x04, 0x0e],
  j: [0x02, 0x00, 0x06, 0x02, 0x0c],
  k: [0x10, 0x12, 0x1c, 0x12, 0x11],
  l: [0x0c, 0x04, 0x04, 0x04, 0x0e],
  m: [0x00, 0x1a, 0x15, 0x11, 0x11],
  n: [0x00, 0x16, 0x19, 0x11, 0x11],
  o: [0x0e, 0x11, 0x11, 0x11, 0x0e],
  p: [0x1e, 0x11, 0x1e, 0x10, 0x10],
  q: [0x0e, 0x11, 0x11, 0x0e, 0x01],
  r: [0x16, 0x19, 0x10, 0x10, 0x10],
  s: [0x0e, 0x10, 0x0e, 0x01, 0x0e],
  t: [0x1f, 0x04, 0x04, 0x04, 0x04],
  u: [0x11, 0x11, 0x11, 0x11, 0x0e],
  v: [0x11, 0x11, 0x11, 0x0a, 0x04],
  w: [0x11, 0x11, 0x15, 0x15, 0x0a],
  x: [0x11, 0x0a, 0x04, 0x0a, 0x11],
  y: [0x11, 0x11, 0x0e, 0x02, 0x04],
  z: [0x1f, 0x02, 0x04, 0x08, 0x1f],
};

const GLYPH_ROWS = 5;
const GLYPH_COLS = 5;
const DEFAULT_GLYPH_ADVANCE = 6;

function normalizeMarkerGlyphChar(raw: string): string {
  return raw.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
}

function measureBitmapLabelWidth(charCount: number, textScale: number, glyphAdvance: number): number {
  return charCount * glyphAdvance * textScale;
}

/** Calcule échelle et espacement pour que les initiales restent dans le disque. */
export function resolveDriverMarkerLabelBitmapLayout(
  sizePx: number,
  label: string
): { trimmed: string; textScale: number; glyphAdvance: number } {
  const trimmed = label.trim().slice(0, 2).toUpperCase();
  const mapScale = sizePx / 24;
  const coreR = 8.2 * mapScale;
  const strokeW = 2.5;
  const innerDiameter = (coreR - strokeW * 0.55) * 2;
  const charCount = trimmed.length > 0 ? trimmed.length : 1;
  const glyphAdvance = trimmed.length >= 2 ? 5 : DEFAULT_GLYPH_ADVANCE;
  const maxTextW = innerDiameter * 0.8;
  const maxTextH = innerDiameter * 0.58;

  let textScale = Math.min(
    Math.floor(maxTextH / GLYPH_ROWS),
    Math.floor(maxTextW / (charCount * glyphAdvance))
  );
  const upperBound = Math.round(2.5 * mapScale);
  textScale = Math.max(2, Math.min(textScale, upperBound));

  return { trimmed, textScale, glyphAdvance };
}

function drawText(
  data: Uint8Array,
  width: number,
  height: number,
  text: string,
  startX: number,
  startY: number,
  color: Rgba,
  scale = 1,
  glyphAdvance = DEFAULT_GLYPH_ADVANCE
): void {
  let cursorX = startX;
  for (const raw of text) {
    const glyph = DIGIT_FONT[normalizeMarkerGlyphChar(raw)];
    if (!glyph) continue;
    for (let row = 0; row < glyph.length; row += 1) {
      const bits = glyph[row] ?? 0;
      for (let col = 0; col < GLYPH_COLS; col += 1) {
        if ((bits >> (GLYPH_COLS - 1 - col)) & 1) {
          for (let sy = 0; sy < scale; sy += 1) {
            for (let sx = 0; sx < scale; sx += 1) {
              setPixel(data, width, cursorX + col * scale + sx, startY + row * scale + sy, color);
            }
          }
        }
      }
    }
    cursorX += glyphAdvance * scale;
  }
}

function drawCenteredMarkerLabel(
  data: Uint8Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  label: string,
  color: Rgba,
  layout: { trimmed: string; textScale: number; glyphAdvance: number }
): void {
  const { trimmed, textScale, glyphAdvance } = layout;
  if (trimmed.length === 0) return;
  const textHeight = GLYPH_ROWS * textScale;
  const textWidth = measureBitmapLabelWidth(trimmed.length, textScale, glyphAdvance);
  drawText(
    data,
    width,
    height,
    trimmed,
    Math.round(cx - textWidth / 2),
    Math.round(cy - textHeight / 2),
    color,
    textScale,
    glyphAdvance
  );
}

export function encodeFleetMarkerPngDataUri(
  cacheKey: string,
  width: number,
  height: number,
  paint: (data: Uint8Array, width: number, height: number) => void
): string {
  const hit = uriCache.get(cacheKey);
  if (hit) return hit;

  const rgba = new Uint8Array(width * height * 4);
  paint(rgba, width, height);
  const base64 = encodeRgbaToPngBase64(width, height, rgba);
  const uri = `data:image/png;base64,${base64}`;
  uriCache.set(cacheKey, uri);
  return uri;
}

export function buildDriverMarkerPngUri(options: {
  fill: string;
  opacity: number;
  label: string;
  sizePx: number;
}): string {
  const { fill, opacity, label, sizePx } = options;
  const key = `driver:${fill}:${opacity}:${label}:${sizePx}`;
  return encodeFleetMarkerPngDataUri(key, sizePx, sizePx, (data, w, h) => {
    const cx = w / 2;
    const cy = h / 2;
    const scale = sizePx / 24;
    const coreR = 8.2 * scale;
    const strokeW = 2.5;
    const fillColor = parseHexColor(fill, Math.round(opacity * 255));
    strokeCircle(data, w, h, cx, cy, coreR, strokeW, parseHexColor("#ffffff", 245));
    fillCircle(data, w, h, cx, cy, coreR - strokeW * 0.4, fillColor);
    const layout = resolveDriverMarkerLabelBitmapLayout(sizePx, label);
    drawCenteredMarkerLabel(
      data,
      w,
      h,
      cx,
      cy,
      label,
      parseHexColor("#ffffff", 255),
      layout
    );
  });
}

export function buildClusterMarkerPngUri(count: number, sizePx: number, fillColor: string): string {
  const key = `cluster:${count}:${sizePx}:${fillColor}`;
  return encodeFleetMarkerPngDataUri(key, sizePx, sizePx, (data, w, h) => {
    const cx = w / 2;
    const cy = h / 2;
    const r = w / 2 - 2;
    const fill = parseHexColor(fillColor, 255);
    fillCircle(data, w, h, cx, cy, r, fill);
    strokeCircle(data, w, h, cx, cy, r, 2.5, parseHexColor("#ffffff", 255));
    const label = count > 99 ? "99" : String(count);
    const textScale = 2;
    const textW = label.length * 6 * textScale;
    drawText(
      data,
      w,
      h,
      label,
      Math.round(cx - textW / 2),
      Math.round(cy - 5 * textScale),
      parseHexColor("#ffffff", 255),
      textScale
    );
  });
}

export function buildClusterCountBadgePngUri(
  label: string,
  widthPx: number,
  heightPx: number
): string {
  const key = `cluster-badge:${label}:${widthPx}:${heightPx}`;
  return encodeFleetMarkerPngDataUri(key, widthPx, heightPx, (data, w, h) => {
    const cx = w / 2;
    const cy = h / 2;
    const border = 3;
    const teal = parseHexColor("#0F766E", 255);
    const white = parseHexColor("#ffffff", 255);

    if (label.length <= 1) {
      const r = Math.min(w, h) / 2 - border;
      fillCircle(data, w, h, cx, cy, r + 1, parseHexColor("#0f172a", 55));
      fillCircle(data, w, h, cx, cy, r, teal);
      strokeCircle(data, w, h, cx, cy, r, border, white);
    } else {
      fillRoundRect(data, w, h, 0, 0, w, h, h / 2, parseHexColor("#0f172a", 50));
      fillRoundRect(data, w, h, 0, 0, w, h, h / 2, white);
      fillRoundRect(
        data,
        w,
        h,
        border,
        border,
        w - border * 2,
        h - border * 2,
        (h - border * 2) / 2,
        teal
      );
    }

    const scale = 3;
    const textW = label.length * 6 * scale;
    const textH = 5 * scale;
    drawText(
      data,
      w,
      h,
      label,
      Math.round(cx - textW / 2),
      Math.round(cy - textH / 2),
      white,
      scale
    );
  });
}

export function buildMissionAnchorPngUri(options: {
  fill: string;
  stroke: string;
  radiusPx: number;
  selected: boolean;
  halo: boolean;
}): string {
  const pad = options.selected ? 7 : 5;
  const sizePx = options.radiusPx * 2 + pad;
  const key = `anchor:${options.fill}:${options.stroke}:${options.radiusPx}:${options.selected}:${options.halo}`;
  return encodeFleetMarkerPngDataUri(key, sizePx, sizePx, (data, w, h) => {
    const cx = w / 2;
    const cy = h / 2;
    const r = options.radiusPx;
    if (options.halo) {
      fillCircle(data, w, h, cx, cy, r + 3, parseHexColor(options.fill, 46));
    }
    strokeCircle(data, w, h, cx, cy, r, options.selected ? 3 : 2, parseHexColor(options.stroke, 242));
    fillCircle(data, w, h, cx, cy, r - 0.5, parseHexColor(options.fill, 255));
  });
}

export function buildEtaBadgePngUri(label: string): { uri: string; width: number; height: number } {
  const trimmed = label.trim().slice(0, 14);
  const height = 26;
  const padX = 10;
  const width = Math.max(58, Math.min(112, Math.round(trimmed.length * 6.1 + padX * 2)));
  const key = `eta:${trimmed}:${width}`;
  const uri = encodeFleetMarkerPngDataUri(key, width, height, (data, w, h) => {
    fillRoundRect(data, w, h, 0, 0, w, h, h / 2, parseHexColor("#00796B", 245));
    strokeCircle(data, w, h, w / 2, h / 2, Math.min(w, h) / 2 - 1, 1, parseHexColor("#ffffff", 170));
    const textW = trimmed.length * 6;
    drawText(data, w, h, trimmed, Math.round(w / 2 - textW / 2), 8, parseHexColor("#ffffff", 255), 1);
  });
  return { uri, width, height };
}

export function clearFleetMarkerPngCache(): void {
  uriCache.clear();
}
