/**
 * Garde anti-régression : aucun composant du frontend ne doit décider qu'une ligne facture est
 * « scindable » (sans retour / sans aller / retrait complet A/R) à partir des seuls drapeaux
 * informatifs `billing_unit === 'round_trip'`, `transport_type === 'A/R'` ou des prédicats
 * d'information `isSingleMergedRoundTripLine` / `isAnyRoundTripLine`.
 *
 * Ces drapeaux signifient « la réservation a été créée en aller-retour », pas « les deux jambes
 * sont facturées sur cette ligne ». Toute décision passe par `roundTripLineStructure`,
 * `invoiceLineHasBothRoundTripLegs` ou `canShowRoundTripLegExcludeActions`
 * (`src/utils/invoiceLineRoundTrip.js`), seul module autorisé à lire ces drapeaux.
 */
import fs from 'fs';
import path from 'path';

const SRC_ROOT = path.resolve(__dirname, '..', '..');
const ALLOWED_FLAG_READERS = new Set([path.join(SRC_ROOT, 'utils', 'invoiceLineRoundTrip.js')]);
const SOURCE_EXT = new Set(['.js', '.jsx', '.ts', '.tsx']);
const COMPARISON = /(?:===|!==|==|!=)/;
const INFO_ONLY_HELPERS = ['isSingleMergedRoundTripLine', 'isAnyRoundTripLine'];

function isTestFile(filePath) {
  return (
    /[\\/]__tests__[\\/]/.test(filePath) ||
    /\.(test|spec)\.[jt]sx?$/.test(filePath) ||
    /[\\/]setupTests\.[jt]s$/.test(filePath)
  );
}

function listSourceFiles(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === '__tests__') continue;
      listSourceFiles(full, out);
      continue;
    }
    if (!SOURCE_EXT.has(path.extname(entry.name))) continue;
    if (isTestFile(full)) continue;
    out.push(full);
  }
  return out;
}

function isCommentLine(line) {
  const t = line.trim();
  return t.startsWith('//') || t.startsWith('*') || t.startsWith('/*');
}

/** Lignes qui comparent un drapeau informatif à sa valeur « A/R » (hors commentaires). */
function flagComparisonViolations(filePath, content) {
  const violations = [];
  content.split(/\r?\n/).forEach((line, idx) => {
    if (isCommentLine(line) || !COMPARISON.test(line)) return;
    const billingUnit = /\bbilling_unit\b/.test(line) && /['"`]round_trip['"`]/.test(line);
    const transportType = /\btransport_type\b/.test(line) && /['"`]A\/R['"`]/.test(line);
    if (billingUnit || transportType) {
      violations.push(`${path.relative(SRC_ROOT, filePath)}:${idx + 1}: ${line.trim()}`);
    }
  });
  return violations;
}

/** Imports des prédicats d'information depuis le module A/R (réservés au badge, dans le module). */
function infoHelperImportViolations(filePath, content) {
  const violations = [];
  const importRe = /import\s*\{([^}]*)\}\s*from\s*['"][^'"]*invoiceLineRoundTrip['"]/g;
  let m;
  while ((m = importRe.exec(content)) !== null) {
    const names = m[1]
      .split(',')
      .map((s) =>
        s
          .trim()
          .split(/\s+as\s+/)[0]
          .trim()
      )
      .filter(Boolean);
    for (const name of names) {
      if (INFO_ONLY_HELPERS.includes(name)) {
        violations.push(`${path.relative(SRC_ROOT, filePath)}: import de ${name}`);
      }
    }
  }
  return violations;
}

describe('garde A/R : les drapeaux informatifs ne pilotent aucune action hors du module dédié', () => {
  const files = listSourceFiles(SRC_ROOT).filter((f) => !ALLOWED_FLAG_READERS.has(f));

  it('scanne bien le code source du frontend', () => {
    expect(files.length).toBeGreaterThan(50);
    expect(fs.existsSync([...ALLOWED_FLAG_READERS][0])).toBe(true);
  });

  it("aucun composant ne compare billing_unit à 'round_trip' ni transport_type à 'A/R'", () => {
    const violations = files.flatMap((f) =>
      flagComparisonViolations(f, fs.readFileSync(f, 'utf8'))
    );
    expect(violations).toEqual([]);
  });

  it('aucun composant n’importe les prédicats d’information (badge) pour décider', () => {
    const violations = files.flatMap((f) =>
      infoHelperImportViolations(f, fs.readFileSync(f, 'utf8'))
    );
    expect(violations).toEqual([]);
  });
});
