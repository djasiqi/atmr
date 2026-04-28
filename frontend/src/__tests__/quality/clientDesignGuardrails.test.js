import fs from 'fs';
import path from 'path';

const filesToCheck = [
  'src/pages/client/Dashboard/ClientDashboard.jsx',
  'src/pages/client/Reservations/ReservationsPage.jsx',
  'src/pages/client/Account/AccountUser.jsx',
];

const forbiddenPatterns = [
  { label: 'emoji dans les titres client', regex: /<h[1-3][^>]*>[^<]*[\u{1F300}-\u{1FAFF}]/u },
  { label: 'gradient décoratif', regex: /linear-gradient\s*\(/i },
  {
    label: 'jargon technique visible client',
    regex: />[^<]*(dispatch|assignation|ops monitor|synchronisation dégradée)[^<]*</i,
  },
];

describe('Client design guardrails', () => {
  it.each(filesToCheck)('respecte les garde-fous visuels: %s', (relativePath) => {
    const absolutePath = path.resolve(process.cwd(), relativePath);
    const content = fs.readFileSync(absolutePath, 'utf8');
    forbiddenPatterns.forEach(({ regex }) => {
      expect(content).not.toMatch(regex);
    });
  });
});

