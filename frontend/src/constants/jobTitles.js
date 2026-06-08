// Fonction / metier (job_title) — donnee descriptive/organisationnelle.
// Independante du role LIRIE (institution_role) : n'accorde AUCUNE permission.
// Reutilisable : creation utilisateur, edition, futur import CSV, filtres, stats.

export const JOB_TITLE_MAX_LENGTH = 120;

const RAW_JOB_TITLES = [
  'Infirmier(ère)',
  'ASSC',
  'Aide-soignant(e)',
  'Réceptionniste',
  'Secrétaire médicale',
  'Médecin',
  'Physiothérapeute',
  'Ergothérapeute',
  'Psychologue',
  'Assistant(e) social(e)',
  'Éducateur(trice)',
  'Coordinateur(trice)',
  'Responsable de service',
  'Administration',
];

// Trie alphabetiquement (locale FR, insensible a la casse/accents) pour que
// les suggestions restent ordonnees meme si on ajoute des entrees plus tard.
export const JOB_TITLE_OPTIONS = [...RAW_JOB_TITLES].sort((a, b) =>
  a.localeCompare(b, 'fr', { sensitivity: 'base' })
);
