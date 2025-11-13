/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const EXAMPLE_FILE = path.resolve(__dirname, '..', 'env.example');

if (!fs.existsSync(EXAMPLE_FILE)) {
  console.error('❌ Fichier env.example introuvable.');
  process.exit(1);
}

const [, , envArg = 'development'] = process.argv;

const normalizeTargetName = (arg) => {
  if (arg.startsWith('.env')) return arg;
  if (arg === 'production') return '.env.production';
  if (arg === 'development' || arg === 'dev') return '.env.development';
  return `.env.${arg}`;
};

const TARGET_FILE = path.resolve(__dirname, '..', normalizeTargetName(envArg));

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const question = (query) =>
  new Promise((resolve) => rl.question(query, (answer) => resolve(answer.trim())));

const rawLines = fs.readFileSync(EXAMPLE_FILE, 'utf8').split(/\r?\n/);

const entries = rawLines.map((line) => {
  if (!line || line.trim().startsWith('#') || !line.includes('=')) {
    return { type: 'literal', value: line };
  }
  const [key, ...rest] = line.split('=');
  return { type: 'env', key: key.trim(), value: rest.join('=').trim() };
});

async function confirmOverwrite() {
  if (!fs.existsSync(TARGET_FILE)) return true;
  const answer = await question(
    `⚠️  ${path.basename(TARGET_FILE)} existe déjà. Écraser ? (y/N) `
  );
  return ['y', 'yes', 'o', 'oui'].includes(answer.toLowerCase());
}

async function promptValues() {
  const answers = [];
  console.log(`\n🛠  Génération de ${path.basename(TARGET_FILE)} (profil ${envArg})`);
  console.log('Laisse vide pour conserver la valeur proposée.\n');

  for (const entry of entries) {
    if (entry.type === 'literal') {
      answers.push(entry);
      continue;
    }
    const answer = await question(
      `${entry.key} [${entry.value || 'vide'}] : `
    );
    answers.push({
      ...entry,
      value: answer.length > 0 ? answer : entry.value,
    });
  }
  return answers;
}

async function main() {
  const canProceed = await confirmOverwrite();
  if (!canProceed) {
    console.log('➡️  Abandon. Aucun fichier créé.');
    rl.close();
    return;
  }

  const answers = await promptValues();
  rl.close();

  const output = answers
    .map((entry) =>
      entry.type === 'literal' ? entry.value : `${entry.key}=${entry.value ?? ''}`
    )
    .join('\n');

  fs.writeFileSync(TARGET_FILE, `${output}\n`, 'utf8');
  console.log(`\n✅ ${path.basename(TARGET_FILE)} généré avec succès.`);
}

main().catch((err) => {
  console.error('❌ Erreur lors de la génération du fichier env:', err);
  rl.close();
  process.exit(1);
});

