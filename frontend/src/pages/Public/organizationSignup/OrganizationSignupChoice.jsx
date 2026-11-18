import { Link } from 'react-router-dom';
import { SLA_NOTICE } from '../../../services/organizationApplicationService';
import styles from './OrganizationSignup.module.css';

const PROFILES = [
  {
    href: '/login?mode=signup',
    title: 'Particulier',
    text: 'Créer un compte client pour réserver un transport.',
  },
  {
    href: '/inscription/organisation/entreprise',
    title: 'Entreprise de transport',
    text: 'Déposer une demande. Aucun accès n’est ouvert avant examen.',
  },
  {
    href: '/inscription/organisation/institution',
    title: 'Institution',
    text: 'Déposer une demande pour un établissement. L’examen reste chez LIRIE.',
  },
];

export default function OrganizationSignupChoice() {
  return (
    <div className={styles.page}>
      <div className={styles.shell}>
        <p className={styles.kicker}>Rejoindre LIRIE</p>
        <h1 className={styles.title}>Choisir un profil</h1>
        <p className={styles.lead}>
          Le particulier crée son compte. L’entreprise et l’institution déposent une demande.
        </p>
        <div className={styles.choices}>
          {PROFILES.map((profile) => (
            <Link key={profile.href} className={styles.profile} to={profile.href}>
              <strong>{profile.title}</strong>
              <span>{profile.text}</span>
            </Link>
          ))}
        </div>
        <p className={styles.note}>
          Votre entreprise de transport doit vous inviter à rejoindre LIRIE.
        </p>
        <p className={styles.notice}>{SLA_NOTICE}</p>
      </div>
    </div>
  );
}
