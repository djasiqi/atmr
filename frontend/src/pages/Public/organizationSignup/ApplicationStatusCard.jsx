import styles from './OrganizationSignup.module.css';

const STATUS_NOTES = {
  email_unverified: 'Confirmez votre adresse avec le dernier lien reçu. Les liens plus anciens ne fonctionnent plus.',
  received: 'Votre dossier est reçu. Aucun accès professionnel n’est ouvert.',
  in_review: 'LIRIE examine le dossier. Aucune action n’est demandée ici.',
  needs_info: 'LIRIE demandera le complément directement. Cette page ne sert pas à y répondre.',
  approved: 'La demande est approuvée. L’ouverture du portail ne se fait pas depuis cette page.',
  refused: 'La demande est refusée. Aucun accès n’a été ouvert.',
  abandoned: 'Cette demande n’est plus suivie.',
};

export default function ApplicationStatusCard({ application }) {
  const events = Array.isArray(application.events) ? application.events : [];
  const note = STATUS_NOTES[application.status] || '';
  return (
    <section className={styles.panel} aria-live="polite">
      <p className={styles.kicker}>{application.kind_label}</p>
      <h2 className={styles.title}>{application.organization_name}</h2>
      <p className={styles.lead}>
        <strong>{application.status_label}</strong>
        {' · '}
        {application.applicant_name}
      </p>
      <p className={styles.email}>{application.applicant_email}</p>
      {note ? <p className={styles.hint}>{note}</p> : null}
      <p className={styles.notice}>{application.sla_notice}</p>
      {application.existing_organization_hint ? (
        <p className={styles.hint}>
          Une organisation déjà enregistrée correspond à ce nom ou à cet identifiant.
          Aucun accès ne vous a été ouvert, et aucun rattachement n’a été fait.
        </p>
      ) : null}
      {events.length > 0 ? (
        <ol className={styles.history}>
          {events.map((event) => (
            <li key={`${event.type}-${event.at}`}>
              {event.label}
              {event.note ? <span className={styles.email}>{event.note}</span> : null}
            </li>
          ))}
        </ol>
      ) : null}
    </section>
  );
}
