export const WORST_CASE_APPLICATION = {
  kind: 'institution',
  kind_label: 'Institution',
  status: 'received',
  status_label: 'Dossier reçu',
  organization_name: 'Établissement médico-social de la Côte et du Jorat — site de Villeneuve',
  applicant_name: 'Aleksandra Wiśniewska-Kowalczyk',
  applicant_email: 'bartholomew.fitzgerald@northwind-industries-holdings.example.com',
  sla_notice:
    'Votre demande sera examinée dans un délai maximal de 24 heures administratives cumulées, à compter de la réception de votre dossier complet.',
  existing_organization_hint: true,
  email_verified: true,
  events: [
    { type: 'created', label: 'Demande enregistrée', at: '2026-10-12T08:00:00+00:00' },
    { type: 'verification_sent', label: 'Lien de confirmation envoyé', at: '2026-10-12T08:00:01+00:00' },
    { type: 'email_verified', label: 'Adresse e-mail confirmée', at: '2026-10-12T08:05:00+00:00' },
    { type: 'received', label: 'Dossier reçu', at: '2026-10-12T08:05:01+00:00' },
  ],
};
