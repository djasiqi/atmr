export const CONTACT_CATEGORIES = [
  {
    key: 'support',
    index: '01',
    label: 'Support technique',
    description: "Assistance liee a l'utilisation de la plateforme.",
    route: '/contact/support',
  },
  {
    key: 'institution',
    index: '02',
    label: 'Institution / Integration',
    description: 'EMS, clinique, hopital, curatelle : mise en place et integration.',
    route: '/contact/institution',
  },
  {
    key: 'transport',
    index: '03',
    label: 'Entreprise de transport',
    description: 'Partenariat, deploiement et questions operationnelles.',
    route: '/contact/transport',
  },
  {
    key: 'demo',
    index: '04',
    label: 'Demonstration',
    description: 'Presentation structuree de la plateforme.',
    route: '/contact/demo',
  },
  {
    key: 'billing',
    index: '05',
    label: 'Facturation',
    description: 'Questions administratives et financieres.',
    route: '/contact/billing',
  },
  {
    key: 'family',
    index: '06',
    label: 'Famille / Proche aidant',
    description: 'Information ou orientation.',
    route: '/contact/family',
  },
];

export const getCategory = (key) => CONTACT_CATEGORIES.find((item) => item.key === key) || null;

export const listCategories = () =>
  [...CONTACT_CATEGORIES].sort((a, b) => Number(a.index) - Number(b.index));

const commonFields = [
  { type: 'text', name: 'name', label: 'Nom *', required: true, placeholder: 'Nom et prenom' },
  { type: 'email', name: 'email', label: 'Email *', required: true, placeholder: 'nom@organisation.ch' },
  { type: 'text', name: 'phone', label: 'Telephone (optionnel)', required: false, placeholder: '+41 79 123 45 67' },
];

const consentText =
  "Vos donnees sont traitees conformement a notre politique de confidentialite. Elles ne sont jamais revendues.";

export const fieldsByCategory = {
  support: {
    introTitle: 'Support technique',
    introText: "Pour toute question liee a l'utilisation operationnelle de la plateforme LIRIE, notre equipe vous repond rapidement.",
    submitLabel: 'Transmettre la demande',
    trackingName: 'contact_support_submit',
    fields: [
      ...commonFields,
      {
        type: 'text',
        name: 'organization',
        label: 'Entreprise ou institution (optionnel)',
        required: false,
        placeholder: "Nom de l'organisation",
      },
      {
        type: 'select',
        name: 'subject_detail',
        label: 'Sujet *',
        required: true,
        placeholder: 'Selectionner un sujet',
        options: [
          { value: 'access', label: 'Connexion / acces' },
          { value: 'booking', label: 'Reservation / planning' },
          { value: 'billing', label: 'Facturation / document' },
          { value: 'bug', label: 'Incident technique / anomalie' },
          { value: 'other', label: 'Autre' },
        ],
      },
      {
        type: 'text',
        name: 'reference',
        label: 'Reference (optionnel)',
        required: false,
        placeholder: 'Ex: BK-20481',
      },
      {
        type: 'select',
        name: 'urgency',
        label: "Niveau d'urgence",
        required: false,
        placeholder: "Selectionner un niveau",
        options: [
          { value: 'normal', label: 'Standard' },
          { value: 'priority', label: 'Prioritaire (impact operationnel)' },
        ],
      },
      {
        type: 'textarea',
        name: 'message',
        label: 'Description du probleme *',
        required: true,
        placeholder:
          "Decrivez le contexte, les etapes reproduites et l'impact constate (si possible, ajoutez la date/heure).",
      },
    ],
    consentText,
  },
  institution: {
    introTitle: 'Institutions et integration',
    introText: 'EMS, clinique, hopital, curatelle : mise en place, flux et integrations.',
    submitLabel: 'Envoyer la demande',
    trackingName: 'contact_institution_submit',
    fields: [
      ...commonFields,
      { type: 'text', name: 'organization', label: 'Organisation / etablissement *', required: true },
      {
        type: 'select',
        name: 'organization_type',
        label: "Type d'etablissement *",
        required: true,
        options: [
          { value: 'ems', label: 'EMS' },
          { value: 'clinic', label: 'Clinique' },
          { value: 'hospital', label: 'Hopital' },
          { value: 'curatorship', label: 'Curatelle / mandataire' },
          { value: 'other', label: 'Autre' },
        ],
      },
      { type: 'text', name: 'sites_count', label: 'Nombre de sites', required: false },
      {
        type: 'select',
        name: 'integration_required',
        label: 'Integration requise *',
        required: true,
        options: [
          { value: 'yes', label: 'Oui' },
          { value: 'no', label: 'Non' },
          { value: 'evaluate', label: 'A evaluer' },
        ],
      },
      { type: 'text', name: 'integration_system', label: 'Systeme principal', required: false },
      { type: 'textarea', name: 'message', label: 'Message *', required: true },
    ],
    consentText,
  },
  transport: {
    introTitle: 'Entreprise de transport',
    introText: 'Partenariat, deploiement, questions operationnelles.',
    submitLabel: 'Envoyer la demande',
    trackingName: 'contact_transport_submit',
    fields: [
      ...commonFields,
      { type: 'text', name: 'organization', label: 'Nom entreprise *', required: true },
      { type: 'text', name: 'fleet_size_range', label: 'Nombre de chauffeurs', required: false },
      { type: 'text', name: 'service_area', label: "Zone d'activite", required: false },
      {
        type: 'select',
        name: 'subject_detail',
        label: 'Interet *',
        required: true,
        options: [
          { value: 'partnership', label: 'Partenariat' },
          { value: 'deployment', label: 'Deploiement' },
          { value: 'information', label: 'Information' },
        ],
      },
      { type: 'textarea', name: 'message', label: 'Message *', required: true },
    ],
    consentText,
  },
  demo: {
    introTitle: 'Demande de demonstration',
    introText: 'Une presentation structuree, adaptee a votre contexte.',
    submitLabel: 'Envoyer la demande',
    trackingName: 'contact_demo_submit',
    fields: [
      ...commonFields,
      { type: 'text', name: 'organization', label: 'Organisation *', required: true },
      {
        type: 'select',
        name: 'organization_type',
        label: "Type d'organisation *",
        required: true,
        options: [
          { value: 'transport', label: 'Entreprise de transport' },
          { value: 'institution', label: 'Institution' },
          { value: 'curatorship', label: 'Curatelle / mandataire' },
        ],
      },
      {
        type: 'select',
        name: 'timing',
        label: 'Timing du projet *',
        required: true,
        options: [
          { value: 'immediate', label: 'Immediat' },
          { value: 'one_three_months', label: '1-3 mois' },
          { value: 'three_plus_months', label: '> 3 mois' },
          { value: 'exploration', label: 'Exploration' },
        ],
      },
      {
        type: 'select',
        name: 'preferred_slot',
        label: 'Creneau souhaite *',
        required: true,
        options: [
          { value: 'this_week', label: 'Cette semaine' },
          { value: 'next_week', label: 'Semaine prochaine' },
          { value: 'to_schedule', label: 'A convenir' },
        ],
      },
      {
        type: 'select',
        name: 'volume_range',
        label: 'Volumetrie',
        required: false,
        options: [
          { value: '1_5', label: '1-5 utilisateurs' },
          { value: '5_20', label: '5-20 utilisateurs' },
          { value: '20_100', label: '20-100 utilisateurs' },
          { value: '100_plus', label: '> 100 utilisateurs' },
        ],
      },
      { type: 'textarea', name: 'message', label: 'Commentaire *', required: true },
    ],
    consentText,
  },
  billing: {
    introTitle: 'Facturation',
    introText: 'Questions administratives et financieres.',
    submitLabel: 'Envoyer la demande',
    trackingName: 'contact_billing_submit',
    fields: [
      ...commonFields,
      { type: 'text', name: 'organization', label: 'Organisation', required: false },
      { type: 'text', name: 'reference', label: 'Reference de facture', required: false },
      { type: 'textarea', name: 'message', label: 'Message *', required: true },
    ],
    consentText,
  },
  family: {
    introTitle: 'Familles et proches aidants',
    introText: "Si vous hesitez sur le bon canal, ecrivez-nous ici : nous orienterons votre demande.",
    submitLabel: 'Envoyer la demande',
    trackingName: 'contact_family_submit',
    fields: [
      { type: 'text', name: 'name', label: 'Nom *', required: true },
      { type: 'email', name: 'email', label: 'Email *', required: true },
      { type: 'text', name: 'situation', label: 'Situation', required: false },
      { type: 'textarea', name: 'message', label: 'Message *', required: true },
    ],
    consentText,
  },
};
