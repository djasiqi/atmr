export const CANCELLATION_REASONS = [
  {
    code: 'LAST_MINUTE',
    label: 'Annulation dernière minute',
    description: 'Annulation tardive côté client ou organisation',
    isClientFault: true,
  },
  {
    code: 'NO_SHOW',
    label: 'Client ne s\'est pas présenté',
    description: 'Le client n\'était pas au lieu de rendez-vous',
    isClientFault: true,
  },
  {
    code: 'CLIENT_REQUEST',
    label: 'Client a demandé l\'annulation',
    description: 'Le client a demandé d\'annuler la course',
    isClientFault: true,
  },
  {
    code: 'COMPANY_ISSUE',
    label: 'Problème entreprise',
    description: 'Problème technique ou organisationnel',
    isClientFault: false,
  },
  {
    code: 'MAJOR_DELAY',
    label: 'Retard important',
    description: 'Retard trop important pour honorer la course',
    isClientFault: false,
  },
  {
    code: 'VEHICLE_ISSUE',
    label: 'Problème véhicule',
    description: 'Panne ou problème mécanique',
    isClientFault: false,
  },
  {
    code: 'OTHER',
    label: 'Autre raison',
    description: 'Autre raison nécessitant une justification',
    isClientFault: false,
  },
];
