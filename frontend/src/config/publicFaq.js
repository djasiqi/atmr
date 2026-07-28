/**
 * FAQ visibles sur /aide — source unique pour le DOM et le JSON-LD FAQPage.
 */

export const FAQ_PATIENTS = [
  {
    q: 'Comment demander un transport ?',
    a: 'Selon votre situation, la demande peut être organisée par votre institution, votre professionnel de santé ou via votre accès personnel lorsqu’il est disponible. L’équipe LIRIE peut vous orienter si vous n’êtes pas certain du circuit à emprunter.',
  },
  {
    q: 'Puis-je réserver directement sur LIRIE ?',
    a: 'Oui : lorsque vous disposez d’un accès personnel sur la plateforme ou que votre organisation le prévoit, vous pouvez réserver directement sur LIRIE. Selon les conventions locales et l’organisation mise en place avec votre institution ou le transporteur partenaire, la demande peut aussi être initiée par l’institution ou un coordinateur — renseignez-vous auprès de votre contact pour savoir quel circuit s’applique à votre situation.',
  },
  {
    q: 'Qui réalise le transport ?',
    a: 'Les missions sont effectuées par des entreprises de transport partenaires juridiquement indépendantes. LIRIE coordonne l’attribution — elle ne conduit pas les véhicules.',
  },
  {
    q: 'Puis-je modifier ou annuler une course ?',
    a: 'Oui, lorsque votre accès le permet : vous pouvez le faire depuis votre portail LIRIE selon les options proposées pour votre mission. Sinon, ou si vous avez besoin d’aide, contactez l’acteur ayant organisé le transport (institution, service coordinateur ou transporteur partenaire). Selon les conditions applicables (délai, convention locale), des frais peuvent s’appliquer — renseignez-vous auprès de votre institution ou du transporteur.',
  },
  {
    q: 'Comment savoir si le transport est confirmé ?',
    a: 'Consultez le statut de votre mission dans votre espace compte / portail LIRIE lorsque vous y avez accès — il reflète l’avancement et la confirmation lorsque les acteurs concernés l’ont saisi. En cas de doute ou si vous n’avez pas d’accès, contactez l’acteur qui a effectué ou organisé la réservation (institution, coordinateur ou transporteur partenaire).',
  },
];

export const FAQ_INSTITUTIONS = [
  {
    q: 'À quoi sert LIRIE pour une institution ?',
    a: 'LIRIE facilite la coordination des transports entre institutions et entreprises partenaires dans un environnement partagé. Vous disposez d’un tableau de bord pour planifier, suivre et gérer les missions en temps réel.',
  },
  {
    q: 'LIRIE remplace-t-elle nos transporteurs actuels ?',
    a: 'Non. La plateforme coordonne vos partenaires existants. Vous conservez vos relations contractuelles — LIRIE apporte la couche technologique de coordination.',
  },
  {
    q: 'Peut-on travailler avec plusieurs transporteurs ?',
    a: 'Oui. LIRIE est conçu nativement pour la coordination multi-transporteurs. Vous pouvez configurer vos priorités et règles d’attribution selon vos besoins.',
  },
  {
    q: 'Qui accède aux informations de mission ?',
    a: 'Les accès sont configurés selon les rôles définis dans votre organisation. Chaque profil ne voit que les données correspondant à son périmètre.',
  },
  {
    q: 'Comment organiser une présentation de la plateforme ?',
    a: 'Vous pouvez contacter l’équipe LIRIE pour planifier une démonstration adaptée à votre structure et à vos besoins spécifiques.',
  },
];

export const FAQ_CHAUFFEURS = [
  {
    q: 'Puis-je travailler directement pour LIRIE ?',
    a: 'Non. LIRIE n’est pas un employeur. Les missions sont réalisées via des entreprises de transport partenaires juridiquement indépendantes.',
  },
  {
    q: 'Puis-je travailler comme indépendant ?',
    a: 'Oui, si vous exercez dans une structure enregistrée disposant d’un numéro IDE et des autorisations nécessaires pour le transport concerné.',
  },
  {
    q: 'Puis-je utiliser mon véhicule personnel ?',
    a: 'Uniquement si celui-ci respecte la réglementation applicable au type de transport concerné (homologation, assurance, équipements requis).',
  },
  {
    q: 'Qui me rémunère ?',
    a: 'Votre employeur ou votre entreprise indépendante. LIRIE n’est en aucun cas l’employeur des chauffeurs opérant sur la plateforme.',
  },
  {
    q: 'Comment rejoindre le réseau ?',
    a: 'Contactez l’équipe partenaires LIRIE pour l’étude de votre situation. L’intégration se fait au travers d’une entreprise de transport enregistrée.',
  },
];

export const FAQ_ENTREPRISES = [
  {
    q: 'Comment intégrer le réseau LIRIE ?',
    a: 'L’intégration se fait progressivement après un échange avec l’équipe partenaires. Un accompagnement est prévu pour la mise en place technique et opérationnelle.',
  },
  {
    q: 'Peut-on connecter plusieurs chauffeurs ?',
    a: 'Oui. L’accès est configurable selon votre organisation interne — flottes et équipes de toute taille peuvent être intégrées.',
  },
  {
    q: 'Peut-on publier des offres chauffeurs ?',
    a: 'Oui. Les entreprises partenaires peuvent diffuser des opportunités de recrutement via le réseau LIRIE.',
  },
  {
    q: 'LIRIE modifie-t-elle la relation avec nos clients ?',
    a: 'Non. La plateforme facilite la coordination opérationnelle sans toucher à vos relations contractuelles existantes avec vos clients institutionnels.',
  },
];

export const FAQ_SITUATIONS = [
  {
    q: 'Je ne vois pas ma course dans l’application',
    a: 'Contactez l’acteur ayant organisé le transport (institution ou transporteur). La visibilité d’une course dépend de votre rôle et de la configuration de votre accès.',
  },
  {
    q: 'Le chauffeur est en retard',
    a: 'Adressez-vous directement au transporteur responsable de la mission. Ils ont les outils pour localiser le véhicule et vous informer du délai.',
  },
  {
    q: 'Le statut de la course ne se met pas à jour',
    a: 'Les mises à jour de statut dépendent des étapes validées par les intervenants terrain. Si le problème persiste, contactez l’organisateur du transport.',
  },
  {
    q: 'Je dois modifier ou annuler une réservation',
    a: 'Contactez l’institution ou le transporteur ayant organisé la mission. Les modifications doivent être traitées par l’acteur responsable de la course.',
  },
  {
    q: 'Je ne sais pas qui contacter',
    a: 'Utilisez la page contact LIRIE — notre équipe vous orientera vers le bon interlocuteur selon votre situation.',
  },
  {
    q: 'LIRIE prend-elle en charge les urgences médicales ?',
    a: 'Non. LIRIE coordonne des transports médicaux non urgents et accompagnés. En cas d’urgence vitale, composez le 144.',
  },
  {
    q: 'Dans quelles régions LIRIE est-elle disponible ?',
    a: 'Le déploiement principal concerne Genève et la Suisse romande, avec des partenaires actifs selon les zones couvertes.',
  },
];

/** Liste plate pour JSON-LD FAQPage (contenu réellement affiché sur /aide). */
export function listAllPublicFaqItems() {
  return [
    ...FAQ_PATIENTS,
    ...FAQ_INSTITUTIONS,
    ...FAQ_CHAUFFEURS,
    ...FAQ_ENTREPRISES,
    ...FAQ_SITUATIONS,
  ];
}

export function buildFaqPageStructuredData() {
  return {
    '@type': 'FAQPage',
    '@id': 'https://www.lirie.ch/aide#faq',
    mainEntity: listAllPublicFaqItems().map((item) => ({
      '@type': 'Question',
      name: item.q,
      acceptedAnswer: {
        '@type': 'Answer',
        text: item.a,
      },
    })),
  };
}
