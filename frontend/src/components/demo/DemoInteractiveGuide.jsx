import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './DemoInteractiveGuide.module.css';

const TOOLTIP_WIDTH = 360;
const TOOLTIP_HEIGHT = 220;
const RECT_EPSILON = 1;
const WAIT_FOR_TARGET_TIMEOUT_MS = 3000;
const ASSIGN_MODAL_TARGET_TIMEOUT_MS = 1200;
const DISPATCH_DETAIL_PANEL_TIMEOUT_MS = 2000;
const INVOICE_MODAL_TARGET_TIMEOUT_MS = 2200;
const INSTITUTION_RESUME_STEP_KEY = 'demo_institution_resume_step';
const DEMO_INSTITUTION_COMPLETED_KEY = 'demo_institution_journey_completed';
const MODAL_FORM_SELECTOR = '[data-tour-id="manual-booking-form"]';
const ASSIGN_DRIVER_MODAL_SELECTOR = '[data-tour-id="assign-driver-modal"]';
const INVOICE_MODAL_SELECTOR = '[data-tour-id="invoice-new-modal"]';

const MODAL_STEP_IDS = new Set([
  'booking-left-panel',
  'booking-medical-section',
  'booking-submit',
  'assign-driver-option',
  'invoice-modal-billing-type',
  'invoice-modal-client',
  'invoice-modal-period',
  'invoice-modal-submit',
  'institution-request-form',
  'institution-request-details',
  'institution-request-patient',
  'institution-request-destination',
  'institution-request-datetime',
  'institution-request-submit',
]);

const GUIDE_STEPS = {
  transporteur: [
    {
      id: 'demo-welcome',
      title: 'Bienvenue dans votre démo guidée',
      description:
        'En quelques étapes, vous allez créer une réservation, la suivre dans le tableau des courses, puis attribuer un chauffeur.',
      actionLabel: "Action attendue : cliquez sur « Suivant » pour commencer.",
    },
    {
      id: 'kpi-grid',
      selector: '[data-tour-id="kpi-grid"]',
      title: 'Lire les indicateurs du jour',
      description:
        'Ces indicateurs résument la situation immédiate : courses en cours, retards, courses à attribuer et disponibilité des chauffeurs.',
    },
    {
      id: 'create-booking',
      selector: '[data-tour-id="create-booking"]',
      title: 'Nouvelle reservation',
      description:
        'Demarrez le scenario en ajoutant une course de demonstration.',
      actionLabel: "Action attendue: cliquer sur 'Nouvelle reservation'.",
      requireClick: true,
    },
    {
      id: 'booking-left-panel',
      selector: '[data-tour-id="booking-left-panel"]',
      title: 'Etape 4 - Complétez la partie gauche',
      description:
        'Renseignez librement les champs de la colonne gauche : client, adresses, date/heure, options et montant.',
      actionLabel: 'Action attendue : complétez les informations, puis cliquez sur « Suivant ».',
    },
    {
      id: 'booking-medical-section',
      selector: '[data-tour-id="booking-medical-section"]',
      title: 'Etape 5 - Informations médicales',
      description:
        'Cette section permet d ajouter des précisions utiles : établissement, service, médecin, notes et accès.',
      actionLabel: 'Action attendue : complétez uniquement les champs nécessaires, puis continuez.',
    },
    {
      id: 'booking-submit',
      selector: '[data-tour-id="booking-submit"]',
      title: 'Etape 6 - Validez la réservation',
      description: 'Cliquez ici pour enregistrer la demande et continuer le guide.',
      requireClick: true,
      autoAdvanceOnClick: true,
      actionLabel: 'Action attendue: cliquer sur "Creer la reservation"',
    },
    {
      id: 'dispatch-followup',
      selector: '[data-tour-id="dispatch-followup"]',
      title: 'Etape 7 - Suivi opérationnel',
      description:
        'Ici tu filtres, priorises et assignes les courses en temps reel.',
    },
    {
      id: 'tab-assigned',
      selector: '[data-tour-id="tab-assigned"]',
      title: 'Etape 8 - Onglet Assignation chauffeur',
      description:
        'Centralise les courses deja attribuees pour suivre execution et ajustements.',
      requireClick: true,
      autoAdvanceOnClick: true,
      actionLabel: "Action attendue: cliquer sur l'onglet Assignation chauffeur.",
    },
    {
      id: 'assigned-assign-action',
      selector: '[data-tour-id="assigned-assign-action"]',
      title: 'Etape 9 - Assignez un chauffeur',
      description:
        "Dans ce tableau, utilisez l action d'assignation pour attribuer un chauffeur a la course.",
      requireClick: true,
      autoAdvanceOnClick: true,
      actionLabel: "Action attendue: cliquer sur 'Assigner un chauffeur'.",
    },
    {
      id: 'assign-driver-option',
      selector: '[data-tour-id="assign-driver-modal"]',
      title: 'Etape 10 - Sélectionnez le chauffeur',
      description:
        'Dernière étape : dans cette fenêtre, choisissez un chauffeur puis validez l attribution.',
      requireClick: true,
      autoAdvanceOnClick: true,
      actionLabel: "Action attendue : cliquez sur le bouton « Assigner ».",
    },
    {
      id: 'sidebar-dispatch',
      selector: '[data-tour-id="sidebar-facturation-link"]',
      title: 'Section Facturation',
      description:
        'La démonstration continue ici. Ouvrez la section Facturation pour poursuivre le parcours.',
      requireClick: true,
      autoAdvanceOnClick: true,
      actionLabel: "Action attendue : cliquez sur « Facturation ».",
    },
  ],
  'dispatch-mini': [
    {
      id: 'dispatch-mini-summary',
      selector: '[data-tour-id="dispatch-demo-summary"]',
      title: "Étape 1 - Vue d'ensemble du jour",
      description:
        "Voici l'activité du jour : transports planifiés, courses à assigner et chauffeurs disponibles.",
    },
    {
      id: 'dispatch-mini-table',
      selector: '[data-tour-id="dispatch-table"]',
      title: 'Étape 2 - Tableau des transports',
      description:
        'Chaque ligne représente un transport avec client, horaire, trajet, chauffeur et statut.',
    },
    {
      id: 'dispatch-mini-row-click',
      selector: '[data-tour-id="dispatch-row-clickable"]',
      title: 'Étape 3 - Ouvrir le détail',
      description:
        'Cliquez sur la première ligne du tableau pour ouvrir le panneau de détail à droite.',
      actionLabel: 'Action attendue : cliquez sur la première ligne.',
      requireClick: true,
      autoAdvanceOnClick: true,
    },
    {
      id: 'dispatch-mini-detail-panel',
      selector: '[data-tour-id="ReservationDetailPanel_panel"]',
      title: "Étape 4 - Détails d'une réservation",
      description:
        'Ce panneau centralise la course : informations client, trajet, facturation et historique pour un suivi complet.',
      waitForElement: true,
      waitTimeoutMs: DISPATCH_DETAIL_PANEL_TIMEOUT_MS,
    },
  ],
  institution: [
    {
      id: 'institution-welcome',
      title: 'Bienvenue dans votre démo institution',
      description:
        "En quelques étapes, vous allez découvrir l'espace institution LIRIE : créer une demande, suivre son traitement et retrouver l'historique.",
      actionLabel: "Action attendue : cliquez sur « Suivant » pour commencer.",
    },
    {
      id: 'institution-kpi-grid',
      selector: '[data-tour-id="institution-kpi-grid"]',
      title: "Lire les indicateurs du tableau de bord",
      description:
        "En haut, on retrouve les indicateurs essentiels : le nombre total de demandes, les transports en cours, les demandes en attente, et les transports terminés. Cela permet d'avoir une vision immédiate de l'activité sans devoir ouvrir chaque dossier.",
    },
    {
      id: 'institution-kpi-pending',
      selector: '[data-tour-id="institution-kpi-pending"]',
      title: "Alerte sur les demandes en attente",
      description:
        "LIRIE attire l'attention sur les demandes en attente depuis un certain temps. L'institution peut donc repérer rapidement un dossier qui nécessite une action ou une relance.",
    },
    {
      id: 'institution-create-cta',
      selector: '[data-tour-id="institution-create-request-cta"]',
      title: "Mission 1 : Créer une demande",
      description:
        "La première action centrale pour une institution, c'est la création d'une demande de transport. Cliquez sur « Nouvelle demande » pour ouvrir le formulaire.",
      actionLabel: "Action attendue : cliquez sur « Nouvelle demande ».",
      requireClick: true,
      autoAdvanceOnClick: true,
    },
    {
      id: 'institution-request-form',
      selector: '[data-tour-id="institution-request-form-left"]',
      tooltipTargetSelector: '[data-tour-id="institution-request-form-tooltip"]',
      title: "Complétez la demande",
      description:
        "Le formulaire est ouvert. Complétez les champs essentiels : sélectionnez un patient, saisissez la destination, puis la date et l'heure.",
      actionLabel: "Action attendue : complétez le formulaire puis cliquez sur « Suivant ».",
      waitForElement: true,
      waitTimeoutMs: 4000,
      allowInteractions: true,
    },
    {
      id: 'institution-request-details',
      selector: '[data-tour-id="institution-request-form-tooltip"]',
      title: "Complétez les détails de la demande",
      description:
        "Vous pouvez maintenant compléter les informations de départ, d'arrivée et les détails patient/contact dans ce panneau.",
      actionLabel: "Action attendue : complétez les champs utiles, puis cliquez sur « Suivant ».",
      allowInteractions: true,
    },
    {
      id: 'institution-request-submit',
      selector: '[data-tour-id="institution-request-submit"]',
      title: "Enregistrez la demande",
      description:
        "Une fois les champs essentiels remplis, cliquez sur « Enregistrer » pour créer la demande.",
      actionLabel: "Action attendue : cliquez sur « Enregistrer ».",
      requireClick: true,
      autoAdvanceOnClick: true,
    },
    {
      id: 'institution-requests-select-draft',
      selector: '[data-tour-id="institution-request-draft-card"]',
      title: "Mission 2 : Ouvrir la demande brouillon",
      description:
        "Sur la page Demandes, cliquez sur la carte en brouillon pour ouvrir son détail et préparer l'envoi au transporteur.",
      actionLabel: "Action attendue : cliquez sur la demande en statut « Brouillon ».",
      requireClick: true,
      autoAdvanceOnClick: true,
      waitForElement: true,
      waitTimeoutMs: 5000,
    },
    {
      id: 'institution-requests-detail',
      selector: '[data-tour-id="institution-request-detail-panel"]',
      title: 'Comprendre le panneau de détail',
      description:
        "Ce panneau centralise les actions et le suivi : bouton d'envoi, informations de trajet, détails, facturation et historique.",
      waitForElement: true,
      waitTimeoutMs: 3000,
    },
    {
      id: 'institution-requests-send',
      selector: '[data-tour-id="institution-request-send-btn"]',
      title: 'Envoyer la demande',
      description:
        "Quand tout est prêt, envoyez la demande au transporteur depuis ce bouton.",
      actionLabel: "Action attendue : cliquez sur « Envoyer ».",
      requireClick: true,
      autoAdvanceOnClick: true,
    },
    {
      id: 'institution-requests-send-confirm',
      selector: '[data-tour-id="institution-request-send-confirm-btn"]',
      title: "Confirmer l'envoi",
      description:
        "La demande sera transmise aux transporteurs disponibles. Validez ici pour lancer l'envoi.",
      actionLabel: "Action attendue : cliquez sur « Envoyer » dans la fenêtre de confirmation.",
      requireClick: true,
      autoAdvanceOnClick: true,
      waitForElement: true,
      waitTimeoutMs: 4000,
    },
    {
      id: 'institution-recent-status',
      selector: '[data-tour-id="institution-request-detail-panel"]',
      title: "Simulation de prise en charge",
      description:
        "Après envoi, la simulation démo fait évoluer la course automatiquement : Envoyée → Acceptée → Course créée → Assignée → En route. Un message transporteur s'affiche dans la discussion à l'étape « En route ».",
    },
    {
      id: 'institution-history',
      selector: '[data-tour-id="institution-history"]',
      title: "Mission 3 : Retrouver l'historique",
      description:
        "La troisième dimension importante, c'est la traçabilité. Les demandes récentes permettent de retrouver rapidement les derniers trajets, avec le patient, l'heure, le point de départ, la destination et le statut.",
    },
    {
      id: 'institution-sidebar',
      selector: '[data-tour-id="institution-sidebar"]',
      title: "Navigation latérale",
      description:
        "La navigation latérale structure l'usage : le tableau de bord pour la vue globale, les demandes pour le suivi opérationnel, les patients pour retrouver les informations centralisées, puis les paramètres pour l'administration de l'espace.",
    },
    {
      id: 'institution-conclusion',
      title: "Conclusion de démo",
      description:
        "En résumé, pour une institution, LIRIE permet de centraliser trois étapes essentielles : créer une demande, suivre son traitement, et retrouver facilement l'historique. L'objectif est de gagner du temps, de mieux coordonner les transports et d'avoir une vision claire de l'activité.",
      actionLabel: "Cliquez sur « Terminer » pour conclure la démonstration.",
    },
  ],
  'invoices-mini': [
    {
      id: 'invoice-registry',
      selector: '[data-tour-id="invoice-registry"]',
      title: 'Étape 1 - Registre des factures',
      description:
        'Voici le registre de vos factures. Vous suivez ici les émissions, paiements et soldes restants.',
    },
    {
      id: 'invoice-stats',
      selector: '[data-tour-id="invoice-stats"]',
      title: 'Étape 2 - Résumé financier',
      description:
        'Ces indicateurs vous donnent une vision immédiate de votre situation financière.',
    },
    {
      id: 'invoice-command-bar',
      selector: '[data-tour-id="invoice-command-bar"]',
      title: 'Étape 3 - Recherche et filtres',
      description:
        'Utilisez la recherche et les filtres pour retrouver rapidement la bonne facture.',
    },
    {
      id: 'invoice-table',
      selector: '[data-tour-id="invoice-table"]',
      title: 'Étape 4 - Liste des factures',
      description:
        'Chaque ligne affiche les informations clés : client, échéance, montant, paiement et statut.',
    },
    {
      id: 'invoice-row-actions',
      selector: '[data-tour-id="invoice-row-actions"]',
      title: 'Étape 5 - Actions facture',
      description:
        'Depuis les actions, vous pouvez ouvrir, envoyer ou suivre une facture.',
    },
    {
      id: 'invoice-open-modal',
      selector: '[data-tour-id="invoice-new-button"]',
      title: 'Étape 6 - Nouvelle facture',
      description:
        'Vous pouvez créer une facture directement depuis ici.',
      actionLabel: "Action attendue : cliquez sur « Nouvelle facture ».",
      requireClick: true,
      autoAdvanceOnClick: true,
    },
    {
      id: 'invoice-modal-billing-type',
      selector: '[data-tour-id="invoice-modal-billing-type"]',
      title: 'Étape 7 - Type de facturation',
      description:
        "Choisissez d'abord le type de facturation.",
      waitForElement: true,
      waitTimeoutMs: INVOICE_MODAL_TARGET_TIMEOUT_MS,
      finishOnTimeout: true,
    },
    {
      id: 'invoice-modal-client',
      selector: '[data-tour-id="invoice-modal-client"]',
      title: 'Étape 8 - Sélection client',
      description:
        'Sélectionnez ensuite le client à facturer.',
    },
    {
      id: 'invoice-modal-period',
      selector: '[data-tour-id="invoice-modal-period"]',
      title: 'Étape 9 - Période',
      description:
        'Choisissez la période à facturer.',
    },
    {
      id: 'invoice-modal-submit',
      selector: '[data-tour-id="invoice-modal-submit"]',
      title: 'Étape 10 - Génération',
      description:
        'Validez pour générer la facture correspondante.',
    },
  ],
};

const GUIDE_PHASES = Object.freeze({
  IDLE: 'IDLE',
  WAITING_TARGET: 'WAITING_TARGET',
  TARGET_VISIBLE: 'TARGET_VISIBLE',
  AWAITING_CLICK: 'AWAITING_CLICK',
  ADVANCING: 'ADVANCING',
  COMPLETED: 'COMPLETED',
});

function nextGuidePhase(currentPhase, event, context = {}) {
  switch (event) {
    case 'OPEN':
      return GUIDE_PHASES.WAITING_TARGET;
    case 'CLOSE':
      return GUIDE_PHASES.COMPLETED;
    case 'PENDING_STEP':
      return GUIDE_PHASES.ADVANCING;
    case 'TARGET_MISSING':
      return currentPhase === GUIDE_PHASES.ADVANCING
        ? GUIDE_PHASES.ADVANCING
        : GUIDE_PHASES.WAITING_TARGET;
    case 'TARGET_FOUND':
      return context.requireClick && !context.hasClicked
        ? GUIDE_PHASES.AWAITING_CLICK
        : GUIDE_PHASES.TARGET_VISIBLE;
    case 'TARGET_CLICKED':
      return GUIDE_PHASES.TARGET_VISIBLE;
    case 'STEP_RESET':
      return GUIDE_PHASES.WAITING_TARGET;
    case 'FINISH':
      return GUIDE_PHASES.COMPLETED;
    default:
      return currentPhase;
  }
}

function getTooltipPosition(targetRect, options = {}) {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const cardWidth = TOOLTIP_WIDTH;
  const cardHeight = TOOLTIP_HEIGHT;
  const margin = 12;
  const pinToContainer = Boolean(options.pinToContainer);
  const containerRect = options.containerRect || null;
  const forcePlacement = options.forcePlacement || null;

  if (pinToContainer) {
    const sideGap = 6;
    const verticalGap = 8;
    const boundsLeft = containerRect ? Math.max(margin, containerRect.left + 8) : margin;
    const boundsRight = containerRect
      ? Math.min(viewportWidth - margin, containerRect.right - 8)
      : viewportWidth - margin;
    const boundsTop = containerRect ? Math.max(margin, containerRect.top + 8) : margin;
    const boundsBottom = containerRect
      ? Math.min(viewportHeight - margin, containerRect.bottom - 8)
      : viewportHeight - margin;

    const canPlaceRight = targetRect.right + sideGap + cardWidth <= boundsRight;
    const canPlaceLeft = targetRect.left - sideGap - cardWidth >= boundsLeft;

    let left = boundsLeft;
    let top;
    const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;

    if (forcePlacement === 'left' && canPlaceLeft) {
      left = targetRect.left - cardWidth - sideGap;
      top = Math.min(Math.max(boundsTop, centeredTop), Math.max(boundsTop, boundsBottom - cardHeight));
      return { top, left, placement: 'left' };
    }
    if (forcePlacement === 'right' && canPlaceRight) {
      left = targetRect.right + sideGap;
      top = Math.min(Math.max(boundsTop, centeredTop), Math.max(boundsTop, boundsBottom - cardHeight));
      return { top, left, placement: 'right' };
    }

    if (canPlaceRight) {
      left = targetRect.right + sideGap;
    } else if (canPlaceLeft) {
      left = targetRect.left - cardWidth - sideGap;
    } else {
      const centeredLeft = targetRect.left + targetRect.width / 2 - cardWidth / 2;
      left = Math.min(Math.max(boundsLeft, centeredLeft), Math.max(boundsLeft, boundsRight - cardWidth));
    }

    top = Math.min(Math.max(boundsTop, centeredTop), Math.max(boundsTop, boundsBottom - cardHeight));

    if (!canPlaceLeft && !canPlaceRight) {
      const canPlaceBelow = targetRect.bottom + verticalGap + cardHeight <= boundsBottom;
      const canPlaceAbove = targetRect.top - verticalGap - cardHeight >= boundsTop;
      if (canPlaceBelow) {
        top = targetRect.bottom + verticalGap;
      } else if (canPlaceAbove) {
        top = targetRect.top - cardHeight - verticalGap;
      }
    }

    return {
      top,
      left,
      placement: 'inside',
    };
  }

  const preferredLeft = targetRect.left;
  const sideGap = Number.isFinite(options.sideGap) ? options.sideGap : 10;
  const verticalGap = Number.isFinite(options.verticalGap) ? options.verticalGap : 12;
  const forceVertical = Boolean(options.forceVertical);
  const rightFallbackBottom = Boolean(options.rightFallbackBottom);
  const canPlaceRight = targetRect.right + sideGap + cardWidth <= viewportWidth - margin;
  const canPlaceLeft = targetRect.left - sideGap - cardWidth >= margin;
  const canPlaceBelow = targetRect.bottom + verticalGap + cardHeight <= viewportHeight - margin;
  const canPlaceAbove = targetRect.top - verticalGap - cardHeight >= margin;

  const centeredLeft = targetRect.left + targetRect.width / 2 - cardWidth / 2;
  const clampedLeft = Math.min(
    Math.max(margin, centeredLeft || preferredLeft),
    Math.max(margin, viewportWidth - cardWidth - margin)
  );

  if (forcePlacement === 'top' && canPlaceAbove) {
    return {
      top: targetRect.top - cardHeight - verticalGap,
      left: clampedLeft,
      placement: 'top',
    };
  }
  if (forcePlacement === 'bottom' && canPlaceBelow) {
    return {
      top: targetRect.bottom + verticalGap,
      left: clampedLeft,
      placement: 'bottom',
    };
  }
  if (forcePlacement === 'right' && canPlaceRight) {
    const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;
    return {
      top: Math.min(
        Math.max(margin, centeredTop),
        Math.max(margin, viewportHeight - cardHeight - margin)
      ),
      left: targetRect.right + sideGap,
      placement: 'right',
    };
  }
  if (forcePlacement === 'right' && rightFallbackBottom && canPlaceBelow) {
    return {
      top: targetRect.bottom + verticalGap,
      left: clampedLeft,
      placement: 'bottom',
    };
  }
  if (forcePlacement === 'right' && canPlaceLeft) {
    const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;
    return {
      top: Math.min(
        Math.max(margin, centeredTop),
        Math.max(margin, viewportHeight - cardHeight - margin)
      ),
      left: targetRect.left - cardWidth - sideGap,
      placement: 'left',
    };
  }
  if (forcePlacement === 'left' && canPlaceLeft) {
    const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;
    return {
      top: Math.min(
        Math.max(margin, centeredTop),
        Math.max(margin, viewportHeight - cardHeight - margin)
      ),
      left: targetRect.left - cardWidth - sideGap,
      placement: 'left',
    };
  }
  if (forcePlacement === 'left' && canPlaceRight) {
    const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;
    return {
      top: Math.min(
        Math.max(margin, centeredTop),
        Math.max(margin, viewportHeight - cardHeight - margin)
      ),
      left: targetRect.right + sideGap,
      placement: 'right',
    };
  }

  // Petites cibles d'action (ex: boutons "Accepter"): privilégier droite/gauche.
  if (targetRect.width <= 84 && !forceVertical && !forcePlacement) {
    if (canPlaceRight) {
      const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;
      return {
        top: Math.min(
          Math.max(margin, centeredTop),
          Math.max(margin, viewportHeight - cardHeight - margin)
        ),
        left: targetRect.right + sideGap,
        placement: 'right',
      };
    }
    if (canPlaceLeft) {
      const centeredTop = targetRect.top + targetRect.height / 2 - cardHeight / 2;
      return {
        top: Math.min(
          Math.max(margin, centeredTop),
          Math.max(margin, viewportHeight - cardHeight - margin)
        ),
        left: targetRect.left - cardWidth - sideGap,
        placement: 'left',
      };
    }
  }

  if (canPlaceBelow) {
    return {
      top: targetRect.bottom + verticalGap,
      left: clampedLeft,
      placement: 'bottom',
    };
  }

  if (canPlaceAbove) {
    return {
      top: targetRect.top - cardHeight - verticalGap,
      left: clampedLeft,
      placement: 'top',
    };
  }

  // Fallback défensif si l'espace est contraint.
  return {
    top: Math.min(
      Math.max(margin, targetRect.top + targetRect.height / 2 - cardHeight / 2),
      Math.max(margin, viewportHeight - cardHeight - margin)
    ),
    left: clampedLeft,
    placement: 'floating',
  };
}

function toStableRect(domRect) {
  return {
    top: domRect.top,
    left: domRect.left,
    right: domRect.right,
    bottom: domRect.bottom,
    width: domRect.width,
    height: domRect.height,
  };
}

function rectDidChange(prevRect, nextRect) {
  if (!prevRect) return true;
  return (
    Math.abs(prevRect.top - nextRect.top) > RECT_EPSILON ||
    Math.abs(prevRect.left - nextRect.left) > RECT_EPSILON ||
    Math.abs(prevRect.width - nextRect.width) > RECT_EPSILON ||
    Math.abs(prevRect.height - nextRect.height) > RECT_EPSILON
  );
}

function waitForElement(selector, onFound, onTimeout, timeoutMs = 3000) {
  let done = false;
  const complete = (callback) => {
    if (done) return;
    done = true;
    callback?.();
  };

  const tryFind = () => {
    const target = document.querySelector(selector);
    if (!target) return false;
    complete(() => onFound(target));
    return true;
  };

  if (tryFind()) return () => {};

  const timeoutId = window.setTimeout(() => {
    complete(onTimeout);
  }, timeoutMs);
  const observer = new MutationObserver(() => {
    if (tryFind()) {
      window.clearTimeout(timeoutId);
      observer.disconnect();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });

  return () => {
    window.clearTimeout(timeoutId);
    observer.disconnect();
  };
}

function isBookingClientSelected() {
  const container = document.querySelector('[data-tour-id="booking-client"]');
  if (!container) return false;
  const singleValueNode = container.querySelector('.react-select__single-value');
  return Boolean(String(singleValueNode?.textContent || '').trim());
}

function isClientMenuOpen() {
  const menu = document.querySelector('.react-select__menu');
  return Boolean(menu);
}

function getBookingClientTargetElement() {
  const container = document.querySelector('[data-tour-id="booking-client"]');
  if (!container) return null;

  // Le menu peut etre rendu dans un portal (document.body), pas forcement dans `container`.
  const input = container.querySelector('#client-select');
  const listboxId = input?.getAttribute('aria-controls');
  const listbox =
    (listboxId && document.getElementById(listboxId)) ||
    document.querySelector('.react-select__menu');
  const firstOption = listbox?.querySelector('.react-select__option');
  if (firstOption) return firstOption;
  if (listbox) return listbox;
  return container.querySelector('.react-select__control');
}

function normalizeGuideText(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

function isDropoffHugQueryEntered() {
  const dropoffInput = document.querySelector('#dropoff_location');
  const normalized = normalizeGuideText(dropoffInput?.value || '');
  return normalized.includes('hug') && normalized.includes('geneve');
}

function resolveStepTargetElement(step) {
  if (!step?.selector) return null;
  if (step.id === 'booking-client') {
    return getBookingClientTargetElement();
  }
  if (step.id === 'booking-dropoff') {
    const dropoffInput = document.querySelector('#dropoff_location');
    if (!isDropoffHugQueryEntered()) {
      return dropoffInput || document.querySelector(step.selector);
    }
    const listboxId = dropoffInput?.getAttribute('aria-controls') || 'dropoff_location-ac-listbox';
    const listbox = listboxId ? document.getElementById(listboxId) : null;
    if (listbox) {
      const options = Array.from(listbox.querySelectorAll('[role="option"]'));
      const hugOption = options.find((option) => {
        const normalized = normalizeGuideText(option.textContent || '');
        return (
          normalized.includes('hopitaux universitaires de geneve') ||
          (normalized.includes('hug') && normalized.includes('gabrielle-perret-gentil')) ||
          (normalized.includes('hug') && normalized.includes('1205'))
        );
      });
      return hugOption || options[0] || listbox;
    }
    return dropoffInput || document.querySelector(step.selector);
  }
  return document.querySelector(step.selector);
}

function isBookingPickupCompleted() {
  const pickupInput = document.querySelector('#pickup_location');
  const pickupValue = String(pickupInput?.value || '').trim();
  if (pickupValue) return true;
  // En demo, le depart est pre-rempli avec le domicile du client selectionne.
  // On accepte donc l'etape si le client est deja choisi.
  return isBookingClientSelected();
}

function isBookingDropoffCompleted() {
  const dropoffInput = document.querySelector('#dropoff_location');
  const dropoffValue = String(dropoffInput?.value || '').trim();
  if (!dropoffValue) return false;

  const normalizedDropoff = normalizeGuideText(dropoffValue);
  const isHugDestination =
    normalizedDropoff.includes('hopitaux universitaires de geneve') ||
    normalizedDropoff.includes('(hug)') ||
    normalizedDropoff.includes(' hug');

  return isHugDestination;
}

function isRoundTripEnabled() {
  const toggle = document.querySelector('[data-tour-id="booking-roundtrip-toggle"]');
  if (!toggle) return false;
  return String(toggle.getAttribute('aria-pressed') || '').toLowerCase() === 'true';
}

function isReturnDateConfigured() {
  const container = document.querySelector('[data-tour-id="booking-return-config"]');
  if (!container) return false;
  const dateInput = container.querySelector('input');
  return Boolean(String(dateInput?.value || '').trim());
}

function getStepWaitTimeoutMs(step) {
  if (!step?.id) return WAIT_FOR_TARGET_TIMEOUT_MS;
  if (Number.isFinite(step.waitTimeoutMs)) return step.waitTimeoutMs;
  return step.id.startsWith('assign-driver-')
    ? ASSIGN_MODAL_TARGET_TIMEOUT_MS
    : step.id.startsWith('invoice-modal-')
      ? INVOICE_MODAL_TARGET_TIMEOUT_MS
    : WAIT_FOR_TARGET_TIMEOUT_MS;
}

function isBookingAmountCompleted() {
  const amountInput = document.querySelector('#amount');
  if (!amountInput) return false;
  if (amountInput.hasAttribute('disabled')) return true;
  const value = Number(String(amountInput.value || '').trim());
  return Number.isFinite(value) && value > 0;
}

function isDepartureDateValid() {
  const dateInput = document.querySelector(
    '[data-tour-id="booking-datetime"] [class*="InlineDatePicker_field__"] input'
  );
  if (!dateInput) return false;
  const dateValue = String(dateInput.value || '').trim();
  if (!dateValue) return false;

  const match = dateValue.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (!match) return false;
  const [, dd, mm, yyyy] = match;
  const day = Number(dd);
  const month = Number(mm);
  const year = Number(yyyy);
  const parsed = new Date(year, month - 1, day);
  if (
    Number.isNaN(parsed.getTime()) ||
    parsed.getFullYear() !== year ||
    parsed.getMonth() !== month - 1 ||
    parsed.getDate() !== day
  ) {
    return false;
  }

  const today = new Date();
  const todayStart = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  return parsed >= todayStart;
}

function isDepartureTimeCompleted() {
  const timeInput = document.querySelector(
    '[data-tour-id="booking-datetime"] [class*="InlineTimePicker_field__"] input'
  );
  if (!timeInput) return false;
  const timeValue = String(timeInput.value || '').trim();
  return timeValue.length > 0;
}

const DemoInteractiveGuide = ({ role = 'transporteur', onFinish, userFirstName, initialStepId = null }) => {
  const steps = useMemo(() => GUIDE_STEPS[role] || GUIDE_STEPS.transporteur, [role]);
  const [active, setActive] = useState(true);
  const [showFreeModeNotice, setShowFreeModeNotice] = useState(false);
  const [guidePhase, setGuidePhase] = useState(GUIDE_PHASES.IDLE);
  const [stepIndex, setStepIndex] = useState(() => {
    if (!initialStepId) return 0;
    const idx = steps.findIndex((step) => step.id === initialStepId);
    return idx >= 0 ? idx : 0;
  });
  const [targetRect, setTargetRect] = useState(null);
  const [tooltipTargetRect, setTooltipTargetRect] = useState(null);
  const [modalRect, setModalRect] = useState(null);
  const [hasClickedTarget, setHasClickedTarget] = useState(false);
  const [pendingStepId, setPendingStepId] = useState(null);
  const rafRef = useRef(null);
  const tooltipRef = useRef(null);
  const delayedAdvanceRef = useRef(null);

  const currentStep = steps[stepIndex] || null;
  const currentSelector = currentStep?.selector || null;
  const isModalStep = currentStep ? MODAL_STEP_IDS.has(currentStep.id) : false;
  const currentModalSelector = useMemo(() => {
    if (!currentStep?.id) return null;
    if (currentStep.id.startsWith('assign-driver-')) return ASSIGN_DRIVER_MODAL_SELECTOR;
    if (currentStep.id.startsWith('invoice-modal-')) return INVOICE_MODAL_SELECTOR;
    if (currentStep.id.startsWith('institution-request-')) return '[data-tour-id="institution-request-create"]';
    if (MODAL_STEP_IDS.has(currentStep.id)) return MODAL_FORM_SELECTOR;
    return null;
  }, [currentStep]);

  const handleCloseToFreeMode = () => {
    setActive(false);
    setShowFreeModeNotice(true);
  };

  const finishGuide = useCallback(() => {
    setGuidePhase((prev) => nextGuidePhase(prev, 'FINISH'));
    setShowFreeModeNotice(false);
    setActive(false);
    if (role === 'institution') {
      try {
        window.sessionStorage.removeItem(INSTITUTION_RESUME_STEP_KEY);
        window.sessionStorage.setItem(DEMO_INSTITUTION_COMPLETED_KEY, '1');
      } catch {
        // ignore
      }
    }
    if (typeof onFinish === 'function') onFinish();
  }, [onFinish, role]);

  useEffect(() => {
    if (!active || !currentStep) {
      setGuidePhase((prev) => nextGuidePhase(prev, 'CLOSE'));
      return;
    }
    setGuidePhase((prev) => nextGuidePhase(prev, 'OPEN'));
  }, [active, currentStep]);

  useEffect(() => {
    if (!active || !currentSelector) {
      setTargetRect(null);
      setModalRect(null);
      return undefined;
    }

    const resizeObserver = window.ResizeObserver ? new ResizeObserver(() => scheduleUpdate()) : null;
    let mutationObserver = null;
    let cancelled = false;
    let observedTarget = null;
    let observedModal = null;

    const updateTargetRect = () => {
      if (cancelled) return;
      const target = resolveStepTargetElement(currentStep);
      if (target) {
        const nextRect = toStableRect(target.getBoundingClientRect());
        setTargetRect((prevRect) => (rectDidChange(prevRect, nextRect) ? nextRect : prevRect));
        setGuidePhase((prev) =>
          nextGuidePhase(prev, 'TARGET_FOUND', {
            requireClick: Boolean(currentStep?.requireClick),
            hasClicked: hasClickedTarget,
          })
        );
      } else {
        setTargetRect(null);
        setGuidePhase((prev) => nextGuidePhase(prev, 'TARGET_MISSING'));
      }

      const tooltipSel = currentStep?.tooltipTargetSelector;
      if (tooltipSel) {
        const tooltipEl = document.querySelector(tooltipSel);
        if (tooltipEl) {
          const nextTooltipRect = toStableRect(tooltipEl.getBoundingClientRect());
          setTooltipTargetRect((prev) => (rectDidChange(prev, nextTooltipRect) ? nextTooltipRect : prev));
        } else {
          setTooltipTargetRect(null);
        }
      } else {
        setTooltipTargetRect(null);
      }

      if (isModalStep && currentModalSelector) {
        const modal = document.querySelector(currentModalSelector);
        if (modal) {
          const nextModalRect = toStableRect(modal.getBoundingClientRect());
          setModalRect((prevRect) =>
            rectDidChange(prevRect, nextModalRect) ? nextModalRect : prevRect
          );
        } else {
          setModalRect(null);
        }
      } else {
        setModalRect(null);
      }

      if (resizeObserver) {
        if (target !== observedTarget) {
          if (observedTarget) resizeObserver.unobserve(observedTarget);
          observedTarget = target;
          if (observedTarget) resizeObserver.observe(observedTarget);
        }

        const modal =
          isModalStep && currentModalSelector
            ? document.querySelector(currentModalSelector)
            : null;
        if (modal !== observedModal) {
          if (observedModal) resizeObserver.unobserve(observedModal);
          observedModal = modal;
          if (observedModal) resizeObserver.observe(observedModal);
        }
      }
    };

    function scheduleUpdate() {
      if (rafRef.current) return;
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null;
        updateTargetRect();
      });
    }

    scheduleUpdate();
    window.addEventListener('resize', scheduleUpdate);
    window.addEventListener('scroll', scheduleUpdate, true);

    mutationObserver = new MutationObserver(scheduleUpdate);
    mutationObserver.observe(document.body, { childList: true, subtree: true });

    return () => {
      cancelled = true;
      window.removeEventListener('resize', scheduleUpdate);
      window.removeEventListener('scroll', scheduleUpdate, true);
      if (rafRef.current) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (resizeObserver) {
        if (observedTarget) resizeObserver.unobserve(observedTarget);
        if (observedModal) resizeObserver.unobserve(observedModal);
        resizeObserver.disconnect();
      }
      if (mutationObserver) mutationObserver.disconnect();
    };
  }, [active, currentSelector, currentStep, hasClickedTarget, isModalStep, currentModalSelector]);

  const effectiveTooltipRect =
    currentStep?.id !== 'institution-request-form' &&
    currentStep?.tooltipTargetSelector &&
    tooltipTargetRect
      ? tooltipTargetRect
      : targetRect;

  useEffect(() => {
    setHasClickedTarget(false);
    setGuidePhase((prev) => nextGuidePhase(prev, 'STEP_RESET'));
  }, [stepIndex]);

  useEffect(() => {
    if (!active || currentStep?.id !== 'booking-client') return undefined;

    let observer = null;
    let rafId = null;
    const checkClientSelection = () => {
      const selected = isBookingClientSelected();
      setHasClickedTarget(selected);
      if (!selected) return;

      const nextStep = steps[stepIndex + 1];
      if (nextStep?.id && pendingStepId !== nextStep.id) {
        setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
        setPendingStepId(nextStep.id);
      }
    };
    const scheduleCheck = () => {
      if (rafId) return;
      rafId = window.requestAnimationFrame(() => {
        rafId = null;
        checkClientSelection();
      });
    };

    scheduleCheck();
    observer = new MutationObserver(scheduleCheck);
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener('click', scheduleCheck, true);
    document.addEventListener('input', scheduleCheck, true);

    return () => {
      if (observer) observer.disconnect();
      document.removeEventListener('click', scheduleCheck, true);
      document.removeEventListener('input', scheduleCheck, true);
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [active, currentStep, pendingStepId, stepIndex, steps]);

  useEffect(() => {
    if (!active || currentStep?.id !== 'booking-pickup') return undefined;

    let observer = null;
    let rafId = null;
    const checkPickup = () => {
      const completed = isBookingPickupCompleted();
      setHasClickedTarget(completed);
      // Etape 10/25: rester sur cette etape et laisser l'utilisateur
      // avancer explicitement avec le bouton "Suivant".
    };
    const scheduleCheck = () => {
      if (rafId) return;
      rafId = window.requestAnimationFrame(() => {
        rafId = null;
        checkPickup();
      });
    };

    scheduleCheck();
    observer = new MutationObserver(scheduleCheck);
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener('click', scheduleCheck, true);
    document.addEventListener('input', scheduleCheck, true);
    document.addEventListener('change', scheduleCheck, true);

    return () => {
      if (observer) observer.disconnect();
      document.removeEventListener('click', scheduleCheck, true);
      document.removeEventListener('input', scheduleCheck, true);
      document.removeEventListener('change', scheduleCheck, true);
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [active, currentStep]);

  useEffect(() => {
    if (!active || currentStep?.id !== 'booking-dropoff') return undefined;

    let observer = null;
    let rafId = null;
    const checkDropoff = () => {
      const completed = isBookingDropoffCompleted();
      setHasClickedTarget(completed);
      if (!completed) return;

      const nextStep = steps[stepIndex + 1];
      if (nextStep?.id && pendingStepId !== nextStep.id) {
        setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
        setPendingStepId(nextStep.id);
      }
    };
    const scheduleCheck = () => {
      if (rafId) return;
      rafId = window.requestAnimationFrame(() => {
        rafId = null;
        checkDropoff();
      });
    };

    scheduleCheck();
    observer = new MutationObserver(scheduleCheck);
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener('click', scheduleCheck, true);
    document.addEventListener('input', scheduleCheck, true);
    document.addEventListener('change', scheduleCheck, true);

    return () => {
      if (observer) observer.disconnect();
      document.removeEventListener('click', scheduleCheck, true);
      document.removeEventListener('input', scheduleCheck, true);
      document.removeEventListener('change', scheduleCheck, true);
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [active, currentStep, pendingStepId, stepIndex, steps]);

  useEffect(() => {
    if (!active || currentStep?.id !== 'booking-roundtrip-toggle') return undefined;

    let observer = null;
    let rafId = null;
    const checkRoundTrip = () => {
      const enabled = isRoundTripEnabled();
      setHasClickedTarget(enabled);
      if (!enabled) return;

      const nextStep = steps[stepIndex + 1];
      if (nextStep?.id && pendingStepId !== nextStep.id) {
        setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
        setPendingStepId(nextStep.id);
      }
    };

    const scheduleCheck = () => {
      if (rafId) return;
      rafId = window.requestAnimationFrame(() => {
        rafId = null;
        checkRoundTrip();
      });
    };

    scheduleCheck();
    observer = new MutationObserver(scheduleCheck);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true });
    document.addEventListener('click', scheduleCheck, true);

    return () => {
      if (observer) observer.disconnect();
      document.removeEventListener('click', scheduleCheck, true);
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [active, currentStep, pendingStepId, stepIndex, steps]);

  useEffect(() => {
    if (!active || !currentStep?.requireClick) return undefined;
    const onClickCapture = (event) => {
      const target = resolveStepTargetElement(currentStep);
      if (!target) return;
      if (target.contains(event.target)) {
        const validatedBySelection =
          currentStep.id === 'booking-client'
            ? isBookingClientSelected()
            : currentStep.id === 'booking-pickup'
              ? isBookingPickupCompleted()
              : currentStep.id === 'booking-dropoff'
                ? isBookingDropoffCompleted()
                : currentStep.id === 'booking-amount'
                  ? isBookingAmountCompleted()
                : currentStep.id === 'booking-plus30'
                  ? isDepartureDateValid()
                : currentStep.id === 'booking-time'
                  ? isDepartureTimeCompleted()
                : currentStep.id === 'assign-driver-option'
                  ? event.target instanceof Element &&
                    Boolean(event.target.closest('[data-tour-id="assign-driver-confirm"]'))
                : currentStep.id === 'booking-roundtrip-toggle'
                  ? isRoundTripEnabled()
                  : currentStep.id === 'booking-return-config'
                    ? isReturnDateConfigured()
              : true;
        setHasClickedTarget(validatedBySelection);
        setGuidePhase((prev) => nextGuidePhase(prev, 'TARGET_CLICKED'));
        if (currentStep.id === 'create-booking') {
          setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
          setPendingStepId('booking-left-panel');
          return;
        }
        if (currentStep.id === 'institution-create-cta') {
          setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
          setPendingStepId('institution-request-form');
          return;
        }
        if (currentStep.id === 'assigned-assign-action') {
          // Transition immédiate vers la modal d'assignation chauffeur.
          setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
          setPendingStepId('assign-driver-option');
          return;
        }
        if (currentStep.id === 'sidebar-dispatch') {
          try {
            window.sessionStorage.setItem('demo_invoices_mini', '1');
          } catch {
            // ignore
          }
        }
        if (currentStep.id === 'invoice-open-modal') {
          const nextStep = steps[stepIndex + 1];
          if (nextStep?.id) {
            setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
            setPendingStepId(nextStep.id);
          } else {
            finishGuide();
          }
          return;
        }
        if (currentStep.id === 'institution-requests-send-confirm' && validatedBySelection) {
          // Après confirmation d'envoi: fermer le guide et laisser l'utilisateur en mode libre.
          finishGuide();
          return;
        }
        if (currentStep.autoAdvanceOnClick && validatedBySelection) {
          const nextStep = steps[stepIndex + 1];
          if (nextStep?.id) {
            if (currentStep.id === 'institution-request-submit') {
              try {
                window.sessionStorage.setItem(INSTITUTION_RESUME_STEP_KEY, nextStep.id);
              } catch {
                // ignore
              }
            }
            if (currentStep.id === 'assign-driver-option' && nextStep.id === 'sidebar-dispatch') {
              if (delayedAdvanceRef.current) {
                window.clearTimeout(delayedAdvanceRef.current);
              }
              try {
                window.sessionStorage.setItem('demo_dispatch_mini', '1');
              } catch {
                // ignore
              }
              // Fermer le guide apres attribution, puis reprendre automatiquement sur Dispatch apres 5s.
              setShowFreeModeNotice(false);
              setActive(false);
              delayedAdvanceRef.current = window.setTimeout(() => {
                delayedAdvanceRef.current = null;
                const dispatchIndex = steps.findIndex((step) => step.id === 'sidebar-dispatch');
                if (dispatchIndex >= 0) {
                  setStepIndex(dispatchIndex);
                  setActive(true);
                }
              }, 5000);
              return;
            }
            setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
            setPendingStepId(nextStep.id);
          } else {
            // Derniere etape: terminer le guide immediatement apres le clic valide.
            finishGuide();
          }
        }
      }
    };
    document.addEventListener('click', onClickCapture, true);
    return () => document.removeEventListener('click', onClickCapture, true);
  }, [active, currentStep, stepIndex, steps, finishGuide]);

  useEffect(() => {
    if (role === 'dispatch-mini') return undefined;
    if (!active || !currentStep) return undefined;

    const isAllowedAutocompleteTarget = (node) => {
      if (!(node instanceof Element)) return false;
      if (currentStep.id === 'booking-left-panel') {
        return Boolean(
          node.closest('[id^="react-select-"][id$="-listbox"]') ||
            node.closest('.react-select__menu') ||
            node.closest('[role="listbox"]') ||
            node.closest('[role="option"]') ||
            node.closest('[class*="InlineDatePicker_popover__"]') ||
            node.closest('[class*="InlineTimePicker_popover__"]')
        );
      }
      if (currentStep.id === 'booking-client') {
        return Boolean(
          node.closest('[id^="react-select-"][id$="-listbox"]') ||
            node.closest('.react-select__menu') ||
            node.closest('[role="option"]')
        );
      }
      if (currentStep.id === 'booking-dropoff') {
        const input = document.querySelector('#dropoff_location');
        const listboxId = input?.getAttribute('aria-controls');
        if (listboxId) {
          const listbox = document.getElementById(listboxId);
          if (listbox && listbox.contains(node)) return true;
        }
        return Boolean(node.closest('[role="listbox"]') || node.closest('[role="option"]'));
      }
      if (currentStep.id === 'institution-request-patient') {
        return Boolean(
          node.closest('[id^="react-select-"][id$="-listbox"]') ||
            node.closest('.react-select__menu') ||
            node.closest('[role="listbox"]') ||
            node.closest('[role="option"]')
        );
      }
      if (currentStep.id === 'institution-request-form') {
        const input = document.querySelector('#dropoff_location');
        const listboxId = input?.getAttribute('aria-controls');
        if (listboxId) {
          const listbox = document.getElementById(listboxId);
          if (listbox && listbox.contains(node)) return true;
        }
        return Boolean(
          node.closest('[id^="react-select-"][id$="-listbox"]') ||
            node.closest('.react-select__menu') ||
            node.closest('[role="listbox"]') ||
            node.closest('[role="option"]') ||
            node.closest('[class*="InlineDatePicker_popover__"]') ||
            node.closest('[class*="InlineTimePicker_popover__"]')
        );
      }
      if (currentStep.id === 'institution-request-destination') {
        const input = document.querySelector('#dropoff_location');
        const listboxId = input?.getAttribute('aria-controls');
        if (listboxId) {
          const listbox = document.getElementById(listboxId);
          if (listbox && listbox.contains(node)) return true;
        }
        return Boolean(node.closest('[role="listbox"]') || node.closest('[role="option"]'));
      }
      if (currentStep.id === 'institution-request-datetime') {
        return Boolean(
          node.closest('[class*="InlineDatePicker_popover__"]') ||
            node.closest('[class*="InlineTimePicker_popover__"]')
        );
      }
      return false;
    };

    const shouldAllowInteraction = (eventTarget) => {
      if (!(eventTarget instanceof Element)) return true;
      const target = resolveStepTargetElement(currentStep);
      if (target?.contains(eventTarget)) return true;
      if (tooltipRef.current?.contains(eventTarget)) return true;
      if (isAllowedAutocompleteTarget(eventTarget)) return true;
      return false;
    };

    const blockOutsideInteractions = (event) => {
      if (shouldAllowInteraction(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
      if (typeof event.stopImmediatePropagation === 'function') {
        event.stopImmediatePropagation();
      }
    };

    document.addEventListener('pointerdown', blockOutsideInteractions, true);
    document.addEventListener('click', blockOutsideInteractions, true);

    return () => {
      document.removeEventListener('pointerdown', blockOutsideInteractions, true);
      document.removeEventListener('click', blockOutsideInteractions, true);
    };
  }, [active, currentStep, role]);

  useEffect(() => {
    if (!active || !pendingStepId) return undefined;
    const pendingIndex = steps.findIndex((step) => step.id === pendingStepId);
    if (pendingIndex < 0) {
      setPendingStepId(null);
      return undefined;
    }

    const pendingStep = steps[pendingIndex];
    let chainedAdvanceTimer = null;
    const cleanupWait = waitForElement(
      pendingStep.selector,
      () => {
        setStepIndex(pendingIndex);
        if (pendingStep.id === 'manual-booking-form') {
          // Quand le modal s'ouvre, enchaîner directement vers le premier champ guide.
          chainedAdvanceTimer = window.setTimeout(() => {
            setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
            setPendingStepId('booking-left-panel');
          }, 100);
          return;
        }
        setPendingStepId(null);
      },
      () => {
        // Fallback defensif: les etapes "waitForElement" peuvent etre sautees proprement.
        if (pendingStep.waitForElement) {
          if (pendingStep.finishOnTimeout) {
            finishGuide();
            return;
          }
          const fallbackIndex = pendingIndex + 1;
          if (fallbackIndex >= steps.length) {
            finishGuide();
            return;
          }
          setStepIndex(fallbackIndex);
          setPendingStepId(null);
          return;
        }
        setStepIndex(pendingIndex);
        setPendingStepId(null);
      },
      getStepWaitTimeoutMs(pendingStep)
    );

    return () => {
      cleanupWait();
      if (chainedAdvanceTimer) window.clearTimeout(chainedAdvanceTimer);
    };
  }, [active, pendingStepId, steps, finishGuide]);

  useEffect(() => {
    const previous = document.querySelector('[data-demo-guide-active="true"]');
    if (previous) previous.removeAttribute('data-demo-guide-active');
    if (!active || !currentStep) return;
    const target = resolveStepTargetElement(currentStep);
    if (target) {
      target.setAttribute('data-demo-guide-active', 'true');
    }
  }, [active, currentStep]);

  useEffect(
    () => () => {
      const previous = document.querySelector('[data-demo-guide-active="true"]');
      if (previous) previous.removeAttribute('data-demo-guide-active');
      if (delayedAdvanceRef.current) {
        window.clearTimeout(delayedAdvanceRef.current);
      }
    },
    []
  );

  if (!currentStep) return null;

  const freeModeNotice = showFreeModeNotice ? (
    <div className={styles.freeModeBackdrop}>
      <div className={styles.freeModeToast} role="alertdialog" aria-live="assertive">
        <h4>Mode libre active</h4>
        <p>
          Vous avez ferme le guide et passe en mode libre. Pour une demonstration complete et plus
          claire, nous vous recommandons de reprendre et terminer les etapes guidees.
        </p>
        <div className={styles.freeModeActions}>
          <button
            type="button"
            className={styles.secondary}
            onClick={() => {
              if (role === 'institution') {
                try {
                  window.sessionStorage.removeItem(INSTITUTION_RESUME_STEP_KEY);
                  window.sessionStorage.setItem(DEMO_INSTITUTION_COMPLETED_KEY, '1');
                } catch {
                  // ignore
                }
              }
              setShowFreeModeNotice(false);
            }}
          >
            Continuer en mode libre
          </button>
          <button
            type="button"
            className={styles.primary}
            onClick={() => {
              setShowFreeModeNotice(false);
              setActive(true);
            }}
          >
            Reprendre le guide
          </button>
        </div>
      </div>
    </div>
  ) : null;

  if (!active) {
    if (!freeModeNotice) return null;
    if (typeof document === 'undefined' || !document.body) return freeModeNotice;
    return createPortal(freeModeNotice, document.body);
  }

  const isLast = stepIndex >= steps.length - 1;
  const createBookingStepIndex = steps.findIndex((step) => step.id === 'create-booking');
  const isWelcomeStep =
    currentStep.id === 'demo-welcome' || currentStep.id === 'institution-welcome';
  const isClientStep = currentStep.id === 'booking-client';
  const clientMenuOpen = isClientStep ? isClientMenuOpen() : false;
  const isPickupStep = currentStep.id === 'booking-pickup';
  const isDropoffStep = currentStep.id === 'booking-dropoff';
  const dropoffQueryEntered = isDropoffStep ? isDropoffHugQueryEntered() : false;
  const dropoffInput = isDropoffStep ? document.querySelector('#dropoff_location') : null;
  const dropoffListboxId =
    isDropoffStep && dropoffInput
      ? dropoffInput.getAttribute('aria-controls') || 'dropoff_location-ac-listbox'
      : null;
  const dropoffMenuOpen = Boolean(
    isDropoffStep && dropoffListboxId && document.getElementById(dropoffListboxId)
  );
  const isRoundTripStep = currentStep.id === 'booking-roundtrip-toggle';
  const isReturnConfigStep = currentStep.id === 'booking-return-config';
  const isAmountStep = currentStep.id === 'booking-amount';
  const isDepartureDateStep = currentStep.id === 'booking-plus30';
  const isDepartureTimeStep = currentStep.id === 'booking-time';
  const isCreateBookingStep = currentStep.id === 'create-booking';
  const canContinue =
    !currentStep.requireClick ||
    (isClientStep
      ? isBookingClientSelected()
      : isPickupStep
        ? isBookingPickupCompleted()
        : isDropoffStep
          ? isBookingDropoffCompleted()
          : isAmountStep
            ? isBookingAmountCompleted()
          : isDepartureDateStep
            ? isDepartureDateValid()
          : isDepartureTimeStep
            ? isDepartureTimeCompleted()
          : isRoundTripStep
            ? isRoundTripEnabled()
            : isReturnConfigStep
              ? isReturnDateConfigured()
        : hasClickedTarget);
  const pos = effectiveTooltipRect
    ? getTooltipPosition(effectiveTooltipRect, {
        // Etapes de reservation: rester ancre au contexte du modal.
        pinToContainer: isModalStep && currentStep.id !== 'assign-driver-option',
        containerRect: modalRect,
        verticalGap:
          currentStep.id === 'tab-pending'
            ? 36
            : currentStep.id === 'pending-row-overview'
              ? 32
            : currentStep.id === 'pending-accept-action'
              ? 26
            : currentStep.id === 'dispatch-mini-table'
              ? 22
            : currentStep.id === 'dispatch-mini-driver'
              ? 16
            : currentStep.id === 'tab-institutions' || currentStep.id === 'tab-assigned'
              ? 24
              : 12,
        forcePlacement:
          currentStep.id === 'institution-request-form'
            ? 'right'
            : currentStep.id === 'institution-requests-detail'
              ? 'left'
            : currentStep.id === 'institution-recent-status'
              ? 'left'
            : currentStep.id === 'pending-accept-action' ||
              currentStep.id === 'pending-row-overview' ||
              currentStep.id === 'assigned-assign-action'
              ? 'top'
              : currentStep.id === 'assign-driver-option'
                ? 'right'
              : currentStep.id === 'dispatch-mini-table'
                ? 'top'
              : currentStep.id === 'dispatch-mini-detail-panel'
                ? 'left'
              : currentStep.id === 'dispatch-mini-driver'
                ? 'right'
              : currentStep.id === 'invoice-table'
                ? 'left'
              : currentStep.id === 'invoice-row-actions'
                ? 'left'
              : currentStep.id === 'invoice-modal-submit'
                ? 'top'
              : undefined,
        forceVertical:
          currentStep.id === 'pending-accept-action' ||
          currentStep.id === 'pending-row-overview' ||
          currentStep.id === 'assigned-assign-action' ||
          currentStep.id === 'assign-driver-option' ||
          currentStep.id === 'dispatch-mini-table' ||
          currentStep.id === 'dispatch-mini-detail-panel' ||
          currentStep.id === 'invoice-table' ||
          currentStep.id === 'invoice-row-actions' ||
          currentStep.id === 'invoice-modal-submit',
        rightFallbackBottom:
          currentStep.id === 'dispatch-mini-driver' || currentStep.id === 'invoice-row-actions',
      })
    : { top: 24, left: 24, placement: 'bottom' };
  const spotlightStyle = targetRect
    ? {
        top: targetRect.top - 4,
        left: targetRect.left - 4,
        width: targetRect.width + 8,
        height: targetRect.height + 8,
      }
    : null;
  const tooltipStyle = isWelcomeStep
    ? { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }
    : { top: pos.top, left: pos.left };

  const allowInteractions = Boolean(currentStep?.allowInteractions);
  const guideContent = (
    <>
      <div
        className={`${styles.overlay} ${isWelcomeStep ? styles.overlayWelcome : ''} ${allowInteractions ? styles.overlayAllowInteractions : ''}`}
        data-tour-id={`demo-guide-${role}`}
      />
      {spotlightStyle && (
        <div
          className={`${styles.spotlight} ${allowInteractions ? styles.spotlightAllowInteractions : ''}`}
          style={spotlightStyle}
        />
      )}
      <div
        className={`${styles.tooltip} ${allowInteractions ? styles.tooltipAllowInteractions : ''}`}
        style={tooltipStyle}
        data-guide-phase={guidePhase}
        ref={tooltipRef}
      >
        <div className={styles.header}>
          <strong>{isWelcomeStep ? 'Guide de démonstration' : `Etape ${stepIndex + 1}/${steps.length}`}</strong>
          <button type="button" className={styles.closeBtn} onClick={handleCloseToFreeMode}>
            Fermer
          </button>
        </div>
        <h3>
          {currentStep.id === 'institution-welcome' && userFirstName
            ? `Bienvenue ${userFirstName}, dans votre démo institution`
            : currentStep.title}
        </h3>
        <p>{currentStep.description}</p>
        {currentStep.actionLabel && (
          <p className={styles.actionLabel}>
            {isClientStep && clientMenuOpen
              ? 'Action attendue: selectionner une ligne client dans la liste.'
              : isDropoffStep && dropoffMenuOpen
                ? "Action attendue: selectionner la suggestion HUG."
                : isDropoffStep && !dropoffQueryEntered
                  ? "Action attendue: saisir 'hug geneve' dans le champ destination."
              : currentStep.actionLabel}
          </p>
        )}
        {currentStep.requireClick &&
          !canContinue &&
          !isCreateBookingStep &&
          !isClientStep &&
          (isPickupStep ||
            isDropoffStep ||
            isDepartureDateStep ||
            isDepartureTimeStep ||
            isRoundTripStep ||
            isReturnConfigStep) &&
          !(isDropoffStep && (!dropoffQueryEntered || dropoffMenuOpen)) && (
          <p className={styles.hint}>
            {isClientStep
              ? clientMenuOpen
                ? 'Selectionnez un client dans la liste pour valider cette etape.'
                : 'Ouvrez la liste des clients puis selectionnez un client.'
              : isPickupStep
                ? 'Confirmez le lieu de prise en charge avant de continuer.'
                : isDropoffStep
                  ? 'Apres la saisie, selectionnez la suggestion HUG proposee.'
                  : isAmountStep
                    ? 'Saisissez un montant superieur a 0 pour valider cette etape.'
                  : isDepartureDateStep
                    ? 'Saisissez une date du jour ou future.'
                  : isDepartureTimeStep
                    ? "Saisissez l heure de depart."
                  : isRoundTripStep
                    ? "Activez le bouton 'Trajet AR' pour continuer."
                    : isReturnConfigStep
                      ? 'Verifiez la date de retour pre-remplie. L heure peut rester vide (a definir).'
              : ''}
          </p>
        )}
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.secondary}
            disabled={stepIndex === 0}
            onClick={() => {
              if (currentStep.id === 'dispatch-followup' && createBookingStepIndex >= 0) {
                setStepIndex(createBookingStepIndex);
                return;
              }
              setStepIndex((i) => Math.max(0, i - 1));
            }}
          >
            Précédent
          </button>
          <button
            type="button"
            className={styles.primary}
            disabled={!canContinue}
            onClick={() => {
              if (isLast) {
                finishGuide();
                return;
              }
              const nextIndex = Math.min(steps.length - 1, stepIndex + 1);
              const nextStep = steps[nextIndex];
              if (nextStep?.waitForElement) {
                setGuidePhase((prev) => nextGuidePhase(prev, 'PENDING_STEP'));
                setPendingStepId(nextStep.id);
                return;
              }
              setStepIndex(nextIndex);
            }}
          >
            {isLast ? 'Terminer' : 'Suivant'}
          </button>
        </div>
      </div>
    </>
  );

  if (typeof document === 'undefined' || !document.body) {
    return guideContent;
  }

  // Rendu en portal pour sortir des contextes de stacking des modales/pages.
  return createPortal(guideContent, document.body);
};

export default DemoInteractiveGuide;

