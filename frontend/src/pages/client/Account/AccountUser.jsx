// src/pages/client/Account/AccountUser.jsx
import React, { useEffect, useState, useCallback, useMemo } from 'react';
import apiClient, { logoutUser } from '../../../utils/apiClient';
import { useParams, useNavigate, Link } from 'react-router-dom';
import HeaderDashboard from '../../../components/layout/Header/HeaderDashboard';
import Footer from '../../../components/layout/Footer/Footer';
import ConfirmationModal from '../../../components/common/ConfirmationModal';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import { getApiErrorMessage } from '../../../utils/apiErrorMessage';
import { getActiveUser, hasActiveSession, setEnvUser } from '../../../utils/webAuthSession';
import { changeClientPassword } from '../../../services/clientService';
import './AccountUser.css';

import avatarMale from '../../../assets/images/avatar-male.png';
import avatarFemale from '../../../assets/images/avatar-female.png';
import defaultAvatar from '../../../assets/images/default-avatar.png';

function cleanDisplay(value) {
  if (value == null) return '';
  const s = String(value).trim();
  if (s === 'Non spécifié') return '';
  return s;
}

/** Mappe la réponse GET /clients/:public_id vers l’état formulaire (aligné domicile + user). */
function mapClientResponseToForm(data) {
  if (!data) return {};
  const u = data.user || {};
  const domicileAddr = cleanDisplay(data.domicile?.address);
  const userAddr = cleanDisplay(u.address);
  const firstName = cleanDisplay(data.first_name || u.first_name);
  const lastName = cleanDisplay(data.last_name || u.last_name);
  const birthDate = cleanDisplay(u.birth_date || data.birth_date);
  const genderRawValue = u.gender || data.gender;
  const genderRaw = genderRawValue != null ? String(genderRawValue) : '';
  const genderNorm =
    genderRaw === 'Non spécifié' || genderRaw === ''
      ? ''
      : genderRaw.toUpperCase();

  const access = data.access || {};

  return {
    first_name: firstName,
    last_name: lastName,
    email: cleanDisplay(u.email || data.contact_email),
    phone: cleanDisplay(data.phone || u.phone),
    address: domicileAddr || userAddr || cleanDisplay(data.address),
    floor: cleanDisplay(access.floor),
    door_code: cleanDisplay(access.door_code),
    access_notes: cleanDisplay(access.notes),
    birth_date: birthDate,
    gender: ['HOMME', 'FEMME', 'AUTRE'].includes(genderNorm) ? genderNorm : '',
    profile_image: u.profile_image || null,
    force_password_change: Boolean(u.force_password_change),
  };
}

function avatarForGender(gender) {
  if (gender === 'HOMME') return avatarMale;
  if (gender === 'FEMME') return avatarFemale;
  return defaultAvatar;
}

function accountInitials(firstName, lastName) {
  const a = String(firstName || '').trim().charAt(0);
  const b = String(lastName || '').trim().charAt(0);
  const pair = `${a}${b}`.toUpperCase();
  return pair || '?';
}

/** Score 0–100 selon les champs utiles aux réservations (heuristique côté client). */
function computeProfileCompletionPercent(profile) {
  if (!profile || typeof profile !== 'object') return 0;
  const t = (v) => String(v ?? '').trim().length > 0;
  let points = 0;
  let max = 0;
  const add = (filled, weight) => {
    max += weight;
    if (filled) points += weight;
  };
  add(t(profile.first_name) && t(profile.last_name), 15);
  add(t(profile.email), 10);
  add(t(profile.phone), 15);
  add(t(profile.birth_date), 10);
  add(t(profile.gender), 10);
  add(t(profile.address), 25);
  add(t(profile.floor) || t(profile.door_code) || t(profile.access_notes), 15);
  if (max === 0) return 0;
  return Math.min(100, Math.round((points / max) * 100));
}

const ACCOUNT_SECTION_META = {
  profile: {
    eyebrow: 'Mon espace',
    title: 'Mon profil',
    sub: 'Gérez vos informations, sécurité et préférences.',
  },
  personal: {
    eyebrow: 'Informations',
    title: 'Informations personnelles',
    sub: 'Coordonnées utilisées pour les réservations et les confirmations.',
  },
  security: {
    eyebrow: 'Sécurité',
    title: 'Sécurité du compte',
    sub: 'Protégez votre compte avec un mot de passe fort et la double authentification.',
  },
  privacy: {
    eyebrow: 'Données',
    title: 'Confidentialité et données',
    sub: 'Contrôlez vos données personnelles et vos préférences de communication.',
  },
};

function translateClientPasswordError(raw) {
  const m = String(raw || '').trim();
  const map = {
    'Incorrect old password': 'Le mot de passe actuel est incorrect.',
    'All fields are required': 'Veuillez remplir tous les champs.',
    'New passwords do not match': 'La confirmation ne correspond pas au nouveau mot de passe.',
  };
  return map[m] || raw;
}

/** Payload PUT conforme à ClientUpdateSchema (champs optionnels, téléphone normalisé). */
function buildClientUpdatePayload(form) {
  const p = {};
  const fn = (form.first_name || '').trim();
  const ln = (form.last_name || '').trim();
  if (fn) p.first_name = fn;
  if (ln) p.last_name = ln;
  const phoneRaw = (form.phone || '').replace(/\s/g, '').trim();
  if (phoneRaw) p.phone = phoneRaw;
  const addr = (form.address || '').trim();
  if (addr) p.address = addr;
  if (form.birth_date) p.birth_date = form.birth_date;
  if (form.gender && ['HOMME', 'FEMME', 'AUTRE'].includes(form.gender)) {
    p.gender = form.gender;
  }
  p.floor = (form.floor ?? '').trim();
  p.door_code = (form.door_code ?? '').trim();
  p.access_notes = (form.access_notes ?? '').trim();
  return p;
}

const AccountUser = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const [updatedProfile, setUpdatedProfile] = useState({});
  const [loadError, setLoadError] = useState(null);
  const [saveError, setSaveError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [profilePic, setProfilePic] = useState(defaultAvatar);
  const [activeSection, setActiveSection] = useState('profile');
  const [pwdCurrent, setPwdCurrent] = useState('');
  const [pwdNew, setPwdNew] = useState('');
  const [pwdConfirm, setPwdConfirm] = useState('');
  const [pwdSaving, setPwdSaving] = useState(false);
  const [pwdError, setPwdError] = useState(null);
  const [pwdSuccess, setPwdSuccess] = useState(false);
  const [showPwdForm, setShowPwdForm] = useState(false);
  const [securityHint, setSecurityHint] = useState(null);
  const [privacyAnalytics, setPrivacyAnalytics] = useState(false);
  const [privacySms, setPrivacySms] = useState(true);
  const [privacyEmail, setPrivacyEmail] = useState(true);
  const [privacyHint, setPrivacyHint] = useState(null);
  const [logoutModalOpen, setLogoutModalOpen] = useState(false);
  const [deleteAccountModalOpen, setDeleteAccountModalOpen] = useState(false);

  const accountSections = [
    { id: 'profile', label: 'Profil' },
    { id: 'personal', label: 'Informations personnelles' },
    { id: 'security', label: 'Sécurité' },
    { id: 'privacy', label: 'Confidentialité et données' },
  ];

  const refreshProfilePic = useCallback((gender, uploadedPic) => {
    if (uploadedPic) {
      setProfilePic(uploadedPic);
    } else {
      setProfilePic(avatarForGender(gender));
    }
  }, []);

  const reloadProfile = useCallback(async () => {
    const { data } = await apiClient.get(`/clients/${public_id}`);
    const mapped = mapClientResponseToForm(data);
    setUpdatedProfile(mapped);
    return mapped;
  }, [public_id]);

  useEffect(() => {
    if (!hasActiveSession()) {
      navigate('/login');
      return;
    }

    setLoading(true);
    setLoadError(null);
    apiClient
      .get(`/clients/${public_id}`)
      .then((response) => {
        const mapped = mapClientResponseToForm(response.data);
        setUpdatedProfile(mapped);
      })
      .catch(() => {
        setLoadError('Impossible de charger le compte utilisateur.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [public_id, navigate]);

  useEffect(() => {
    refreshProfilePic(
      updatedProfile.gender || '',
      updatedProfile.profile_image || null
    );
  }, [updatedProfile.gender, updatedProfile.profile_image, refreshProfilePic]);

  useEffect(() => {
    if (activeSection !== 'security') {
      setPwdError(null);
      setShowPwdForm(false);
      setSecurityHint(null);
    }
    if (activeSection !== 'privacy') {
      setPrivacyHint(null);
    }
  }, [activeSection]);

  const sessionsSummaryLine = useMemo(() => {
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
      if (/^Europe\/(Zurich|Geneva)$/i.test(tz)) {
        return '1 appareil connecté — navigateur web, Suisse';
      }
    } catch {
      /* ignore */
    }
    const lang = typeof navigator !== 'undefined' ? navigator.language || '' : '';
    if (lang.toLowerCase().startsWith('fr')) {
      return '1 appareil connecté — navigateur web, Suisse';
    }
    return '1 appareil connecté — navigateur web';
  }, []);

  const profileCompletionPct = useMemo(
    () => computeProfileCompletionPercent(updatedProfile),
    [updatedProfile]
  );

  const handleUpdateProfile = () => {
    setSaveError(null);
    setSaveSuccess(false);
    const payload = buildClientUpdatePayload(updatedProfile);
    if (Object.keys(payload).length === 0) {
      setSaveError('Aucune modification à enregistrer.');
      return;
    }
    setSaving(true);
    apiClient
      .put(`/clients/${public_id}`, payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      })
      .then(async () => {
        setSaveSuccess(true);
        await reloadProfile();
        try {
          const u = getActiveUser();
          if (u) {
            if (payload.first_name != null) u.first_name = payload.first_name;
            if (payload.last_name != null) u.last_name = payload.last_name;
            setEnvUser(u);
          }
        } catch {
          /* ignore */
        }
      })
      .catch((err) => {
        setSaveError(
          getApiErrorMessage(err, 'Impossible de mettre à jour le compte.')
        );
      })
      .finally(() => {
        setSaving(false);
      });
  };

  const handleChangePassword = (e) => {
    e.preventDefault();
    setPwdError(null);
    setPwdSuccess(false);
    if (!pwdCurrent || !pwdNew || !pwdConfirm) {
      setPwdError('Veuillez remplir tous les champs.');
      return;
    }
    if (pwdNew !== pwdConfirm) {
      setPwdError('La confirmation ne correspond pas au nouveau mot de passe.');
      return;
    }
    if (pwdNew.length < 8) {
      setPwdError('Le nouveau mot de passe doit contenir au moins 8 caractères.');
      return;
    }
    setPwdSaving(true);
    changeClientPassword(public_id, {
      oldPassword: pwdCurrent,
      newPassword: pwdNew,
      confirmPassword: pwdConfirm,
    })
      .then(() => {
        setPwdSuccess(true);
        setPwdCurrent('');
        setPwdNew('');
        setPwdConfirm('');
        setShowPwdForm(false);
      })
      .catch((err) => {
        setPwdSuccess(false);
        const base = getApiErrorMessage(err, 'Impossible de modifier le mot de passe.');
        setPwdError(translateClientPasswordError(base));
      })
      .finally(() => {
        setPwdSaving(false);
      });
  };

  const handleEnable2fa = useCallback(() => {
    setSecurityHint(
      'La double authentification (2FA) sera disponible prochainement. En attendant, utilisez un mot de passe unique et fort.'
    );
  }, []);

  const handleDisconnectAll = useCallback(() => {
    setLogoutModalOpen(true);
  }, []);

  const confirmLogoutSession = useCallback(() => {
    setLogoutModalOpen(false);
    logoutUser();
  }, []);

  const handlePrivacyExport = useCallback(() => {
    setPrivacyHint(
      'Votre demande d’export a été enregistrée. Vous recevrez un lien sécurisé par e-mail lorsque la fonctionnalité sera activée côté Lirie.'
    );
  }, []);

  const handleDeleteAccount = useCallback(() => {
    setDeleteAccountModalOpen(true);
  }, []);

  const confirmDeleteAccountStep = useCallback(() => {
    setDeleteAccountModalOpen(false);
    setPrivacyHint(
      'La suppression en ligne de compte n’est pas encore disponible. Utilisez la page Contact (Support) depuis le site et indiquez votre identité de compte pour qu’un opérateur traite votre demande.'
    );
  }, []);

  if (loading) return <p>Chargement...</p>;
  if (loadError) {
    return (
      <div className="account-container">
        <HeaderDashboard />
        <div className="account-mainSurface">
          <p className="error" role="alert">
            {loadError}
          </p>
          <Footer />
        </div>
      </div>
    );
  }

  return (
    <div className="account-container">
      <HeaderDashboard />
      <div className="account-mainSurface">
        <main className="account-content">
          <div className="account-body">
          <div className="account-inner">
            <div className="account-shell">
              <nav className="account-sidebar" aria-label="Navigation compte">
                <div className="account-sidebarUser">
                  <div className="account-sidebarAvatar" aria-hidden="true">
                    {accountInitials(updatedProfile.first_name, updatedProfile.last_name)}
                  </div>
                  <div className="account-sidebarUserText">
                    <span className="account-sidebarUserName">
                      {`${updatedProfile.first_name || ''} ${updatedProfile.last_name || ''}`.trim() || 'Compte'}
                    </span>
                    <span className="account-sidebarUserEmail" title={updatedProfile.email || ''}>
                      {updatedProfile.email || '—'}
                    </span>
                  </div>
                </div>
                <ul className="account-sidebarNav">
                  {accountSections.map((section) => (
                    <li key={section.id}>
                      <button
                        type="button"
                        className={activeSection === section.id ? 'is-active' : ''}
                        aria-current={activeSection === section.id ? 'true' : undefined}
                        onClick={() => setActiveSection(section.id)}
                      >
                        {section.label}
                      </button>
                    </li>
                  ))}
                </ul>
              </nav>

              <section className="account-main" aria-live="polite">
                <header className="account-panelHeader">
                  <p className="account-panelEyebrow">{ACCOUNT_SECTION_META[activeSection].eyebrow}</p>
                  <h1 className="account-panelTitle">{ACCOUNT_SECTION_META[activeSection].title}</h1>
                  <p className="account-panelSub">{ACCOUNT_SECTION_META[activeSection].sub}</p>
                </header>

                {activeSection === 'profile' ? (
              <div className="home-panel" data-testid="account-profile-ui">
                <section className="home-identity-card">
                  <div className="home-identity-inner">
                    <img src={profilePic} alt="" className="profile-pic profile-pic--identityHero" />
                    <div className="home-identity-text">
                      <h2 className="home-identity-name">
                        {`${updatedProfile.first_name || ''} ${updatedProfile.last_name || ''}`.trim() || 'Mon profil'}
                      </h2>
                      <p className="home-identity-email">{updatedProfile.email || 'Courriel non renseigné'}</p>
                      {updatedProfile.email ? (
                        <span className="home-identity-verifiedPill">Compte vérifié</span>
                      ) : null}
                    </div>
                  </div>
                  <div
                    className="home-completion-row"
                    role="group"
                    aria-label={`Profil complété à ${profileCompletionPct} pour cent`}
                  >
                    <span className="home-completion-label" id="account-profile-completion-label">
                      Profil complété à
                    </span>
                    <div
                      className="home-completion-barWrap"
                      role="progressbar"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={profileCompletionPct}
                      aria-labelledby="account-profile-completion-label"
                    >
                      <div
                        className="home-completion-bar"
                        style={{ width: `${profileCompletionPct}%` }}
                      />
                    </div>
                    <span className="home-completion-pct" aria-hidden="true">
                      {profileCompletionPct} %
                    </span>
                  </div>
                </section>
                <div className="home-actions-card">
                  <div className="home-actions-cardHead" id="account-shortcuts-label">
                    Gérer mon compte
                  </div>
                  <div className="home-actions-grid" role="group" aria-labelledby="account-shortcuts-label">
                    <button
                      type="button"
                      className="home-action-tile"
                      onClick={() => setActiveSection('personal')}
                    >
                      <span className="home-action-tile-label">Informations personnelles</span>
                      <span className="home-action-tile-sub">Nom, adresse, téléphone</span>
                    </button>
                    <button type="button" className="home-action-tile" onClick={() => setActiveSection('security')}>
                      <span className="home-action-tile-label">Sécurité</span>
                      <span className="home-action-tile-sub">Mot de passe, connexions</span>
                    </button>
                    <button type="button" className="home-action-tile" onClick={() => setActiveSection('privacy')}>
                      <span className="home-action-tile-label">Confidentialité</span>
                      <span className="home-action-tile-sub">Données, suppressions</span>
                    </button>
                  </div>
                </div>
                <section className="home-suggestion-card">
                  <h3>Terminer votre profil</h3>
                  <p>
                    Complétez vos informations personnelles et de sécurité pour accélérer vos
                    réservations et protéger votre compte.
                  </p>
                  <button type="button" onClick={() => setActiveSection('personal')}>
                    Continuer
                  </button>
                </section>
              </div>
            ) : null}

            {activeSection === 'personal' ? (
              <>
                <section className="profile-card" aria-label="Informations personnelles">
                  <div className="profile-cardForm">
                    <fieldset className="profile-fieldset">
                      <legend className="visually-hidden">Identité</legend>
                      <div className="profile-identityLayout">
                        <figure className="profile-avatarBlock">
                          <img src={profilePic} alt="" className="profile-pic" />
                          <figcaption className="photo-hint profile-avatarCaption">
                            Photo non modifiable ici.
                          </figcaption>
                        </figure>
                        <div className="profile-identityFields">
                          <div className="profile-form-row">
                            <div className="profile-field">
                              <label htmlFor="account-first-name">Prénom</label>
                              <input
                                id="account-first-name"
                                type="text"
                                autoComplete="given-name"
                                value={updatedProfile.first_name || ''}
                                onChange={(e) =>
                                  setUpdatedProfile({
                                    ...updatedProfile,
                                    first_name: e.target.value,
                                  })
                                }
                              />
                            </div>
                            <div className="profile-field">
                              <label htmlFor="account-last-name">Nom</label>
                              <input
                                id="account-last-name"
                                type="text"
                                autoComplete="family-name"
                                value={updatedProfile.last_name || ''}
                                onChange={(e) =>
                                  setUpdatedProfile({
                                    ...updatedProfile,
                                    last_name: e.target.value,
                                  })
                                }
                              />
                            </div>
                          </div>
                          <div className="profile-form-row">
                            <div className="profile-field">
                              <label htmlFor="account-birth">Date de naissance</label>
                              <input
                                id="account-birth"
                                type="date"
                                autoComplete="bday"
                                value={updatedProfile.birth_date || ''}
                                onChange={(e) =>
                                  setUpdatedProfile({
                                    ...updatedProfile,
                                    birth_date: e.target.value,
                                  })
                                }
                              />
                            </div>
                            <div className="profile-field">
                              <label htmlFor="account-gender">Genre</label>
                              <select
                                id="account-gender"
                                value={updatedProfile.gender || ''}
                                onChange={(e) =>
                                  setUpdatedProfile({ ...updatedProfile, gender: e.target.value })
                                }
                              >
                                <option value="">Sélectionner…</option>
                                <option value="HOMME">Homme</option>
                                <option value="FEMME">Femme</option>
                                <option value="AUTRE">Autre</option>
                              </select>
                            </div>
                          </div>
                        </div>
                      </div>
                    </fieldset>

                    <fieldset className="profile-fieldset">
                      <legend className="visually-hidden">Contact</legend>
                      <div className="profile-form-row">
                        <div className="profile-field">
                          <label htmlFor="account-email">Courriel</label>
                          <input
                            id="account-email"
                            type="email"
                            autoComplete="email"
                            value={updatedProfile.email || ''}
                            disabled
                          />
                          <p className="field-hint profile-fieldNote">Non modifiable.</p>
                        </div>
                        <div className="profile-field">
                          <label htmlFor="account-phone">Téléphone</label>
                          <input
                            id="account-phone"
                            type="tel"
                            autoComplete="tel"
                            inputMode="tel"
                            placeholder="ex. +41791234567 ou 0791234567"
                            value={updatedProfile.phone || ''}
                            onChange={(e) =>
                              setUpdatedProfile({ ...updatedProfile, phone: e.target.value })
                            }
                          />
                        </div>
                      </div>
                    </fieldset>

                    <fieldset className="profile-fieldset">
                      <legend className="visually-hidden">Adresse et accès</legend>
                      <div className="profile-field">
                        <label htmlFor="account-address">Adresse</label>
                        <AddressAutocomplete
                          inputId="account-address"
                          name="address"
                          value={updatedProfile.address || ''}
                          onChange={(e) =>
                            setUpdatedProfile({
                              ...updatedProfile,
                              address: e.target.value,
                            })
                          }
                          onSelect={(item) =>
                            setUpdatedProfile({
                              ...updatedProfile,
                              address: item.label || '',
                            })
                          }
                          placeholder="Rechercher votre adresse (Suisse)…"
                          inputClassName="account-ac-input"
                        />
                      </div>
                      <div className="profile-form-row">
                        <div className="profile-field">
                          <label htmlFor="account-floor">Étage / appartement</label>
                          <input
                            id="account-floor"
                            type="text"
                            autoComplete="off"
                            maxLength={20}
                            placeholder="ex. 3e, Rez"
                            value={updatedProfile.floor || ''}
                            onChange={(e) =>
                              setUpdatedProfile({ ...updatedProfile, floor: e.target.value })
                            }
                          />
                        </div>
                        <div className="profile-field">
                          <label htmlFor="account-door-code">Code / interphone</label>
                          <input
                            id="account-door-code"
                            type="text"
                            autoComplete="off"
                            maxLength={50}
                            placeholder="Digicode, nom affiché…"
                            value={updatedProfile.door_code || ''}
                            onChange={(e) =>
                              setUpdatedProfile({ ...updatedProfile, door_code: e.target.value })
                            }
                          />
                        </div>
                      </div>
                      <div className="profile-field">
                        <label htmlFor="account-access-notes">Complément d’accès</label>
                        <textarea
                          id="account-access-notes"
                          className="profile-textarea"
                          rows={3}
                          maxLength={4000}
                          placeholder="Entrée, parking, accès PMR, consignes…"
                          value={updatedProfile.access_notes || ''}
                          onChange={(e) =>
                            setUpdatedProfile({
                              ...updatedProfile,
                              access_notes: e.target.value,
                            })
                          }
                        />
                      </div>
                    </fieldset>
                  </div>

                  <footer className="profile-cardActions">
                    {saveSuccess ? (
                      <p className="success-banner" role="status">
                        Profil mis à jour.
                      </p>
                    ) : null}
                    {saveError ? (
                      <p className="error profile-saveError" role="alert">
                        {saveError}
                      </p>
                    ) : null}

                    <button
                      type="button"
                      className="save-button"
                      onClick={handleUpdateProfile}
                      disabled={saving}
                    >
                      {saving ? 'Enregistrement…' : 'Sauvegarder'}
                    </button>
                  </footer>
                </section>

                <section className="payment-info-section" aria-labelledby="payment-info-heading">
                  <h2 id="payment-info-heading">Paiement des courses</h2>
                  <p className="payment-info-text">
                    Aucune carte bancaire ni TWINT n’est stocké sur ce compte. Pour une course à votre charge, le
                    règlement se fait par un paiement sécurisé <strong>Saferpay</strong> immédiatement après la
                    réservation (redirection automatique depuis votre tableau de bord). Si l’ouverture du paiement
                    échoue, vous pouvez réessayer depuis le tableau de bord ou depuis{' '}
                    <Link to={`/reservations/${public_id}`} className="payment-link">
                      Mes courses
                    </Link>
                    .
                  </p>
                  <p className="payment-info-text">
                    Les courses prises en charge par une assurance ou une institution ne passent pas par ce
                    paiement en ligne.
                  </p>
                  <p className="payment-info-text">
                    Tant que le paiement n’est pas validé, la demande n’est pas transmise aux entreprises de
                    transport comme réservation à traiter.
                  </p>
                </section>
              </>
            ) : null}

            {activeSection === 'security' ? (
              <div className="security-pageBlock" data-testid="security-info-ui">
                {securityHint ? (
                  <p className="security-inlineNotice" role="status">
                    {securityHint}
                  </p>
                ) : null}
                <section className="security-rowsCard" aria-label="Réglages de sécurité">
                  <div className="security-row">
                    <div className="security-rowMain">
                      <div className="security-rowTitle">Mot de passe</div>
                      <p className="security-rowDesc">
                        {pwdSuccess
                          ? 'Dernière modification : à l’instant'
                          : 'Dernière modification : il y a plus de 6 mois'}
                      </p>
                    </div>
                    <span
                      className={`security-rowBadge${updatedProfile.force_password_change ? ' security-rowBadge--warn' : ' security-rowBadge--muted'}`}
                    >
                      {updatedProfile.force_password_change ? 'Action requise' : 'Non renforcé'}
                    </span>
                    <button
                      type="button"
                      className="security-rowAction"
                      onClick={() => setShowPwdForm((v) => !v)}
                      aria-expanded={showPwdForm}
                    >
                      {showPwdForm ? 'Fermer' : 'Modifier'}
                    </button>
                  </div>

                  {showPwdForm ? (
                    <div className="security-passwordDrawer">
                      <p className="security-password-intro">
                        Saisissez votre mot de passe actuel puis choisissez un nouveau mot de passe fort.
                      </p>
                      <form className="profile-cardForm" onSubmit={handleChangePassword} noValidate>
                        <div className="profile-field">
                          <label htmlFor="client-pwd-current">Mot de passe actuel</label>
                          <input
                            id="client-pwd-current"
                            name="currentPassword"
                            type="password"
                            autoComplete="current-password"
                            value={pwdCurrent}
                            onChange={(ev) => setPwdCurrent(ev.target.value)}
                            disabled={pwdSaving}
                          />
                        </div>
                        <div className="profile-field">
                          <label htmlFor="client-pwd-new">Nouveau mot de passe</label>
                          <input
                            id="client-pwd-new"
                            name="newPassword"
                            type="password"
                            autoComplete="new-password"
                            value={pwdNew}
                            onChange={(ev) => setPwdNew(ev.target.value)}
                            disabled={pwdSaving}
                          />
                          <p className="field-hint">
                            Au moins 8 caractères ; majuscule, minuscule et chiffre si exigé par la plateforme.
                          </p>
                        </div>
                        <div className="profile-field">
                          <label htmlFor="client-pwd-confirm">Confirmer le nouveau mot de passe</label>
                          <input
                            id="client-pwd-confirm"
                            name="confirmPassword"
                            type="password"
                            autoComplete="new-password"
                            value={pwdConfirm}
                            onChange={(ev) => setPwdConfirm(ev.target.value)}
                            disabled={pwdSaving}
                          />
                        </div>
                        {pwdError ? (
                          <p className="error profile-saveError" role="alert">
                            {pwdError}
                          </p>
                        ) : null}
                        {pwdSuccess ? (
                          <p className="success-banner" role="status">
                            Mot de passe mis à jour. Un e-mail de confirmation peut vous être envoyé.
                          </p>
                        ) : null}
                        <button type="submit" className="save-button" disabled={pwdSaving}>
                          {pwdSaving ? 'Enregistrement…' : 'Mettre à jour le mot de passe'}
                        </button>
                      </form>
                    </div>
                  ) : null}

                  <div className="security-row">
                    <div className="security-rowMain">
                      <div className="security-rowTitle">Double authentification (2FA)</div>
                      <p className="security-rowDesc">
                        Protégez votre compte avec un second facteur à la connexion.
                      </p>
                    </div>
                    <span className="security-rowBadge security-rowBadge--muted">Désactivée</span>
                    <button type="button" className="security-rowAction" onClick={handleEnable2fa}>
                      Activer
                    </button>
                  </div>

                  <div className="security-row">
                    <div className="security-rowMain">
                      <div className="security-rowTitle">Adresse e-mail vérifiée</div>
                      <p className="security-rowDesc">{updatedProfile.email || '—'}</p>
                    </div>
                    {updatedProfile.email ? (
                      <span className="security-rowBadge security-rowBadge--ok">Vérifiée</span>
                    ) : (
                      <span className="security-rowBadge security-rowBadge--muted">Non renseignée</span>
                    )}
                  </div>

                  <div className="security-row security-row--last">
                    <div className="security-rowMain">
                      <div className="security-rowTitle">Sessions actives</div>
                      <p className="security-rowDesc">{sessionsSummaryLine}</p>
                    </div>
                    <button type="button" className="security-rowAction" onClick={handleDisconnectAll}>
                      Déconnecter tout
                    </button>
                  </div>
                </section>
              </div>
            ) : null}

            {activeSection === 'privacy' ? (
              <div className="privacy-pageBlock" data-testid="data-access-ui">
                {privacyHint ? (
                  <p className="privacy-inlineNotice" role="status">
                    {privacyHint}
                  </p>
                ) : null}
                <section className="privacy-rowsCard" aria-label="Préférences de confidentialité">
                  <div className="privacy-row">
                    <div className="privacy-rowMain">
                      <div className="privacy-rowTitle">Analyse d&apos;usage</div>
                      <p className="privacy-rowDesc">
                        Permettre à Lirie d&apos;analyser votre utilisation pour améliorer le service (anonymisé).
                      </p>
                    </div>
                    <label className="privacy-toggle">
                      <span className="visually-hidden">Analyse d&apos;usage</span>
                      <input
                        type="checkbox"
                        checked={privacyAnalytics}
                        onChange={(e) => setPrivacyAnalytics(e.target.checked)}
                      />
                      <span className="privacy-toggleSlider" aria-hidden="true" />
                    </label>
                  </div>

                  <div className="privacy-row">
                    <div className="privacy-rowMain">
                      <div className="privacy-rowTitle">Notifications SMS</div>
                      <p className="privacy-rowDesc">
                        Recevoir les confirmations et alertes de vos transports par SMS.
                      </p>
                    </div>
                    <label className="privacy-toggle">
                      <span className="visually-hidden">Notifications SMS</span>
                      <input
                        type="checkbox"
                        checked={privacySms}
                        onChange={(e) => setPrivacySms(e.target.checked)}
                      />
                      <span className="privacy-toggleSlider" aria-hidden="true" />
                    </label>
                  </div>

                  <div className="privacy-row">
                    <div className="privacy-rowMain">
                      <div className="privacy-rowTitle">Notifications e-mail</div>
                      <p className="privacy-rowDesc">
                        Recevoir les résumés et confirmations de courses par e-mail.
                      </p>
                    </div>
                    <label className="privacy-toggle">
                      <span className="visually-hidden">Notifications e-mail</span>
                      <input
                        type="checkbox"
                        checked={privacyEmail}
                        onChange={(e) => setPrivacyEmail(e.target.checked)}
                      />
                      <span className="privacy-toggleSlider" aria-hidden="true" />
                    </label>
                  </div>

                  <div className="privacy-row privacy-row--last">
                    <div className="privacy-rowMain">
                      <div className="privacy-rowTitle">Exporter mes données</div>
                      <p className="privacy-rowDesc">
                        Télécharger l&apos;ensemble de vos données personnelles (RGPD).
                      </p>
                    </div>
                    <button type="button" className="privacy-rowAction" onClick={handlePrivacyExport}>
                      Demander
                    </button>
                  </div>
                </section>

                <section className="privacy-dangerZone" aria-labelledby="privacy-delete-heading">
                  <div className="privacy-dangerText">
                    <h2 id="privacy-delete-heading" className="privacy-dangerTitle">
                      Supprimer mon compte
                    </h2>
                    <p className="privacy-dangerDesc">
                      La suppression est irréversible. Toutes vos données, réservations et historique seront effacés
                      définitivement.
                    </p>
                  </div>
                  <button type="button" className="privacy-dangerBtn" onClick={handleDeleteAccount}>
                    Supprimer mon compte
                  </button>
                </section>
              </div>
            ) : null}
              </section>
            </div>
          </div>
        </div>
        </main>
        <Footer />
        <ConfirmationModal
          isOpen={logoutModalOpen}
          onClose={() => setLogoutModalOpen(false)}
          onConfirm={confirmLogoutSession}
          title="Déconnexion"
          message="Déconnecter cette session ? Vous devrez vous reconnecter pour accéder à nouveau à votre compte."
          confirmText="Déconnecter"
          cancelText="Annuler"
          confirmButtonVariant="primary"
        />
        <ConfirmationModal
          isOpen={deleteAccountModalOpen}
          onClose={() => setDeleteAccountModalOpen(false)}
          onConfirm={confirmDeleteAccountStep}
          title="Supprimer mon compte"
          message="La suppression du compte est irréversible : réservations, historique et données seront effacés définitivement. Souhaitez-vous poursuivre pour afficher la suite des instructions ?"
          confirmText="Continuer"
          cancelText="Annuler"
          confirmButtonVariant="danger"
        />
      </div>
    </div>
  );
};

export default AccountUser;
