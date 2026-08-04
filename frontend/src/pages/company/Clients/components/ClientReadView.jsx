import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  FiX,
  FiEdit2,
  FiUser,
  FiHome,
  FiCheckCircle,
  FiAlertTriangle,
  FiHash,
  FiFileText,
  FiMapPin,
  FiCreditCard,
  FiClock,
  FiCornerUpLeft,
} from 'react-icons/fi';
import ClientStaysSection from './ClientStaysSection';
import ClientBillingPartiesSection from './ClientBillingPartiesSection';
import { fetchClientReservations } from '../../../../services/companyService';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import styles from './ClientReadView.module.css';

const ClientReadView = ({ client, onEdit, onClose, loading }) => {
  const [reservations, setReservations] = useState([]);
  const [reservationsLoading, setReservationsLoading] = useState(false);
  const [reservationsError, setReservationsError] = useState(null);
  const [searchParams] = useSearchParams();
  const returnTo = useMemo(() => {
    const raw = searchParams.get('returnTo');
    if (!raw) return null;
    try {
      const decoded = decodeURIComponent(raw);
      if (
        decoded.startsWith('/dashboard/company/')
        || decoded.startsWith('/company/')
      ) {
        return decoded;
      }
      return null;
    } catch {
      return null;
    }
  }, [searchParams]);

  const formatDate = (dateString) => {
    if (!dateString) return null;
    try {
      return new Date(dateString).toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
      });
    } catch {
      return null;
    }
  };

  const firstName = client.user_first_name || client.first_name || client.user?.first_name || '';
  const lastName = client.user_last_name || client.last_name || client.user?.last_name || '';
  const fullName = client.full_name || `${firstName} ${lastName}`.trim();

  const displayName = client.is_institution
    ? client.institution_name || `Institution #${client.id}`
    : fullName || `Client #${client.id}`;

  useEffect(() => {
    if (!client?.id || client.is_institution) return undefined;
    let isMounted = true;

    const loadReservations = async () => {
      try {
        setReservationsLoading(true);
        setReservationsError(null);
        const data = await fetchClientReservations(client.id, {
          limit: 3,
          includeInvoices: false,
        });
        const list = Array.isArray(data?.reservations)
          ? data.reservations
          : Array.isArray(data)
            ? data
            : [];
        if (isMounted) {
          setReservations(list);
        }
      } catch (err) {
        if (isMounted) {
          setReservationsError(
            err?.response?.data?.error || 'Erreur lors du chargement des courses'
          );
          setReservations([]);
        }
      } finally {
        if (isMounted) {
          setReservationsLoading(false);
        }
      }
    };

    loadReservations();

    return () => {
      isMounted = false;
    };
  }, [client?.id, client?.is_institution]);

  const recentReservations = useMemo(() => {
    const list = Array.isArray(reservations) ? [...reservations] : [];
    list.sort((a, b) => {
      const aTime = Date.parse(
        a?.scheduled_time || a?.pickup_time || a?.created_at || ''
      );
      const bTime = Date.parse(
        b?.scheduled_time || b?.pickup_time || b?.created_at || ''
      );
      const aVal = Number.isNaN(aTime) ? 0 : aTime;
      const bVal = Number.isNaN(bTime) ? 0 : bTime;
      return bVal - aVal;
    });
    return list.slice(0, 3);
  }, [reservations]);

  const formatStatus = (status) => {
    const normalized = String(status || '').toLowerCase();
    const labels = {
      pending: 'En attente',
      assigned: 'Assignée',
      accepted: 'Acceptée',
      in_progress: 'En cours',
      completed: 'Terminée',
      canceled: 'Annulée',
      cancelled: 'Annulée',
      rejected: 'Refusée',
      confirmed: 'Confirmée',
    };
    return labels[normalized] || (status ? String(status) : 'Statut inconnu');
  };

  const getLocationValue = (reservation, type) => {
    if (type === 'pickup') {
      return (
        reservation?.pickup_location ||
        reservation?.pickup_address ||
        reservation?.origin ||
        '\u2014'
      );
    }
    return (
      reservation?.dropoff_location ||
      reservation?.dropoff_address ||
      reservation?.destination ||
      '\u2014'
    );
  };

  const getGenderLabel = () => {
    const gender = String(client.user_gender || client.user?.gender || '').toUpperCase();
    if (gender === 'HOMME' || gender === 'MALE') return 'Monsieur';
    if (gender === 'FEMME' || gender === 'FEMALE') return 'Madame';
    return 'Autre';
  };

  const birthDate = formatDate(client.user_birth_date || client.user?.birth_date);
  const hasGender = !!(client.user_gender || client.user?.gender);
  const hasAvs = !!client.avs_number;
  const hasEmail = !!client.contact_email;
  const hasPhone = !!(client.contact_phone || client.phone);
  const hasEssentialInfo = !client.is_institution
    ? (birthDate || hasGender || hasAvs || hasEmail || hasPhone)
    : (hasEmail || hasPhone);

  const address = client.domicile_address || client.domicile?.address || '';
  const zip = client.domicile_zip || client.domicile?.zip || '';
  const city = client.domicile_city || client.domicile?.city || '';
  const hasAddress = !!(address || zip || city);
  const hasFacility = !!client.residence_facility;
  const hasLocationSection = hasAddress || hasFacility;

  const addressText = hasAddress
    ? [address, zip, city].filter(Boolean).join(', ')
    : null;

  const hasPreferentialRate = !!client.preferential_rate;
  const hasBillingCompany = client.is_institution && !!client.default_billing?.billed_to_company;
  const hasBillingContact = !!client.default_billing?.billed_to_contact;
  const hasBillingType = !!client.default_billing?.billed_to_type;
  const hasBillingSection = hasPreferentialRate || hasBillingCompany || hasBillingContact || hasBillingType;

  const Field = ({ label, children }) => {
    if (!children) return null;
    return (
      <div className={styles.infoItem}>
        <span className={styles.infoLabel}>{label}</span>
        <span className={styles.infoValue}>{children}</span>
      </div>
    );
  };

  return (
    <div className={styles.readView}>
      {/* Header */}
      <header className={styles.header} data-drawer-header>
        <div className={styles.headerTop}>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onClose?.();
            }}
            className={styles.closeButton}
            aria-label="Fermer"
            title="Fermer (ESC)"
          >
            <FiX size={18} />
          </button>
          <div className={styles.headerTitle}>
            <h2 className={styles.clientName}>{displayName}{client?.id != null && <span className={styles.clientId} title="ID client"> #{client.id}</span>}</h2>
            <div className={styles.headerBadges}>
              {client.is_institution ? (
                <span className={styles.badgeInstitution}>
                  <FiHome size={11} />
                  Institution
                </span>
              ) : (
                <span className={styles.badgeClient}>
                  <FiUser size={11} />
                  Client
                </span>
              )}
              {client.is_active ? (
                <span className={styles.badgeActive}>
                  <FiCheckCircle size={11} />
                  Actif
                </span>
              ) : (
                <span className={styles.badgeInactive}>
                  <FiAlertTriangle size={11} />
                  Inactif
                </span>
              )}
              <span className={styles.badgeId}>
                <FiHash size={11} />
                {client.id}
              </span>
            </div>
          </div>
          <button
            onClick={onEdit}
            className={styles.editButton}
            disabled={loading}
            title="Modifier le client"
          >
            <FiEdit2 size={14} />
            Modifier
          </button>
          {returnTo && (
            <button
              onClick={() => window.location.assign(returnTo)}
              className={styles.returnButton}
              title="Retour au contrôle facturation"
            >
              <FiCornerUpLeft size={14} />
              Retour
            </button>
          )}
        </div>
      </header>

      {/* Contenu scrollable */}
      <div className={styles.content}>
        {/* Informations essentielles */}
        {hasEssentialInfo && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>
              <FiFileText size={14} className={styles.sectionIcon} />
              Informations essentielles
            </h3>
            <div className={styles.infoGrid}>
              {!client.is_institution && (
                <>
                  <Field label="Date de naissance">{birthDate}</Field>
                  {hasGender && <Field label="Civilité">{getGenderLabel()}</Field>}
                  {hasAvs && <Field label="Numéro AVS">{client.avs_number}</Field>}
                </>
              )}
              {hasEmail && (
                <Field label="Email de contact">
                  <a href={`mailto:${client.contact_email}`}>{client.contact_email}</a>
                </Field>
              )}
              {hasPhone && (
                <Field label="Téléphone">
                  <a href={`tel:${client.contact_phone || client.phone}`}>
                    {client.contact_phone || client.phone}
                  </a>
                </Field>
              )}
            </div>
          </section>
        )}

        {/* Localisation */}
        {hasLocationSection && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>
              <FiMapPin size={14} className={styles.sectionIcon} />
              Localisation
            </h3>
            <div className={styles.infoGrid}>
              {hasFacility && (
                <Field label="Établissement de résidence">{client.residence_facility}</Field>
              )}
              {hasAddress && (
                <Field label="Adresse de domicile">{addressText}</Field>
              )}
            </div>
          </section>
        )}

        {/* Facturation */}
        {hasBillingSection && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>
              <FiCreditCard size={14} className={styles.sectionIcon} />
              Facturation
            </h3>
            <div className={styles.infoGrid}>
              {hasPreferentialRate && (
                <Field label="Tarif préférentiel">
                  {parseFloat(client.preferential_rate).toFixed(2)} CHF / trajet
                </Field>
              )}

              {hasBillingCompany && (
                <>
                  <Field label="Entreprise payeur">
                    {client.default_billing.billed_to_company.name || null}
                  </Field>
                  {client.default_billing.billed_to_company.address && (
                    <Field label="Adresse de l'entreprise">
                      {client.default_billing.billed_to_company.address}
                      {client.default_billing.billed_to_company.domicile_zip &&
                       client.default_billing.billed_to_company.domicile_city && (
                        <span>
                          , {client.default_billing.billed_to_company.domicile_zip}{' '}
                          {client.default_billing.billed_to_company.domicile_city}
                        </span>
                      )}
                    </Field>
                  )}
                  {client.default_billing.billed_to_company.contact_email && (
                    <Field label="Email entreprise">
                      <a href={`mailto:${client.default_billing.billed_to_company.contact_email}`}>
                        {client.default_billing.billed_to_company.contact_email}
                      </a>
                    </Field>
                  )}
                  {client.default_billing.billed_to_company.contact_phone && (
                    <Field label="Téléphone entreprise">
                      <a href={`tel:${client.default_billing.billed_to_company.contact_phone}`}>
                        {client.default_billing.billed_to_company.contact_phone}
                      </a>
                    </Field>
                  )}
                  {client.default_billing.billed_to_company.preferential_rate && (
                    <Field label="Tarif préférentiel entreprise">
                      {parseFloat(client.default_billing.billed_to_company.preferential_rate).toFixed(2)} CHF / trajet
                    </Field>
                  )}
                </>
              )}

              {hasBillingContact && (
                <Field label="Contact de facturation">
                  {client.default_billing.billed_to_contact}
                </Field>
              )}

              {hasBillingType && (
                <Field label="Type de facturation">
                  {client.default_billing.billed_to_type === 'clinic' ? 'Clinique' :
                   client.default_billing.billed_to_type === 'patient' ? 'Patient' :
                   client.default_billing.billed_to_type}
                </Field>
              )}
            </div>
          </section>
        )}

        {/* Tiers payeur (uniquement pour clients) */}
        {!client.is_institution && (
          <ClientBillingPartiesSection clientId={client.id} readOnly={true} />
        )}

        {/* Sejours d'hospitalisation (uniquement pour clients) */}
        {!client.is_institution && (
          <ClientStaysSection clientId={client.id} />
        )}

        {/* Dernières courses (uniquement pour clients) */}
        {!client.is_institution && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>
              <FiClock size={14} className={styles.sectionIcon} />
              Dernières courses ({recentReservations.length || 0})
            </h3>
            {reservationsLoading ? (
              <div className={styles.mutedText}>Chargement des courses…</div>
            ) : reservationsError ? (
              <div className={styles.errorText}>{reservationsError}</div>
            ) : recentReservations.length === 0 ? (
              <div className={styles.mutedText}>Aucune course récente.</div>
            ) : (
              <div className={styles.reservationsList}>
                {recentReservations.map((reservation) => {
                  const status = String(reservation?.status || '').toLowerCase();
                  return (
                    <div key={reservation.id} className={styles.reservationCard}>
                      <div className={styles.reservationMeta}>
                        <span className={styles.reservationDate}>
                          {renderBookingDateTime(reservation)}
                        </span>
                        <span className={styles.reservationStatus} data-status={status}>
                          {formatStatus(reservation?.status)}
                        </span>
                      </div>
                      <div className={styles.reservationRoute}>
                        <div className={styles.reservationRow}>
                          <span className={styles.reservationLabel}>Départ</span>
                          <span className={styles.reservationValue}>
                            {getLocationValue(reservation, 'pickup')}
                          </span>
                        </div>
                        <div className={styles.reservationRow}>
                          <span className={styles.reservationLabel}>Arrivée</span>
                          <span className={styles.reservationValue}>
                            {getLocationValue(reservation, 'dropoff')}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
};

export default ClientReadView;
