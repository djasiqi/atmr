// frontend/src/pages/company/Clients/components/ClientReadView.jsx
import React, { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import ClientStaysSection from './ClientStaysSection';
import ClientBillingPartiesSection from './ClientBillingPartiesSection';
import { fetchClientReservations } from '../../../../services/companyService';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import styles from './ClientReadView.module.css';

/**
 * Vue en mode LECTURE du drawer client
 * Affiche toutes les informations de manière organisée et lisible
 */
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
    if (!dateString) return 'Non renseigné';
    try {
      return new Date(dateString).toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
      });
    } catch {
      return 'Non renseigné';
    }
  };

  // Gérer les deux formats de données : DTO (user_first_name) et serialize (first_name)
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
        '—'
      );
    }
    return (
      reservation?.dropoff_location ||
      reservation?.dropoff_address ||
      reservation?.destination ||
      '—'
    );
  };

  return (
    <div className={styles.readView}>
      {/* Header */}
      <header className={styles.header} data-drawer-header>
        <div className={styles.headerTop}>
          <button
            onClick={onClose}
            className={styles.closeButton}
            aria-label="Fermer"
            title="Fermer (ESC)"
          >
            ✕
          </button>
          <div className={styles.headerTitle}>
            <h2 className={styles.clientName}>{displayName}</h2>
            <div className={styles.headerBadges}>
              {client.is_institution ? (
                <span className={styles.badgeInstitution}>🏢 Institution</span>
              ) : (
                <span className={styles.badgeClient}>👤 Client</span>
              )}
              {client.is_active ? (
                <span className={styles.badgeActive}>✅ Actif</span>
              ) : (
                <span className={styles.badgeInactive}>⚠️ Inactif</span>
              )}
              <span className={styles.badgeId}>#{client.id}</span>
            </div>
          </div>
          <button
            onClick={onEdit}
            className={styles.editButton}
            disabled={loading}
            title="Modifier le client"
          >
            ✏️ Modifier
          </button>
          {returnTo && (
            <button
              onClick={() => window.location.assign(returnTo)}
              className={styles.returnButton}
              title="Retour au contrôle facturation"
            >
              ↩ Retour au contrôle facturation
            </button>
          )}
        </div>
      </header>

      {/* Contenu scrollable */}
      <div className={styles.content}>
        {/* Informations essentielles */}
        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>📋 Informations essentielles</h3>
          <div className={styles.infoGrid}>
            {!client.is_institution && (
              <>
                {(client.user_birth_date || client.user?.birth_date) && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Date de naissance</span>
                    <span className={styles.infoValue}>
                      {formatDate(client.user_birth_date || client.user?.birth_date)}
                    </span>
                  </div>
                )}
                {(client.user_gender || client.user?.gender) && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Civilité</span>
                    <span className={styles.infoValue}>
                      {(() => {
                        const gender = String(client.user_gender || client.user?.gender || '').toUpperCase();
                        if (gender === 'HOMME' || gender === 'MALE') return 'Monsieur';
                        if (gender === 'FEMME' || gender === 'FEMALE') return 'Madame';
                        return 'Autre';
                      })()}
                    </span>
                  </div>
                )}
                {client.avs_number && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Numéro AVS</span>
                    <span className={styles.infoValue}>{client.avs_number}</span>
                  </div>
                )}
              </>
            )}
            {client.contact_email && (
              <div className={styles.infoItem}>
                <span className={styles.infoLabel}>Email de contact</span>
                <span className={styles.infoValue}>
                  <a href={`mailto:${client.contact_email}`}>{client.contact_email}</a>
                </span>
              </div>
            )}
            {(client.contact_phone || client.phone) && (
              <div className={styles.infoItem}>
                <span className={styles.infoLabel}>Téléphone</span>
                <span className={styles.infoValue}>
                  <a href={`tel:${client.contact_phone || client.phone}`}>
                    {client.contact_phone || client.phone}
                  </a>
                </span>
              </div>
            )}
          </div>
        </section>

        {/* Localisation */}
        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>📍 Localisation</h3>
          <div className={styles.infoGrid}>
            {client.residence_facility && (
              <div className={styles.infoItem}>
                <span className={styles.infoLabel}>Établissement de résidence</span>
                <span className={styles.infoValue}>{client.residence_facility}</span>
              </div>
            )}
            <div className={styles.infoItem}>
              <span className={styles.infoLabel}>Adresse de domicile</span>
              <span className={styles.infoValue}>
                {(() => {
                  // Gérer les deux formats : DTO (domicile_address) et serialize (domicile.address)
                  const address = client.domicile_address || client.domicile?.address || '';
                  const zip = client.domicile_zip || client.domicile?.zip || '';
                  const city = client.domicile_city || client.domicile?.city || '';
                  
                  if (address || zip || city) {
                    const parts = [address, zip, city].filter(Boolean);
                    return parts.join(', ');
                  }
                  return 'Non renseignée';
                })()}
              </span>
            </div>
          </div>
        </section>

        {/* Facturation */}
        <section className={styles.section}>
          <h3 className={styles.sectionTitle}>💰 Facturation</h3>
          <div className={styles.infoGrid}>
            {client.preferential_rate && (
              <div className={styles.infoItem}>
                <span className={styles.infoLabel}>Tarif préférentiel</span>
                <span className={styles.infoValue}>
                  {parseFloat(client.preferential_rate).toFixed(2)} CHF / trajet
                </span>
              </div>
            )}
            
            {/* Informations de la Company payeur (pour institutions) */}
            {client.is_institution && client.default_billing?.billed_to_company && (
              <>
                <div className={styles.infoItem}>
                  <span className={styles.infoLabel}>Entreprise payeur</span>
                  <span className={styles.infoValue}>
                    {client.default_billing.billed_to_company.name || '—'}
                  </span>
                </div>
                {client.default_billing.billed_to_company.address && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Adresse de l'entreprise</span>
                    <span className={styles.infoValue}>
                      {client.default_billing.billed_to_company.address}
                      {client.default_billing.billed_to_company.domicile_zip && 
                       client.default_billing.billed_to_company.domicile_city && (
                        <span>
                          , {client.default_billing.billed_to_company.domicile_zip}{' '}
                          {client.default_billing.billed_to_company.domicile_city}
                        </span>
                      )}
                    </span>
                  </div>
                )}
                {client.default_billing.billed_to_company.contact_email && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Email entreprise</span>
                    <span className={styles.infoValue}>
                      <a href={`mailto:${client.default_billing.billed_to_company.contact_email}`}>
                        {client.default_billing.billed_to_company.contact_email}
                      </a>
                    </span>
                  </div>
                )}
                {client.default_billing.billed_to_company.contact_phone && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Téléphone entreprise</span>
                    <span className={styles.infoValue}>
                      <a href={`tel:${client.default_billing.billed_to_company.contact_phone}`}>
                        {client.default_billing.billed_to_company.contact_phone}
                      </a>
                    </span>
                  </div>
                )}
                {client.default_billing.billed_to_company.preferential_rate && (
                  <div className={styles.infoItem}>
                    <span className={styles.infoLabel}>Tarif préférentiel entreprise</span>
                    <span className={styles.infoValue}>
                      {parseFloat(client.default_billing.billed_to_company.preferential_rate).toFixed(2)} CHF / trajet
                    </span>
                  </div>
                )}
              </>
            )}

            {/* Contact de facturation */}
            {client.default_billing?.billed_to_contact && (
              <div className={styles.infoItem}>
                <span className={styles.infoLabel}>Contact de facturation</span>
                <span className={styles.infoValue}>
                  {client.default_billing.billed_to_contact}
                </span>
              </div>
            )}

            {/* Type de facturation */}
            {client.default_billing?.billed_to_type && (
              <div className={styles.infoItem}>
                <span className={styles.infoLabel}>Type de facturation</span>
                <span className={styles.infoValue}>
                  {client.default_billing.billed_to_type === 'clinic' ? 'Clinique' :
                   client.default_billing.billed_to_type === 'patient' ? 'Patient' :
                   client.default_billing.billed_to_type}
                </span>
              </div>
            )}
          </div>
        </section>

        {/* Tiers payeur (uniquement pour clients) */}
        {!client.is_institution && (
          <ClientBillingPartiesSection clientId={client.id} readOnly={true} />
        )}

        {/* Séjours d'hospitalisation (uniquement pour clients) */}
        {!client.is_institution && (
          <ClientStaysSection clientId={client.id} />
        )}

        {/* Dernières courses (uniquement pour clients) */}
        {!client.is_institution && (
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>🚕 Dernières courses (3)</h3>
            {reservationsLoading ? (
              <div className={styles.mutedText}>Chargement des courses...</div>
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
