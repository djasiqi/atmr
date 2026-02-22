/**
 * Composant de configuration des emails transactionnels (Brevo)
 *
 * Permet de :
 * - Configurer l'adresse email d'envoi (from_email + from_name)
 * - Récupérer les enregistrements DNS à configurer (SPF + DKIM)
 * - Vérifier que le domaine est validé dans Brevo
 */

import React, { useState, useEffect } from 'react';
import {
  FiCheck,
  FiX,
  FiMail,
  FiClipboard,
  FiLoader,
  FiRefreshCw,
  FiSearch,
  FiBarChart2,
  FiInfo,
  FiAlertTriangle,
  FiTool,
  FiHelpCircle,
} from 'react-icons/fi';
import {
  getEmailConfig,
  setupEmailDomain,
  verifyEmailDomain,
  diagnosticEmailDomain,
} from '../../../../services/emailService';
import styles from './EmailConfigSection.module.css';

const EmailConfigSection = ({ companyId, showHeader = true, compact = false }) => {
  const [loading, setLoading] = useState(false);
  const [configuring, setConfiguring] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [diagnosing, setDiagnosing] = useState(false);
  const [config, setConfig] = useState(null);
  const [dnsRecords, setDnsRecords] = useState(null);
  const [domainVerified, setDomainVerified] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [diagnostic, setDiagnostic] = useState(null);

  // État du formulaire
  const [fromEmail, setFromEmail] = useState('');
  const [fromName, setFromName] = useState('');

  // Charger la configuration au montage
  useEffect(() => {
    loadConfig();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [companyId]);

  const loadConfig = async () => {
    setLoading(true);
    try {
      const data = await getEmailConfig();
      setConfig(data);

      if (data.configured) {
        setFromEmail(data.from_email || '');
        setFromName(data.from_name || '');
        setDomainVerified(data.domain_verified || false);
        setDnsRecords(data.dns_records);
      }
    } catch (err) {
      console.error('Erreur lors du chargement de la configuration email:', err);
      setError('Impossible de charger la configuration email');
    } finally {
      setLoading(false);
    }
  };

  const handleConfigure = async (e) => {
    e.preventDefault();
    setConfiguring(true);
    setMessage('');
    setError('');

    try {
      const result = await setupEmailDomain({
        from_email: fromEmail,
        from_name: fromName,
      });

      if (result.success) {
        setMessage(result.message);
        setDnsRecords(result.dns_records);

        setDomainVerified(result.verified);
        setConfig({
          ...config,
          configured: true,
          from_email: result.from_email,
          from_name: result.from_name,
          domain_verified: result.verified,
          dns_records: result.dns_records,
        });
      }
    } catch (err) {
      console.error('Erreur lors de la configuration:', err);
      const errorMsg = err.response?.data?.error || 'Erreur lors de la configuration';
      setError(errorMsg);
    } finally {
      setConfiguring(false);
    }
  };

  const handleVerify = async () => {
    setVerifying(true);
    setMessage('');
    setError('');

    try {
      const result = await verifyEmailDomain();

      if (result.verified) {
        setMessage(result.message);
        setDomainVerified(true);
      } else {
        setError(result.message);
        setDomainVerified(false);
      }
    } catch (err) {
      console.error('Erreur lors de la vérification:', err);
      const errorMsg = err.response?.data?.error || 'Erreur lors de la vérification';
      setError(errorMsg);
    } finally {
      setVerifying(false);
    }
  };

  const handleDiagnostic = async () => {
    setDiagnosing(true);
    setMessage('');
    setError('');
    setDiagnostic(null);

    try {
      const result = await diagnosticEmailDomain();

      if (result.success) {
        setDiagnostic(result);
        setMessage('Diagnostic effectué avec succès');
      } else {
        setError(result.error || 'Erreur lors du diagnostic');
        setDiagnostic(result);
      }
    } catch (err) {
      console.error('Erreur lors du diagnostic:', err);
      const errorMsg = err.response?.data?.error || 'Erreur lors du diagnostic';
      setError(errorMsg);
    } finally {
      setDiagnosing(false);
    }
  };

  const copyToClipboard = (text, label) => {
    navigator.clipboard.writeText(text);
    setMessage(`${label} copié dans le presse-papier`);
    setTimeout(() => setMessage(''), 3000);
  };

  const getStatusBadge = () => {
    if (!config?.configured) {
      return <span className={styles.badgeDefault}>Non configuré</span>;
    }
    if (domainVerified) {
      return (
        <span className={`${styles.badgeSuccess} ${styles.badgeWithIcon}`}>
          <FiCheck /> Vérifié
        </span>
      );
    }
    return (
      <span className={`${styles.badgeWarning} ${styles.badgeWithIcon}`}>
        <FiLoader className={styles.spinnerInline} /> En attente de validation DNS
      </span>
    );
  };

  if (loading) {
    return (
      <div
        className={`${styles.emailConfigCard} ${compact ? styles.emailConfigCompact : ''}`}
      >
        {showHeader && (
          <h2 className={styles.headerTitleWithIcon}>
            <FiMail /> Configuration Email Transactionnel
          </h2>
        )}
        {!showHeader && <div className={styles.headerCompact}>{getStatusBadge()}</div>}
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Chargement...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${styles.emailConfigCard} ${compact ? styles.emailConfigCompact : ''}`}>
      {showHeader ? (
        <div className={styles.header}>
          <h2 className={styles.headerTitleWithIcon}>
            <FiMail /> Configuration Email Transactionnel
          </h2>
          {getStatusBadge()}
        </div>
      ) : (
        <div className={styles.headerCompact}>{getStatusBadge()}</div>
      )}

      {/* Messages */}
      {message && <div className={styles.successMessage}>{message}</div>}
      {error && <div className={styles.errorMessage}>{error}</div>}

      {/* Introduction */}
      <div className={styles.infoBox}>
        <p>
          <strong>Envoi d'emails depuis votre propre domaine</strong>
        </p>
        <p>
          Configurez l'adresse email depuis laquelle vos factures seront envoyées. Nous utilisons{' '}
          <strong>Brevo</strong> comme service d'envoi pour garantir une excellente délivrabilité.
        </p>
        <p>
          <strong>Avantages :</strong> Pas de mot de passe à gérer, meilleure délivrabilité, envoi
          garanti.
        </p>
      </div>

      {/* Formulaire de configuration */}
      <form onSubmit={handleConfigure} className={styles.form}>
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label htmlFor="from_email">
              Adresse email d'envoi <span className={styles.required}>*</span>
            </label>
            <input
              type="email"
              id="from_email"
              value={fromEmail}
              onChange={(e) => setFromEmail(e.target.value)}
              placeholder="noreply@entreprise.ch"
              disabled={domainVerified}
              required
            />
            <small>L'adresse depuis laquelle vos emails seront envoyés</small>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="from_name">
              Nom d'expéditeur <span className={styles.required}>*</span>
            </label>
            <input
              type="text"
              id="from_name"
              value={fromName}
              onChange={(e) => setFromName(e.target.value)}
              placeholder="Votre Entreprise SA"
              disabled={domainVerified}
              maxLength={100}
              required
            />
            <small>Le nom qui apparaîtra comme expéditeur dans les emails</small>
          </div>
        </div>

        <button
          type="submit"
          className={styles.primaryButton}
          disabled={configuring || domainVerified}
        >
          {configuring ? (
            <span className={styles.buttonWithIcon}>
              <FiLoader className={styles.spinnerInline} /> Configuration...
            </span>
          ) : config?.configured ? (
            'Mettre à jour'
          ) : (
            'Configurer'
          )}
        </button>
      </form>

      {/* Section DNS Records */}
      {dnsRecords && !domainVerified && (
        <div className={styles.dnsSection}>
          <h3 className={styles.sectionTitleWithIcon}>
            <FiClipboard /> Étape suivante : Configurer les enregistrements DNS
          </h3>

          <div className={styles.warningBox}>
            <p>
              <strong>Configuration DNS requise</strong>
            </p>
            <p>
              Pour envoyer des emails depuis votre domaine, vous devez ajouter ces enregistrements
              DNS chez votre hébergeur (ex: Infomaniak, OVH, GoDaddy).
            </p>
            <p>
              <strong>Délai :</strong> La propagation DNS peut prendre de 15 minutes à 24 heures.
              Une fois ajoutés, cliquez sur "Vérifier" ci-dessous.
            </p>
          </div>

          {/* SPF Record */}
          <div className={styles.dnsRecord}>
            <div className={styles.dnsRecordHeader}>
              <span className={styles.tagBlue}>SPF (TXT)</span>
              <strong>Enregistrement SPF</strong>
            </div>
            <div className={styles.dnsRecordContent}>
              <textarea
                value={dnsRecords.spf}
                readOnly
                rows={3}
                className={styles.dnsRecordValue}
              />
              <button
                type="button"
                onClick={() => copyToClipboard(dnsRecords.spf, 'SPF')}
                className={styles.secondaryButton}
              >
                <span className={styles.buttonWithIcon}>
                  <FiClipboard /> Copier
                </span>
              </button>
            </div>
          </div>

          {/* DKIM Record */}
          <div className={styles.dnsRecord}>
            <div className={styles.dnsRecordHeader}>
              <span className={styles.tagGreen}>DKIM (TXT)</span>
              <strong>Enregistrement DKIM</strong>
            </div>
            <div className={styles.dnsRecordContent}>
              <textarea
                value={dnsRecords.dkim}
                readOnly
                rows={4}
                className={styles.dnsRecordValue}
              />
              <button
                type="button"
                onClick={() => copyToClipboard(dnsRecords.dkim, 'DKIM')}
                className={styles.secondaryButton}
              >
                <span className={styles.buttonWithIcon}>
                  <FiClipboard /> Copier
                </span>
              </button>
            </div>
          </div>

          {/* Bouton de vérification */}
          <div className={styles.verifyButtonWrap}>
            <button
              type="button"
              onClick={handleVerify}
              disabled={verifying}
              className={styles.primaryButton}
            >
              {verifying ? (
                <span className={styles.buttonWithIcon}>
                  <FiLoader className={styles.spinnerInline} /> Vérification...
                </span>
              ) : (
                <span className={styles.buttonWithIcon}>
                  <FiRefreshCw /> Vérifier la configuration DNS
                </span>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Domaine vérifié */}
      {domainVerified && (
        <div className={styles.successBox}>
          <p>
            <strong className={styles.badgeWithIcon}>
              <FiCheck /> Domaine vérifié avec succès !
            </strong>
          </p>
          <p>
            Vous pouvez maintenant envoyer des emails depuis votre domaine. Vos factures seront
            envoyées depuis l'adresse configurée.
          </p>
        </div>
      )}

      {/* Bouton de diagnostic (visible uniquement si domaine configuré mais pas vérifié) */}
      {config?.configured && !domainVerified && (
        <div className={styles.diagnosticWrap}>
          <p className={styles.diagnosticWrapTitle}>
            <strong className={styles.sectionTitleWithIcon}>
              <FiSearch /> Outil de diagnostic
            </strong>
          </p>
          <p className={styles.diagnosticWrapDesc}>
            Vérifiez le statut exact de votre domaine dans Brevo et obtenez des détails sur chaque enregistrement DNS.
          </p>
          <button
            type="button"
            onClick={handleDiagnostic}
            disabled={diagnosing}
            className={styles.secondaryButton}
          >
            {diagnosing ? (
              <span className={styles.buttonWithIcon}>
                <FiLoader className={styles.spinnerInline} /> Diagnostic en cours...
              </span>
            ) : (
              <span className={styles.buttonWithIcon}>
                <FiSearch /> Lancer le diagnostic complet
              </span>
            )}
          </button>
        </div>
      )}

      {/* Résultats du diagnostic */}
      {diagnostic && (
        <div className={`${diagnostic.success ? styles.infoBox : styles.warningBox} ${styles.diagnosticResultsBox}`}>
          <h4 className={styles.sectionTitleWithIcon}>
            <FiBarChart2 /> Résultats du diagnostic pour {diagnostic.domain}
          </h4>

          {/* Statut Brevo */}
          <div className={styles.diagnosticBlock}>
            <strong>Statut Brevo :</strong>
            <ul className={styles.diagnosticList}>
              <li className={styles.badgeWithIcon}>
                Vérifié : {diagnostic.brevo_status?.verified ? <><FiCheck /> Oui</> : <><FiX /> Non</>}
              </li>
              <li className={styles.badgeWithIcon}>
                Authentifié : {diagnostic.brevo_status?.authenticated ? <><FiCheck /> Oui</> : <><FiX /> Non</>}
              </li>
            </ul>
          </div>

          {/* Validation DNS */}
          <div className={styles.diagnosticBlock}>
            <strong>Validation DNS :</strong>
            <ul className={styles.diagnosticList}>
              <li className={styles.badgeWithIcon}>
                Brevo Code (SPF) : {diagnostic.dns_validation?.brevo_code_valid ? <><FiCheck /> Valide</> : <><FiX /> Non valide</>}
              </li>
              <li className={styles.badgeWithIcon}>
                DKIM 1 : {diagnostic.dns_validation?.dkim1_valid ? <><FiCheck /> Valide</> : <><FiX /> Non valide</>}
              </li>
              <li className={styles.badgeWithIcon}>
                DKIM 2 : {diagnostic.dns_validation?.dkim2_valid ? <><FiCheck /> Valide</> : <><FiX /> Non valide</>}
              </li>
            </ul>
          </div>

          {/* Détails des enregistrements */}
          <details className={styles.diagnosticDetails}>
            <summary className={styles.diagnosticDetailsSummary}>
              <span className={styles.sectionTitleWithIcon}>
                <FiClipboard /> Voir les détails des enregistrements DNS
              </span>
            </summary>
            <div className={styles.diagnosticDetailsContent}>
              <div className={styles.diagnosticDetailItem}>
                <strong>Brevo Code (SPF) :</strong>
                <div>Hôte : {diagnostic.dns_records?.brevo_code?.host}</div>
                <div>Valeur : {diagnostic.dns_records?.brevo_code?.value}</div>
                <div className={styles.badgeWithIcon}>Statut : {diagnostic.dns_records?.brevo_code?.is_valid ? <><FiCheck /> Valide</> : <><FiX /> Invalide</>}</div>
              </div>
              <div className={styles.diagnosticDetailItem}>
                <strong>DKIM 1 :</strong>
                <div>Hôte : {diagnostic.dns_records?.dkim1?.host}</div>
                <div>Valeur : {diagnostic.dns_records?.dkim1?.value}</div>
                <div className={styles.badgeWithIcon}>Statut : {diagnostic.dns_records?.dkim1?.is_valid ? <><FiCheck /> Valide</> : <><FiX /> Invalide</>}</div>
              </div>
              <div className={styles.diagnosticDetailItem}>
                <strong>DKIM 2 :</strong>
                <div>Hôte : {diagnostic.dns_records?.dkim2?.host}</div>
                <div>Valeur : {diagnostic.dns_records?.dkim2?.value}</div>
                <div className={styles.badgeWithIcon}>Statut : {diagnostic.dns_records?.dkim2?.is_valid ? <><FiCheck /> Valide</> : <><FiX /> Invalide</>}</div>
              </div>
            </div>
          </details>

          {/* Message d'action */}
          <div className={styles.diagnosticActionBox}>
            <strong className={styles.sectionTitleWithIcon}>
              <FiInfo /> Prochaine étape :
            </strong>
            <p>{diagnostic.message}</p>

            {!diagnostic.dns_validation?.all_valid && (
              <p className={styles.diagnosticActionDanger}>
                <strong className={styles.badgeWithIcon}>
                  <FiAlertTriangle /> Action requise :
                </strong>{' '}
                Certains enregistrements DNS ne sont pas encore détectés par Brevo.
                Vérifiez que vous avez bien configuré TOUS les enregistrements (SPF + les 2 DKIM) et attendez quelques heures
                pour la propagation DNS.
              </p>
            )}

            {diagnostic.dns_validation?.all_valid && !diagnostic.brevo_status?.verified && (
              <p className={styles.diagnosticActionWarning}>
                <strong className={styles.badgeWithIcon}>
                  <FiCheck /> Tous les DNS sont valides !
                </strong>{' '}
                Brevo n'a pas encore marqué le domaine comme vérifié.
                Cela peut prendre jusqu'à 72h. Si le problème persiste après ce délai, contactez le support Brevo à
                support@brevo.com ou via le chat sur app.brevo.com
              </p>
            )}
          </div>

          {/* Réponse API complète (mode debug) */}
          {diagnostic.raw_response && (
            <details className={styles.diagnosticDebugDetails}>
              <summary className={styles.diagnosticDebugSummary}>
                <span className={styles.sectionTitleWithIcon}>
                  <FiTool /> Réponse API Brevo complète (debug)
                </span>
              </summary>
              <pre className={styles.diagnosticDebugPre}>
                {JSON.stringify(diagnostic.raw_response, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      {/* Aide */}
      <div className={styles.helpSection}>
        <h4 className={styles.sectionTitleWithIcon}>
          <FiHelpCircle /> Besoin d'aide ?
        </h4>
        <ul>
          <li>
            <strong>Où ajouter les enregistrements DNS ?</strong> Connectez-vous à l'interface de
            gestion de votre hébergeur (Infomaniak, OVH, GoDaddy, etc.) et accédez à la section
            "DNS" ou "Zone DNS".
          </li>
          <li>
            <strong>Type d'enregistrement :</strong> Les deux enregistrements sont de type "TXT".
          </li>
          <li>
            <strong>Combien de temps attendre ?</strong> Généralement 15-60 minutes, mais cela peut
            prendre jusqu'à 24h selon votre hébergeur.
          </li>
          <li>
            <strong>Vérification échouée ?</strong> Vérifiez que vous avez bien copié l'intégralité
            des enregistrements (sans espaces en trop) et attendez quelques heures avant de
            réessayer.
          </li>
        </ul>
      </div>
    </div>
  );
};

export default EmailConfigSection;
