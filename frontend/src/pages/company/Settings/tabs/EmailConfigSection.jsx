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

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: 'EmailConfigSection.jsx:70',
          message: 'Received result from setupEmailDomain',
          data: {
            result: result,
            dns_records_in_result: result?.dns_records,
            dns_records_type: typeof result?.dns_records,
          },
          timestamp: Date.now(),
          sessionId: 'debug-session',
          runId: 'initial',
          hypothesisId: 'C',
        }),
      }).catch(() => {});
      // #endregion

      if (result.success) {
        setMessage(result.message);
        setDnsRecords(result.dns_records);

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            location: 'EmailConfigSection.jsx:78',
            message: 'After setDnsRecords',
            data: {
              dns_records_value: result.dns_records,
              dns_records_spf: result.dns_records?.spf,
              dns_records_dkim: result.dns_records?.dkim,
            },
            timestamp: Date.now(),
            sessionId: 'debug-session',
            runId: 'initial',
            hypothesisId: 'D',
          }),
        }).catch(() => {});
        // #endregion

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
      return <span className={styles.badgeDefault}>⚪ Non configuré</span>;
    }
    if (domainVerified) {
      return <span className={styles.badgeSuccess}>✅ Vérifié</span>;
    }
    return <span className={styles.badgeWarning}>⏳ En attente de validation DNS</span>;
  };

  if (loading) {
    return (
      <div
        className={`${styles.emailConfigCard} ${compact ? styles.emailConfigCompact : ''}`}
      >
        {showHeader && <h2>📧 Configuration Email Transactionnel</h2>}
        {!showHeader && <div className={styles.headerCompact}>{getStatusBadge()}</div>}
        <div className={styles.loading}>
          <div className={styles.spinner}></div>
          <p>Chargement...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`${styles.emailConfigCard} ${compact ? styles.emailConfigCompact : ''}`}>
      {showHeader ? (
        <div className={styles.header}>
          <h2>📧 Configuration Email Transactionnel</h2>
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
          {configuring
            ? '⏳ Configuration...'
            : config?.configured
            ? 'Mettre à jour'
            : 'Configurer'}
        </button>
      </form>

      {/* Section DNS Records */}
      {dnsRecords && !domainVerified && (
        <div className={styles.dnsSection}>
          {/* #region agent log */}
          {fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              location: 'EmailConfigSection.jsx:228',
              message: 'Rendering DNS section',
              data: {
                dnsRecords: dnsRecords,
                dnsRecords_spf: dnsRecords?.spf,
                dnsRecords_dkim: dnsRecords?.dkim,
                dnsRecords_keys: Object.keys(dnsRecords || {}),
              },
              timestamp: Date.now(),
              sessionId: 'debug-session',
              runId: 'initial',
              hypothesisId: 'E',
            }),
          }).catch(() => {}) && null}
          {/* #endregion */}
          <h3>📋 Étape suivante : Configurer les enregistrements DNS</h3>

          <div className={styles.warningBox}>
            <p>
              <strong>Configuration DNS requise</strong>
            </p>
            <p>
              Pour envoyer des emails depuis votre domaine, vous devez ajouter ces enregistrements
              DNS chez votre hébergeur (ex: Infomaniak, OVH, GoDaddy).
            </p>
            <p>
              <strong>⏱ Délai :</strong> La propagation DNS peut prendre de 15 minutes à 24 heures.
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
                📋 Copier
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
                📋 Copier
              </button>
            </div>
          </div>

          {/* Bouton de vérification */}
          <div style={{ textAlign: 'center', marginTop: '24px' }}>
            <button
              type="button"
              onClick={handleVerify}
              disabled={verifying}
              className={styles.primaryButton}
            >
              {verifying ? '⏳ Vérification...' : '🔄 Vérifier la configuration DNS'}
            </button>
          </div>
        </div>
      )}

      {/* Domaine vérifié */}
      {domainVerified && (
        <div className={styles.successBox}>
          <p>
            <strong>✅ Domaine vérifié avec succès !</strong>
          </p>
          <p>
            Vous pouvez maintenant envoyer des emails depuis votre domaine. Vos factures seront
            envoyées depuis l'adresse configurée.
          </p>
        </div>
      )}

      {/* Bouton de diagnostic (visible uniquement si domaine configuré mais pas vérifié) */}
      {config?.configured && !domainVerified && (
        <div style={{ textAlign: 'center', marginTop: '24px', padding: '16px', background: '#f8f9fa', borderRadius: '8px' }}>
          <p style={{ marginBottom: '12px', color: '#666' }}>
            <strong>🔍 Outil de diagnostic</strong>
          </p>
          <p style={{ marginBottom: '16px', fontSize: '14px', color: '#666' }}>
            Vérifiez le statut exact de votre domaine dans Brevo et obtenez des détails sur chaque enregistrement DNS.
          </p>
          <button
            type="button"
            onClick={handleDiagnostic}
            disabled={diagnosing}
            className={styles.secondaryButton}
          >
            {diagnosing ? '⏳ Diagnostic en cours...' : '🔍 Lancer le diagnostic complet'}
          </button>
        </div>
      )}

      {/* Résultats du diagnostic */}
      {diagnostic && (
        <div className={diagnostic.success ? styles.infoBox : styles.warningBox} style={{ marginTop: '16px' }}>
          <h4>📊 Résultats du diagnostic pour {diagnostic.domain}</h4>
          
          {/* Statut Brevo */}
          <div style={{ marginTop: '16px' }}>
            <strong>Statut Brevo :</strong>
            <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
              <li>
                Vérifié : {diagnostic.brevo_status?.verified ? '✅ Oui' : '❌ Non'}
              </li>
              <li>
                Authentifié : {diagnostic.brevo_status?.authenticated ? '✅ Oui' : '❌ Non'}
              </li>
            </ul>
          </div>

          {/* Validation DNS */}
          <div style={{ marginTop: '16px' }}>
            <strong>Validation DNS :</strong>
            <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
              <li>
                Brevo Code (SPF) : {diagnostic.dns_validation?.brevo_code_valid ? '✅ Valide' : '❌ Non valide'}
              </li>
              <li>
                DKIM 1 : {diagnostic.dns_validation?.dkim1_valid ? '✅ Valide' : '❌ Non valide'}
              </li>
              <li>
                DKIM 2 : {diagnostic.dns_validation?.dkim2_valid ? '✅ Valide' : '❌ Non valide'}
              </li>
            </ul>
          </div>

          {/* Détails des enregistrements */}
          <details style={{ marginTop: '16px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
              📋 Voir les détails des enregistrements DNS
            </summary>
            <div style={{ marginTop: '12px', fontSize: '13px', fontFamily: 'monospace', background: '#f5f5f5', padding: '12px', borderRadius: '4px' }}>
              <div style={{ marginBottom: '12px' }}>
                <strong>Brevo Code (SPF) :</strong>
                <div>Hôte : {diagnostic.dns_records?.brevo_code?.host}</div>
                <div>Valeur : {diagnostic.dns_records?.brevo_code?.value}</div>
                <div>Statut : {diagnostic.dns_records?.brevo_code?.is_valid ? '✅' : '❌'}</div>
              </div>
              <div style={{ marginBottom: '12px' }}>
                <strong>DKIM 1 :</strong>
                <div>Hôte : {diagnostic.dns_records?.dkim1?.host}</div>
                <div>Valeur : {diagnostic.dns_records?.dkim1?.value}</div>
                <div>Statut : {diagnostic.dns_records?.dkim1?.is_valid ? '✅' : '❌'}</div>
              </div>
              <div>
                <strong>DKIM 2 :</strong>
                <div>Hôte : {diagnostic.dns_records?.dkim2?.host}</div>
                <div>Valeur : {diagnostic.dns_records?.dkim2?.value}</div>
                <div>Statut : {diagnostic.dns_records?.dkim2?.is_valid ? '✅' : '❌'}</div>
              </div>
            </div>
          </details>

          {/* Message d'action */}
          <div style={{ marginTop: '16px', padding: '12px', background: '#fff', border: '1px solid #ddd', borderRadius: '4px' }}>
            <strong>💡 Prochaine étape :</strong>
            <p style={{ marginTop: '8px' }}>{diagnostic.message}</p>
            
            {!diagnostic.dns_validation?.all_valid && (
              <p style={{ marginTop: '8px', color: '#d32f2f' }}>
                <strong>⚠️ Action requise :</strong> Certains enregistrements DNS ne sont pas encore détectés par Brevo. 
                Vérifiez que vous avez bien configuré TOUS les enregistrements (SPF + les 2 DKIM) et attendez quelques heures 
                pour la propagation DNS.
              </p>
            )}

            {diagnostic.dns_validation?.all_valid && !diagnostic.brevo_status?.verified && (
              <p style={{ marginTop: '8px', color: '#f57c00' }}>
                <strong>✅ Tous les DNS sont valides !</strong> Brevo n'a pas encore marqué le domaine comme vérifié. 
                Cela peut prendre jusqu'à 72h. Si le problème persiste après ce délai, contactez le support Brevo à 
                support@brevo.com ou via le chat sur app.brevo.com
              </p>
            )}
          </div>

          {/* Réponse API complète (mode debug) */}
          {diagnostic.raw_response && (
            <details style={{ marginTop: '16px' }}>
              <summary style={{ cursor: 'pointer', fontWeight: 'bold', color: '#666' }}>
                🔧 Réponse API Brevo complète (debug)
              </summary>
              <pre style={{ marginTop: '12px', fontSize: '11px', background: '#f5f5f5', padding: '12px', borderRadius: '4px', overflow: 'auto', maxHeight: '300px' }}>
                {JSON.stringify(diagnostic.raw_response, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      {/* Aide */}
      <div className={styles.helpSection}>
        <h4>❓ Besoin d'aide ?</h4>
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
