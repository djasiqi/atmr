import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { FiX, FiFileText, FiCalendar, FiCreditCard, FiHash, FiChevronDown } from 'react-icons/fi';
import styles from './PaymentModal.module.css';
import InlineDatePicker from '../../../../../components/ui/InlineDatePicker';

function MethodChipDropdown({ options, value, onChange }) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef(null);
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });

  useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: Math.max(r.width, 180) });
  }, []);

  useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const selected = options.find((o) => o.value === value);

  return (
    <div className={styles.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${styles.chipBtn} ${value ? styles.chipBtnActive : ''}`}
        onClick={() => setOpen((p) => !p)}
      >
        <span className={styles.chipText}>{selected?.label || 'Sélectionner'}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={styles.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10001 }}
        >
          {options.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${styles.chipOption} ${o.value === value ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

const PaymentModal = ({ open, invoice, onClose, onPayment }) => {
  const [formData, setFormData] = useState({
    amount: '',
    paid_at: new Date().toISOString().split('T')[0],
    method: 'bank_transfer',
    reference: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const amountRef = useRef(null);

  const paymentMethods = [
    { value: 'bank_transfer', label: 'Virement bancaire' },
    { value: 'cash', label: 'Especes' },
    { value: 'card', label: 'Carte bancaire' },
    { value: 'adjustment', label: 'Ajustement' },
  ];

  useEffect(() => {
    if (open && invoice) {
      setFormData({
        amount: invoice.balance_due > 0 ? invoice.balance_due.toFixed(2) : '',
        paid_at: new Date().toISOString().split('T')[0],
        method: 'bank_transfer',
        reference: '',
      });
      setError(null);
      setTimeout(() => amountRef.current?.select(), 100);
    }
  }, [open, invoice]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e) => {
      if (e.key === 'Escape') handleClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const getClientName = () => {
    if (!invoice) return '';
    if (invoice.billed_to_company_id && invoice.billed_to_company) {
      return invoice.billed_to_company.name || 'Clinique';
    }
    if (invoice.client) {
      return (
        invoice.client.institution_name ||
        `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
        invoice.client.username ||
        ''
      );
    }
    return 'Client inconnu';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const amt = parseFloat(formData.amount);

    if (!amt || amt <= 0) {
      setError('Le montant doit etre positif');
      return;
    }
    if (amt > invoice.balance_due) {
      setError('Le montant ne peut pas depasser le solde du');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      await onPayment(invoice.id, {
        ...formData,
        amount: amt,
        paid_at: new Date(formData.paid_at).toISOString(),
      });
    } catch (err) {
      setError(err.message || "Erreur lors de l'enregistrement du paiement");
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    if (error) setError(null);
  };

  const handleClose = () => {
    setFormData({
      amount: '',
      paid_at: new Date().toISOString().split('T')[0],
      method: 'bank_transfer',
      reference: '',
    });
    setError(null);
    onClose();
  };

  const handlePayFullBalance = () => {
    setFormData((prev) => ({ ...prev, amount: invoice.balance_due.toFixed(2) }));
    if (error) setError(null);
  };

  if (!open || !invoice) return null;

  const paidPercent = invoice.total_amount > 0
    ? Math.round((invoice.amount_paid / invoice.total_amount) * 100)
    : 0;

  return (
    <div className={styles.overlay} onClick={handleClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerTitle}>
            <FiFileText size={18} className={styles.headerIcon} />
            <h2>Enregistrer un paiement</h2>
          </div>
          <button className={styles.closeBtn} onClick={handleClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>

        {/* Invoice summary card */}
        <div className={styles.invoiceCard}>
          <div className={styles.invoiceCardHeader}>
            <span className={styles.invoiceNumber}>{invoice.invoice_number}</span>
            <span className={styles.clientName}>{getClientName()}</span>
          </div>
          <div className={styles.invoiceAmounts}>
            <div className={styles.amountRow}>
              <span className={styles.amountLabel}>Montant total</span>
              <span className={styles.amountValue}>{invoice.total_amount.toFixed(2)} CHF</span>
            </div>
            <div className={styles.amountRow}>
              <span className={styles.amountLabel}>Deja paye</span>
              <span className={styles.amountValue}>{invoice.amount_paid.toFixed(2)} CHF</span>
            </div>
            {/* Progress bar */}
            <div className={styles.progressBar}>
              <div
                className={styles.progressFill}
                style={{ width: `${Math.min(paidPercent, 100)}%` }}
              />
            </div>
            <div className={styles.balanceRow}>
              <span className={styles.balanceLabel}>Solde du</span>
              <span className={styles.balanceValue}>{invoice.balance_due.toFixed(2)} CHF</span>
            </div>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="payment-amount" className={styles.label}>
              Montant du paiement
            </label>
            <div className={styles.amountInputWrapper}>
              <input
                ref={amountRef}
                type="number"
                id="payment-amount"
                name="amount"
                value={formData.amount}
                onChange={handleInputChange}
                className={styles.input}
                step="0.01"
                min="0.01"
                max={invoice.balance_due}
                placeholder="0.00"
                required
              />
              <span className={styles.inputSuffix}>CHF</span>
              {parseFloat(formData.amount) !== invoice.balance_due && invoice.balance_due > 0 && (
                <button
                  type="button"
                  className={styles.payFullBtn}
                  onClick={handlePayFullBalance}
                  title="Payer le solde complet"
                >
                  Tout payer
                </button>
              )}
            </div>
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label className={styles.label}>
                <FiCalendar size={13} className={styles.labelIcon} />
                Date du paiement
              </label>
              <InlineDatePicker
                value={formData.paid_at || ''}
                onChange={(v) => handleInputChange({ target: { name: 'paid_at', value: v } })}
              />
            </div>

            <div className={styles.formGroup}>
              <label className={styles.label}>
                <FiCreditCard size={13} className={styles.labelIcon} />
                Mode de paiement
              </label>
              <MethodChipDropdown
                options={paymentMethods}
                value={formData.method}
                onChange={(v) => handleInputChange({ target: { name: 'method', value: v } })}
              />
            </div>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="payment-reference" className={styles.label}>
              <FiHash size={13} className={styles.labelIcon} />
              Reference
              <span className={styles.labelOptional}>optionnel</span>
            </label>
            <input
              type="text"
              id="payment-reference"
              name="reference"
              value={formData.reference}
              onChange={handleInputChange}
              className={styles.input}
              placeholder="Reference bancaire, n. transaction..."
            />
          </div>

          {error && <div className={styles.error}>{error}</div>}

          <div className={styles.footer}>
            <button
              type="button"
              className={styles.cancelBtn}
              onClick={handleClose}
              disabled={loading}
            >
              Annuler
            </button>
            <button type="submit" className={styles.submitBtn} disabled={loading}>
              {loading ? 'Enregistrement...' : 'Enregistrer le paiement'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PaymentModal;
