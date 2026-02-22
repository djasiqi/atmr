import ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';

const STATUS_LABELS = {
  pending: 'En attente',
  accepted: 'Acceptee',
  assigned: 'Assignee',
  en_route: 'En route',
  in_progress: 'En cours',
  completed: 'Terminee',
  return_completed: 'Retour termine',
  canceled: 'Annulee',
  cancelled: 'Annulee',
  rejected: 'Refusee',
  no_show: 'Non presente',
};

const BRAND_COLOR = '00796B';
const HEADER_BG = '00796B';
const HEADER_FONT = 'FFFFFF';
const STRIPE_BG = 'F0FDFA';
const BORDER_COLOR = 'E2E8F0';

function formatDateExcel(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()}`;
  } catch {
    return '';
  }
}

function formatTimeExcel(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return '';
    const pad = (n) => String(n).padStart(2, '0');
    const h = d.getHours();
    const m = d.getMinutes();
    if (h === 0 && m === 0) return 'A confirmer';
    return `${pad(h)}:${pad(m)}`;
  } catch {
    return '';
  }
}

/**
 * Exporte les reservations en fichier Excel formate professionnellement.
 * @param {Array} reservations - Liste des reservations
 * @param {Object} options - Options d'export
 * @param {string} options.companyName - Nom de l'entreprise
 * @param {string} options.periodLabel - Label de la periode (ex: "Toutes", "17.02.2026")
 * @param {Object} options.stats - Stats KPI {total, inProgress, completed, revenue}
 */
export async function exportReservationsExcel(reservations, options = {}) {
  const {
    companyName = 'Entreprise',
    periodLabel = '',
    stats = {},
  } = options;

  const wb = new ExcelJS.Workbook();
  wb.creator = 'Lirie - Plateforme Transport Medical';
  wb.created = new Date();

  // --- Feuille principale : Reservations ---
  const ws = wb.addWorksheet('Reservations', {
    properties: { defaultRowHeight: 22 },
    views: [{ state: 'frozen', ySplit: 5 }],
  });

  // Colonnes avec largeurs optimisees
  ws.columns = [
    { key: 'id', width: 10 },
    { key: 'date', width: 14 },
    { key: 'heure', width: 10 },
    { key: 'client', width: 26 },
    { key: 'institution', width: 22 },
    { key: 'depart', width: 38 },
    { key: 'arrivee', width: 38 },
    { key: 'chauffeur', width: 22 },
    { key: 'montant', width: 14 },
    { key: 'statut', width: 16 },
    { key: 'type', width: 14 },
  ];

  // --- En-tete du rapport ---
  const titleRow = ws.addRow([`Rapport des reservations - ${companyName}`]);
  titleRow.height = 32;
  titleRow.getCell(1).font = { size: 16, bold: true, color: { argb: BRAND_COLOR } };
  ws.mergeCells('A1:K1');
  titleRow.getCell(1).alignment = { vertical: 'middle' };

  const subtitleParts = [`Genere le ${new Date().toLocaleDateString('fr-CH')}`];
  if (periodLabel && periodLabel !== 'Toutes') {
    subtitleParts.push(`Periode : ${periodLabel}`);
  }
  const subtitleRow = ws.addRow([subtitleParts.join(' | ')]);
  subtitleRow.getCell(1).font = { size: 10, italic: true, color: { argb: '64748B' } };
  ws.mergeCells('A2:K2');

  // --- Ligne KPI resume ---
  const kpiParts = [];
  if (stats.total !== undefined) kpiParts.push(`Total : ${stats.total}`);
  if (stats.inProgress !== undefined) kpiParts.push(`En cours : ${stats.inProgress}`);
  if (stats.completed !== undefined) kpiParts.push(`Terminees : ${stats.completed}`);
  if (stats.revenue !== undefined) kpiParts.push(`Revenus : ${Number(stats.revenue).toFixed(2)} CHF`);

  if (kpiParts.length > 0) {
    const kpiRow = ws.addRow([kpiParts.join('  |  ')]);
    kpiRow.getCell(1).font = { size: 10, bold: true, color: { argb: '334155' } };
    ws.mergeCells('A3:K3');
  } else {
    ws.addRow([]);
  }

  // Ligne vide separatrice
  ws.addRow([]);

  // --- En-tetes colonnes ---
  const headers = ['N', 'Date', 'Heure', 'Client', 'Institution', 'Depart', 'Arrivee', 'Chauffeur', 'Montant (CHF)', 'Statut', 'Type'];
  const headerRow = ws.addRow(headers);
  headerRow.height = 28;

  headerRow.eachCell((cell) => {
    cell.font = { bold: true, size: 11, color: { argb: HEADER_FONT } };
    cell.fill = {
      type: 'pattern',
      pattern: 'solid',
      fgColor: { argb: HEADER_BG },
    };
    cell.alignment = { vertical: 'middle', horizontal: 'center', wrapText: true };
    cell.border = {
      bottom: { style: 'medium', color: { argb: BRAND_COLOR } },
    };
  });

  // --- Donnees ---
  reservations.forEach((r, index) => {
    const status = r.status?.toLowerCase() || '';
    const clientName = r.client?.full_name || r.client_name || '';
    const institution = r.client?.institution_name || r.institution_name || '';
    const driverName = r.driver?.full_name || r.driver_name || '';
    const montant = Number(r.amount || 0);
    const type = r.is_return ? 'Retour' : 'Aller';

    const row = ws.addRow([
      r.id || '',
      formatDateExcel(r.scheduled_time),
      formatTimeExcel(r.scheduled_time),
      clientName,
      institution,
      r.pickup_location || '',
      r.dropoff_location || '',
      driverName,
      montant,
      STATUS_LABELS[status] || status,
      type,
    ]);

    row.height = 24;

    // Formatage alternatif (zebra)
    const isEven = index % 2 === 0;
    row.eachCell((cell, colNum) => {
      cell.font = { size: 10.5 };
      cell.alignment = { vertical: 'middle', wrapText: false };
      cell.border = {
        bottom: { style: 'thin', color: { argb: BORDER_COLOR } },
      };

      if (isEven) {
        cell.fill = {
          type: 'pattern',
          pattern: 'solid',
          fgColor: { argb: STRIPE_BG },
        };
      }

      // Formatage specifique par colonne
      if (colNum === 1) {
        cell.alignment = { ...cell.alignment, horizontal: 'center' };
        cell.font = { ...cell.font, color: { argb: '64748B' } };
      }
      if (colNum === 2 || colNum === 3) {
        cell.alignment = { ...cell.alignment, horizontal: 'center' };
      }
      if (colNum === 4) {
        cell.font = { ...cell.font, bold: true };
      }
      if (colNum === 5) {
        cell.font = { ...cell.font, color: { argb: '64748B' }, italic: true };
      }
      if (colNum === 9) {
        cell.numFmt = '#,##0.00';
        cell.alignment = { ...cell.alignment, horizontal: 'right' };
        cell.font = { ...cell.font, bold: true };
      }
      if (colNum === 10) {
        cell.alignment = { ...cell.alignment, horizontal: 'center' };
        const statusColor = getStatusColor(status);
        if (statusColor) {
          cell.font = { ...cell.font, bold: true, color: { argb: statusColor } };
        }
      }
      if (colNum === 11) {
        cell.alignment = { ...cell.alignment, horizontal: 'center' };
      }
    });
  });

  // --- Ligne de total ---
  const totalRowNum = ws.lastRow.number + 1;
  const totalRow = ws.addRow([]);
  totalRow.height = 28;

  const totalLabelCell = totalRow.getCell(8);
  totalLabelCell.value = 'TOTAL';
  totalLabelCell.font = { bold: true, size: 11, color: { argb: BRAND_COLOR } };
  totalLabelCell.alignment = { horizontal: 'right', vertical: 'middle' };

  const totalValueCell = totalRow.getCell(9);
  const firstDataRow = 6;
  const lastDataRow = totalRowNum - 1;
  totalValueCell.value = { formula: `SUM(I${firstDataRow}:I${lastDataRow})` };
  totalValueCell.numFmt = '#,##0.00';
  totalValueCell.font = { bold: true, size: 12, color: { argb: BRAND_COLOR } };
  totalValueCell.alignment = { horizontal: 'right', vertical: 'middle' };
  totalValueCell.border = {
    top: { style: 'double', color: { argb: BRAND_COLOR } },
    bottom: { style: 'double', color: { argb: BRAND_COLOR } },
  };

  const totalCountCell = totalRow.getCell(10);
  totalCountCell.value = `${reservations.length} reservations`;
  totalCountCell.font = { size: 10, italic: true, color: { argb: '64748B' } };
  totalCountCell.alignment = { horizontal: 'center', vertical: 'middle' };

  // --- Auto-filtre sur les en-tetes ---
  ws.autoFilter = {
    from: { row: 5, column: 1 },
    to: { row: 5, column: 11 },
  };

  // --- Pied de page ---
  ws.addRow([]);
  const footerRow = ws.addRow([`Exporte depuis Lirie - ${new Date().toLocaleString('fr-CH')}`]);
  footerRow.getCell(1).font = { size: 9, italic: true, color: { argb: '94A3B8' } };
  ws.mergeCells(`A${footerRow.number}:K${footerRow.number}`);

  // --- Mise en page impression ---
  ws.pageSetup = {
    orientation: 'landscape',
    fitToPage: true,
    fitToWidth: 1,
    fitToHeight: 0,
    paperSize: 9,
    margins: {
      left: 0.4, right: 0.4,
      top: 0.6, bottom: 0.6,
      header: 0.3, footer: 0.3,
    },
  };

  ws.headerFooter = {
    oddHeader: `&C&B${companyName} - Reservations`,
    oddFooter: '&LLirie&CPage &P / &N&R&D',
  };

  // --- Generation et telechargement ---
  const buffer = await wb.xlsx.writeBuffer();
  const dateStr = new Date().toISOString().split('T')[0].replace(/-/g, '');
  const fileName = `reservations_${companyName.toLowerCase().replace(/\s+/g, '_')}_${dateStr}.xlsx`;

  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });
  saveAs(blob, fileName);

  return fileName;
}

function getStatusColor(status) {
  switch (status) {
    case 'pending': return 'D97706';
    case 'accepted': return '2563EB';
    case 'assigned': return '7C3AED';
    case 'en_route': return '0891B2';
    case 'in_progress': return '0891B2';
    case 'completed':
    case 'return_completed': return '059669';
    case 'canceled':
    case 'cancelled': return 'DC2626';
    case 'rejected': return 'DC2626';
    case 'no_show': return '9333EA';
    default: return null;
  }
}
