import { PDFDocument } from "pdf-lib";
import { saveAs } from "file-saver";
import { generateInvoicePDF } from "./invoiceGenerator";
import { generateQRBillPDF } from "./qrbillGenerator";

export async function mergeInvoiceAndQRBill(invoiceData) {
  try {
    console.log("📄 Début de la fusion des PDF...");

    // ✅ Générer les PDF des deux parties
    const invoicePDFBytes = await generateInvoicePDF(invoiceData);
    const qrBillPDFBytes = await generateQRBillPDF(invoiceData);

    // ✅ Charger les fichiers PDF avec `pdf-lib`
    const invoiceDoc = await PDFDocument.load(invoicePDFBytes);
    const qrBillDoc = await PDFDocument.load(qrBillPDFBytes);

    // ✅ Création d'un nouveau PDF fusionné
    const finalDoc = await PDFDocument.create();

    // ✅ Copier les pages de la facture dans le document final
    const invoicePages = await finalDoc.copyPages(
      invoiceDoc,
      invoiceDoc.getPageIndices()
    );
    invoicePages.forEach((page) => finalDoc.addPage(page));

    // ✅ Copier les pages du QR-Bill et les ajouter en tant que deuxième page
    const qrBillPages = await finalDoc.copyPages(
      qrBillDoc,
      qrBillDoc.getPageIndices()
    );
    qrBillPages.forEach((page) => finalDoc.addPage(page));

    console.log("✅ Fusion des PDF réussie avec QR-Bill bien positionné !");

    // ✅ Génération et téléchargement du PDF final
    const pdfBytesFinal = await finalDoc.save();
    const blob = new Blob([pdfBytesFinal], { type: "application/pdf" });
    saveAs(blob, `Facture_${invoiceData.client.lastName}.pdf`);

    console.log("✅ Facture complète avec QR-Bill générée avec succès !");
  } catch (error) {
    console.error("❌ Erreur lors de la fusion des PDF :", error);
  }
}
