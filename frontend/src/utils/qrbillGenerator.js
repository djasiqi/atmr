import { PDFDocument } from "pdf-lib";
import { SwissQRBill } from "swissqrbill/svg";

/**
 * Génère un QR-Bill en SVG et le convertit en PNG avant de l'attacher à un PDF.
 * @param {Object} invoiceData - Données de facturation
 * @returns {Promise<Uint8Array>} - PDF contenant le QR-Bill
 */
export async function generateQRBillPDF(invoiceData) {
  try {
    console.log("📄 Début de la génération du QR-Bill en SVG...");

    // ✅ Définition des données du QR-Bill
    const qrBillData = {
      amount: parseFloat(invoiceData.totalPrice),
      currency: "CHF",
      creditor: {
        account: "CH65 0900 0000 1526 3128 9", // ✅ IBAN standard (PAS QR-IBAN)
        name: "Emmenez-moi Sàrl",
        address: "Route de Chevrens 145",
        zip: 1247,
        city: "Anières",
        country: "CH",
      },
      debtor: {
        name: `${invoiceData.client.firstName} ${invoiceData.client.lastName}`,
        address: invoiceData.client.address || "Adresse inconnue",
        zip: invoiceData.client.zipCode || "0000",
        city: invoiceData.client.city || "Ville inconnue",
        country: "CH",
      },
      unstructuredMessage: `Facture ${invoiceData.invoiceNumber} - Paiement des services de transport`, // ✅ Message libre
      language: "fr", // ✅ QR-Bill en français
    };

    // ✅ Générer le QR-Bill en SVG
    const qrBillSvg = new SwissQRBill(qrBillData).toString();

    // ✅ Convertir le SVG en un BLOB URL
    const blob = new Blob([qrBillSvg], { type: "image/svg+xml" });
    const qrSvgUrl = URL.createObjectURL(blob);

    // ✅ Chargement du QR-Bill dans un Canvas avec meilleure qualité
    const qrPngBytes = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = async () => {
        const scaleFactor = 4; // 🔥 Augmentation de la qualité (4x plus grand)

        const canvas = document.createElement("canvas");
        canvas.width = 595 * scaleFactor; // 🔥 Largeur A4 en haute résolution
        canvas.height = 300 * scaleFactor;
        const ctx = canvas.getContext("2d");

        // 🔥 Amélioration du rendu
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
          if (blob) {
            resolve(await blob.arrayBuffer());
          } else {
            reject(new Error("Échec de la conversion SVG → PNG"));
          }
        }, "image/png");
      };

      img.onerror = (error) => {
        console.error("❌ Erreur lors du chargement du QR Code SVG :", error);
        reject(new Error("Échec du chargement de l'image SVG du QR Bill."));
      };

      img.src = qrSvgUrl; // ✅ Utilisation de l'URL Blob au lieu de `btoa()`
    });

    // ✅ Créer un PDF contenant l'image PNG du QR-Bill
    const pdfDoc = await PDFDocument.create();
    const page = pdfDoc.addPage([595, 842]); // A4

    const qrImage = await pdfDoc.embedPng(qrPngBytes);
    page.drawImage(qrImage, {
      x: 0, // Aligner à gauche
      y: 0, // Placer en bas de la page
      width: 595, // Largeur complète de la page
      height: 300,
    });

    console.log(
      "✅ QR-Bill converti en PNG avec haute qualité et ajouté au PDF !"
    );
    return await pdfDoc.save(); // ✅ Retourne un PDF bien formé
  } catch (error) {
    console.error("❌ Erreur lors de la génération du QR-Bill :", error);
    throw error;
  }
}
