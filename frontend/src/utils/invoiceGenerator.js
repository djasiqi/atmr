import { PDFDocument, rgb, StandardFonts } from "pdf-lib";
import logo from "../assets/images/logo.png";

/**
 * Génère la première page de la facture.
 */
export async function generateInvoicePDF(invoiceData) {
  const pdfDoc = await PDFDocument.create();
  const page = pdfDoc.addPage([595, 842]); // Format A4
  const { width, height } = page.getSize();
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);

  // ✅ Génération d'un numéro de facture unique (À DÉPLACER AVANT UTILISATION)
  const invoiceNumber = `F${new Date().getFullYear()}-${(
    new Date().getMonth() + 1
  )
    .toString()
    .padStart(2, "0")}-${Math.floor(1000 + Math.random() * 9000)}`;

  // ✅ Ajout du numéro de facture dans la facture
  page.drawText(`Facture N°: ${invoiceNumber}`, {
    x: 50,
    y: height - 140,
    size: 14,
    font,
    color: rgb(0, 0, 0),
  });

  // ✅ Ajout du logo sans déformation
  try {
    const logoBytes = await fetch(logo).then((res) => res.arrayBuffer());
    const logoImage = await pdfDoc.embedPng(logoBytes);
    const scaleFactor = 0.09; // 📏 Ajuste la taille proportionnellement
    const scaledWidth = logoImage.width * scaleFactor;
    const scaledHeight = logoImage.height * scaleFactor;

    page.drawImage(logoImage, {
      x: 50,
      y: height - 80, // 📌 Ajusté pour ne pas trop descendre
      width: scaledWidth,
      height: scaledHeight,
    });
  } catch (error) {
    console.warn("⚠️ Impossible de charger le logo.");
  }

  // ✅ Coordonnées de l'entreprise
  page.drawText("Emmenez-moi Sàrl", {
    x: 350,
    y: height - 50,
    size: 12,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText("Route de Chevrens 145, 1247 Anières", {
    x: 350,
    y: height - 70,
    size: 10,
    font,
  });
  page.drawText("Téléphone : 079.291.50.37", {
    x: 350,
    y: height - 90,
    size: 10,
    font,
  });
  page.drawText("Email : contact@emmenez-moi.ch", {
    x: 350,
    y: height - 110,
    size: 10,
    font,
  });

  // ✅ Détails de la facture
  page.drawText(`Facture N°: ${invoiceNumber}`, {
    x: 50,
    y: height - 140,
    size: 14,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText(`Date: ${new Date().toLocaleDateString("fr-FR")}`, {
    x: 50,
    y: height - 160,
    size: 12,
    font,
  });

  // ✅ Informations du client
  page.drawText(
    `Client : ${invoiceData.client.firstName} ${invoiceData.client.lastName}`,
    { x: 50, y: height - 190, size: 12, font }
  );
  page.drawText(
    `Adresse : ${invoiceData.client.address}, ${invoiceData.client.zipCode} ${invoiceData.client.city}`,
    { x: 50, y: height - 210, size: 12, font }
  );

  // ✅ Tableau des trajets
  let tableStartY = height - 250;
  page.drawText("Date", { x: 50, y: tableStartY, size: 12, font });
  page.drawText("Départ", { x: 150, y: tableStartY, size: 12, font });
  page.drawText("Arrivée", { x: 300, y: tableStartY, size: 12, font });
  page.drawText("Montant (CHF)", { x: 450, y: tableStartY, size: 12, font });

  page.drawLine({
    start: { x: 50, y: tableStartY - 5 },
    end: { x: 550, y: tableStartY - 5 },
    thickness: 1,
  });

  let totalAmount = 0;

  invoiceData.bookings.forEach((booking) => {
    tableStartY -= 20;

    // ✅ Vérification du montant
    const amount = parseFloat(booking.amount || 0).toFixed(2);
    totalAmount += parseFloat(amount);

    page.drawText(
      new Date(booking.scheduled_time).toLocaleDateString("fr-FR"),
      { x: 50, y: tableStartY, size: 10, font }
    );
    const formatAddress = (address) => {
      return address
        .replace(/, Suisse$/, "") // Supprime "Suisse"
        .replace(/,\s*(\d{4,} [^,]+)/, "\n$1"); // Déplace le code postal + ville sur une nouvelle ligne
    };
    const lineSpacing = 12; // 📏 Augmente légèrement l’espace entre chaque ligne (de 18 à 22)
    const rowSpacing = 4; // 📏 Ajoute plus d’espace entre chaque ligne du tableau

    page.drawText(formatAddress(booking.pickup_location || "Inconnu"), {
      x: 150,
      y: tableStartY,
      size: 10,
      font,
      lineHeight: lineSpacing,
    });

    tableStartY -= rowSpacing; // 📏 Ajoute plus d’espace après chaque ligne

    page.drawText(formatAddress(booking.dropoff_location || "Inconnu"), {
      x: 300,
      y: tableStartY,
      size: 10,
      font,
      lineHeight: lineSpacing,
    });

    // ✅ Décale chaque ligne plus bas après l’affichage de l’adresse arrivée
    tableStartY -= rowSpacing;

    page.drawText(`${amount}`, { x: 450, y: tableStartY, size: 10, font });
  });

  // ✅ Affichage du montant total aligné à la colonne Montant
  tableStartY -= 30;
  page.drawText("TOTAL :", {
    x: 350,
    y: tableStartY,
    size: 12,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText(`${totalAmount.toFixed(2)} CHF`, {
    x: 450,
    y: tableStartY,
    size: 12,
    font,
  });

  // ✅ Texte centré pour les conditions de paiement
  tableStartY -= 50;
  page.drawText(
    "En votre aimable règlement sous 10 jours avec nos remerciements anticipés.",
    {
      x: width / 2 - 180,
      y: tableStartY,
      size: 10,
      font,
    }
  );
  tableStartY -= 15;
  page.drawText(
    "En cas de retard, des frais de rappel de 15 CHF seront facturés.",
    {
      x: width / 2 - 150,
      y: tableStartY,
      size: 10,
      font,
    }
  );

  // ✅ Générer le PDF
  return await pdfDoc.save();
}
