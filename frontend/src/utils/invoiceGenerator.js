import { PDFDocument, rgb, StandardFonts } from "pdf-lib";
import logo from "../assets/images/logo.png";

/**
 * Génère la première page de la facture avec les vraies données de l'entreprise et du client.
 */
export async function generateInvoicePDF(invoiceData) {
  const pdfDoc = await PDFDocument.create();
  const page = pdfDoc.addPage([595, 842]); // Format A4
  const { height } = page.getSize();
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);

  console.log("📄 Génération de la facture avec les données:", invoiceData);
  console.log("🏢 Données de l'entreprise:", invoiceData.company);
  console.log("🖼️ Logo URL reçu:", invoiceData.company?.logo_url);

  // ✅ Génération d'un numéro de facture unique
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

  // ✅ Ajout du logo de l'entreprise (si disponible)
  try {
    if (invoiceData.company?.logo_url) {
      console.log(
        "🖼️ Chargement du logo de l'entreprise:",
        invoiceData.company.logo_url
      );

      // Construire l'URL complète pour le logo
      let logoUrl = invoiceData.company.logo_url;
      if (logoUrl.startsWith("/uploads/")) {
        // Si c'est un chemin relatif, construire l'URL complète
        // Retirer le préfixe /api/ s'il est présent dans l'URL de base
        let baseUrl =
          process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";
        if (baseUrl.endsWith("/api")) {
          baseUrl = baseUrl.replace("/api", "");
        }
        logoUrl = `${baseUrl}${logoUrl}`;
      }

      console.log("🖼️ URL complète du logo:", logoUrl);

      const logoResponse = await fetch(logoUrl);
      if (!logoResponse.ok) {
        throw new Error(`Erreur HTTP: ${logoResponse.status}`);
      }

      const logoBytes = await logoResponse.arrayBuffer();

      // Détecter le type d'image et l'embedder
      let logoImage;
      const contentType = logoResponse.headers.get("content-type");

      if (contentType?.includes("image/png")) {
        logoImage = await pdfDoc.embedPng(logoBytes);
      } else if (
        contentType?.includes("image/jpeg") ||
        contentType?.includes("image/jpg")
      ) {
        logoImage = await pdfDoc.embedJpg(logoBytes);
      } else {
        // Essayer PNG par défaut
        logoImage = await pdfDoc.embedPng(logoBytes);
      }

      // Calculer la taille optimale (max 80px de hauteur)
      const maxHeight = 80;
      const scaleFactor = Math.min(maxHeight / logoImage.height, 0.15);
      const scaledWidth = logoImage.width * scaleFactor;
      const scaledHeight = logoImage.height * scaleFactor;

      page.drawImage(logoImage, {
        x: 50,
        y: height - 80,
        width: scaledWidth,
        height: scaledHeight,
      });

      console.log("✅ Logo de l'entreprise intégré avec succès");
    } else {
      console.log("📷 Utilisation du logo par défaut");
      // Fallback vers le logo par défaut
      const logoBytes = await fetch(logo).then((res) => res.arrayBuffer());
      const logoImage = await pdfDoc.embedPng(logoBytes);
      const scaleFactor = 0.09;
      const scaledWidth = logoImage.width * scaleFactor;
      const scaledHeight = logoImage.height * scaleFactor;

      page.drawImage(logoImage, {
        x: 50,
        y: height - 80,
        width: scaledWidth,
        height: scaledHeight,
      });
    }
  } catch (error) {
    console.warn("⚠️ Impossible de charger le logo:", error.message);
    // En cas d'erreur, on continue sans logo
  }

  // ✅ Coordonnées de l'entreprise (simplifiées)
  const company = invoiceData.company || {};
  const companyName = company.name || "Emmenez-moi Sàrl";
  const companyAddress =
    company.address || "Route de Chevrens 145, 1247 Anières";
  const companyPhone = company.contact_phone || "079.291.50.37";
  const billingEmail =
    company.billing_email || company.contact_email || "contact@emmenez-moi.ch";
  const companyUid = company.uid_ide || "";

  // Nom de l'entreprise
  page.drawText(companyName, {
    x: 350,
    y: height - 50,
    size: 12,
    font,
    color: rgb(0, 0, 0),
  });

  // Adresse
  page.drawText(companyAddress, {
    x: 350,
    y: height - 70,
    size: 10,
    font,
  });

  // Email de facturation
  page.drawText(`Email facturation : ${billingEmail}`, {
    x: 350,
    y: height - 90,
    size: 10,
    font,
  });

  // Téléphone
  page.drawText(`Téléphone : ${companyPhone}`, {
    x: 350,
    y: height - 110,
    size: 10,
    font,
  });

  // IDE/UID (si présent)
  if (companyUid) {
    page.drawText(`IDE/UID : ${companyUid}`, {
      x: 350,
      y: height - 130,
      size: 10,
      font,
    });
  }

  // ✅ Coordonnées du destinataire de la facture
  // Si c'est une facture tierce (institution), afficher l'institution, sinon le client
  const client = invoiceData.client || {};
  const institution = invoiceData.institution || null;

  let billedToName = "";
  let billedToAddress = "";

  if (institution) {
    // 🏥 Facturation tierce : afficher l'institution (clinique)
    billedToName = institution.name || "Institution";
    billedToAddress = institution.address || "Adresse non renseignée";
    console.log("🏥 Facture tierce pour institution:", institution);
  } else {
    // 👤 Facturation directe : afficher le client
    billedToName =
      `${client.firstName || ""} ${client.lastName || ""}`.trim() || "Client";

    // Récupération de l'adresse client avec fallbacks
    let clientAddress = "Adresse non renseignée";
    let clientZipCode = "";
    let clientCity = "";

    // Priorité 1: Adresse du Client (domicile)
    if (client.domicile?.address && client.domicile.address.trim()) {
      clientAddress = client.domicile.address.trim();
    }
    // Priorité 2: Adresse de l'utilisateur
    else if (client.user?.address && client.user.address.trim()) {
      clientAddress = client.user.address.trim();
    }
    // Priorité 3: Adresse directe du client
    else if (client.address && client.address.trim()) {
      clientAddress = client.address.trim();
    }

    // Code postal avec même logique de priorité (ignorer "Non spécifié" et valeurs vides)
    if (
      client.domicile?.zip &&
      client.domicile.zip.trim() &&
      client.domicile.zip.trim() !== "Non spécifié"
    ) {
      clientZipCode = client.domicile.zip.trim();
    } else if (
      client.user?.zip_code &&
      client.user.zip_code.trim() &&
      client.user.zip_code.trim() !== "Non spécifié"
    ) {
      clientZipCode = client.user.zip_code.trim();
    } else if (
      client.zipCode &&
      client.zipCode.trim() &&
      client.zipCode.trim() !== "Non spécifié"
    ) {
      clientZipCode = client.zipCode.trim();
    }

    // Ville avec même logique de priorité (ignorer "Non spécifié" et valeurs vides)
    if (
      client.domicile?.city &&
      client.domicile.city.trim() &&
      client.domicile.city.trim() !== "Non spécifié"
    ) {
      clientCity = client.domicile.city.trim();
    } else if (
      client.user?.city &&
      client.user.city.trim() &&
      client.user.city.trim() !== "Non spécifié"
    ) {
      clientCity = client.user.city.trim();
    } else if (
      client.city &&
      client.city.trim() &&
      client.city.trim() !== "Non spécifié"
    ) {
      clientCity = client.city.trim();
    }

    // Construire l'adresse complète du client
    billedToAddress = clientAddress;
    if (clientZipCode && clientCity) {
      billedToAddress += `, ${clientZipCode} ${clientCity}`;
    } else if (clientZipCode) {
      billedToAddress += `, ${clientZipCode}`;
    } else if (clientCity) {
      billedToAddress += `, ${clientCity}`;
    }

    console.log("👤 Facturation directe pour client:", client);
  }

  console.log("📍 Adresse de facturation finale:", billedToAddress);

  page.drawText("Facturé à :", {
    x: 50,
    y: height - 180,
    size: 12,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText(billedToName, {
    x: 50,
    y: height - 200,
    size: 11,
    font,
  });
  page.drawText(billedToAddress, {
    x: 50,
    y: height - 220,
    size: 10,
    font,
  });

  // ✅ Tableau des services avec mise en page améliorée
  let tableStartY = height - 280;

  // Définition des colonnes avec largeurs fixes
  // Si facturation tierce, ajouter une colonne "Patient"
  const isThirdPartyBilling = !!institution;

  const columns = isThirdPartyBilling
    ? {
        date: { x: 50, width: 60 },
        patient: { x: 115, width: 95 },
        departure: { x: 215, width: 120 },
        arrival: { x: 340, width: 120 },
        amount: { x: 465, width: 80 },
      }
    : {
        date: { x: 50, width: 80 },
        departure: { x: 140, width: 140 },
        arrival: { x: 290, width: 140 },
        amount: { x: 440, width: 100 },
      };

  // Fonction pour formater une adresse avec retour à la ligne
  const formatAddress = (address, maxWidth) => {
    if (!address || address === "Inconnu") return ["Adresse non renseignée"];

    // Nettoyer l'adresse
    let cleanAddress = address
      .replace(/, Suisse$/, "") // Supprime "Suisse" en fin
      .replace(/\s+/g, " ") // Supprime les espaces multiples
      .trim();

    // Si l'adresse est courte, la retourner telle quelle
    if (cleanAddress.length <= 25) {
      return [cleanAddress];
    }

    // Diviser l'adresse en mots
    const words = cleanAddress.split(" ");
    const lines = [];
    let currentLine = "";

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;

      // Si la ligne dépasse la largeur, commencer une nouvelle ligne
      if (testLine.length > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }

    // Ajouter la dernière ligne
    if (currentLine) {
      lines.push(currentLine);
    }

    return lines.slice(0, 3); // Maximum 3 lignes
  };

  // En-têtes du tableau
  page.drawText("Date", {
    x: columns.date.x,
    y: tableStartY,
    size: 10,
    font,
    color: rgb(0, 0, 0),
  });

  // Ajouter "Patient" seulement si facturation tierce
  if (isThirdPartyBilling) {
    page.drawText("Patient", {
      x: columns.patient.x,
      y: tableStartY,
      size: 10,
      font,
      color: rgb(0, 0, 0),
    });
  }

  page.drawText("Départ", {
    x: columns.departure.x,
    y: tableStartY,
    size: 10,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText("Arrivée", {
    x: columns.arrival.x,
    y: tableStartY,
    size: 10,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText("Montant", {
    x: columns.amount.x,
    y: tableStartY,
    size: 10,
    font,
    color: rgb(0, 0, 0),
  });

  let totalAmount = 0;

  invoiceData.bookings.forEach((booking) => {
    // Calculer la hauteur nécessaire pour cette ligne
    const departureLines = formatAddress(
      booking.pickup_location,
      isThirdPartyBilling ? 25 : 35
    );
    const arrivalLines = formatAddress(
      booking.dropoff_location,
      isThirdPartyBilling ? 25 : 35
    );
    const maxLines = Math.max(departureLines.length, arrivalLines.length, 1);

    // Espacement entre les lignes
    const lineHeight = 11;
    const rowHeight = maxLines * lineHeight + 8; // Hauteur de la ligne + marge

    tableStartY -= rowHeight;

    // ✅ Vérification du montant
    const amount = parseFloat(booking.amount || 0).toFixed(2);
    totalAmount += parseFloat(amount);

    // Date
    page.drawText(
      new Date(booking.scheduled_time).toLocaleDateString("fr-FR"),
      { x: columns.date.x, y: tableStartY, size: 9, font }
    );

    // Nom du patient (seulement si facturation tierce)
    if (isThirdPartyBilling) {
      const patientName =
        booking.customer_name ||
        `${booking.client_first_name || ""} ${
          booking.client_last_name || ""
        }`.trim() ||
        "Patient";

      // Tronquer si trop long
      const maxPatientLength = 16;
      const displayName =
        patientName.length > maxPatientLength
          ? patientName.substring(0, maxPatientLength - 1) + "."
          : patientName;

      page.drawText(displayName, {
        x: columns.patient.x,
        y: tableStartY,
        size: 8,
        font,
      });
    }

    // Adresse de départ (avec retour à la ligne)
    departureLines.forEach((line, index) => {
      page.drawText(line, {
        x: columns.departure.x,
        y: tableStartY - index * lineHeight,
        size: 8,
        font,
      });
    });

    // Adresse d'arrivée (avec retour à la ligne)
    arrivalLines.forEach((line, index) => {
      page.drawText(line, {
        x: columns.arrival.x,
        y: tableStartY - index * lineHeight,
        size: 8,
        font,
      });
    });

    // Montant
    page.drawText(`${amount} CHF`, {
      x: columns.amount.x,
      y: tableStartY,
      size: 9,
      font,
    });
  });

  // ✅ Affichage du montant total
  tableStartY -= 30;
  page.drawText("TOTAL :", {
    x: columns.amount.x - 60,
    y: tableStartY,
    size: 12,
    font,
    color: rgb(0, 0, 0),
  });
  page.drawText(`${totalAmount.toFixed(2)} CHF`, {
    x: columns.amount.x,
    y: tableStartY,
    size: 12,
    font,
    color: rgb(0, 0, 0),
  });

  // ✅ Pied de page avec marges équilibrées et espacement correct
  const footerY = 80; // Position fixe en bas de page
  const pageWidth = 595; // Largeur A4
  const marginLeft = 50;
  const marginRight = 50;
  const maxTextWidth = pageWidth - marginLeft - marginRight; // 495px disponibles

  // Fonction améliorée pour diviser le texte en lignes avec coupures naturelles
  const wrapText = (text, maxWidth, fontSize = 9) => {
    // D'abord, diviser par phrases (points, virgules)
    const sentences = text.split(/(?<=[.!?])\s+/);
    const lines = [];

    for (const sentence of sentences) {
      const words = sentence.split(" ");
      let currentLine = "";

      for (const word of words) {
        const testLine = currentLine ? `${currentLine} ${word}` : word;
        // Estimation plus précise de la largeur (1 caractère ≈ fontSize * 0.55 pour Helvetica)
        const estimatedWidth = testLine.length * fontSize * 0.55;

        if (estimatedWidth > maxWidth && currentLine) {
          // Si la ligne dépasse, sauvegarder la ligne actuelle et commencer une nouvelle ligne
          lines.push(currentLine);
          currentLine = word;
        } else {
          currentLine = testLine;
        }
      }

      // Ajouter la ligne actuelle si elle n'est pas vide
      if (currentLine) {
        lines.push(currentLine);
      }
    }

    return lines;
  };

  // Notes de facturation avec retour à la ligne automatique
  const billingNotes =
    company.billing_notes ||
    "En votre aimable règlement net sous 10 jours avec nos remerciements anticipés. En cas de retard de paiement, des frais de rappel d'un montant de CHF 15.- vous seront facturés, conformément à nos conditions générales.";

  // Utiliser une largeur plus conservatrice pour éviter les débordements
  const conservativeMaxWidth = maxTextWidth * 1.25; // Largeur souhaitée par l'utilisateur
  const notesLines = wrapText(billingNotes, conservativeMaxWidth);

  console.log("📝 Notes de facturation:", billingNotes);
  console.log("📏 Largeur maximale:", conservativeMaxWidth);
  console.log("📄 Lignes générées:", notesLines);

  // Calculer la position de départ pour les notes (en fonction du nombre de lignes)
  const notesStartY = footerY + 20;

  // Afficher les notes de facturation ligne par ligne
  notesLines.forEach((line, index) => {
    page.drawText(line, {
      x: marginLeft,
      y: notesStartY - index * 12, // Interligne de 12px
      size: 9,
      font,
      color: rgb(0, 0, 0),
    });
  });

  // Calculer la position des informations bancaires (après toutes les notes)
  const bankingY = notesStartY - notesLines.length * 12 - 20; // 20px d'espacement supplémentaire

  // Informations bancaires sur la même ligne
  const paymentText = `Paiement par virement bancaire : IBAN : ${
    company.iban || "Non renseigné"
  }`;
  page.drawText(paymentText, {
    x: marginLeft,
    y: bankingY,
    size: 9,
    font,
    color: rgb(0, 0, 0),
  });

  // ✅ Ajouter le numéro de facture aux données pour le QR-Bill
  invoiceData.invoiceNumber = invoiceNumber;

  const pdfBytes = await pdfDoc.save();
  return pdfBytes;
}
