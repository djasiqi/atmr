import React from 'react';
import ContactSubpageTemplate from './components/ContactSubpageTemplate';
import { fieldsByCategory } from './contactCategories';

const ContactInstitution = () => (
  <ContactSubpageTemplate category="institution" config={fieldsByCategory.institution} />
);

export default ContactInstitution;
