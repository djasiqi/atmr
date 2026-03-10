import React from 'react';
import ContactSubpageTemplate from './components/ContactSubpageTemplate';
import { fieldsByCategory } from './contactCategories';

const ContactTransport = () => (
  <ContactSubpageTemplate category="transport" config={fieldsByCategory.transport} />
);

export default ContactTransport;
