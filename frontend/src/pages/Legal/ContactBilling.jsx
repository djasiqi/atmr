import React from 'react';
import ContactSubpageTemplate from './components/ContactSubpageTemplate';
import { fieldsByCategory } from './contactCategories';

const ContactBilling = () => <ContactSubpageTemplate category="billing" config={fieldsByCategory.billing} />;

export default ContactBilling;
