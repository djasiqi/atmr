import React from 'react';
import ContactSubpageTemplate from './components/ContactSubpageTemplate';
import { fieldsByCategory } from './contactCategories';

const ContactSupport = () => <ContactSubpageTemplate category="support" config={fieldsByCategory.support} />;

export default ContactSupport;
