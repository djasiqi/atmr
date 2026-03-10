import React from 'react';
import ContactSubpageTemplate from './components/ContactSubpageTemplate';
import { fieldsByCategory } from './contactCategories';

const ContactFamily = () => <ContactSubpageTemplate category="family" config={fieldsByCategory.family} />;

export default ContactFamily;
