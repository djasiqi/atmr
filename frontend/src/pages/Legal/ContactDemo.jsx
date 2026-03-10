import React from 'react';
import ContactSubpageTemplate from './components/ContactSubpageTemplate';
import { fieldsByCategory } from './contactCategories';

const ContactDemo = () => <ContactSubpageTemplate category="demo" config={fieldsByCategory.demo} />;

export default ContactDemo;
