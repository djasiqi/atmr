import { toast } from 'sonner';

const COMING_SOON_TOAST_ID = 'global-coming-soon';

export const showComingSoonToast = () => {
  toast.dismiss(COMING_SOON_TOAST_ID);
  toast.message('Bientôt disponible', {
    id: COMING_SOON_TOAST_ID,
    duration: 5000,
    description:
      'Notre équipe finalise cette section. Écrivez-nous à info@lirie.ch pour être informé du lancement.',
  });
};

export default showComingSoonToast;
