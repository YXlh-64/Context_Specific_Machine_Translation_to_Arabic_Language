import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export type AppLanguage = 'en' | 'fr' | 'ar';

interface Translations {
  // Header
  appName: string;
  appSubtitle: string;
  signIn: string;
  signOut: string;
  loggedIn: string;
  
  // Auth Modal
  login: string;
  signup: string;
  name: string;
  password: string;
  enterName: string;
  enterPassword: string;
  continueBtn: string;
  createAccount: string;
  noAccount: string;
  haveAccount: string;
  
  // Sidebar
  history: string;
  noHistory: string;
  settings: string;
  language: string;
  
  // Input Pane
  typeToTranslate: string;
  words: string;
  translateTo: string;
  translating: string;
  uploadFile: string;
  dropFileHere: string;
  supportedFormats: string;
  removeFile: string;
  
  // Output Pane
  translations: string;
  dragToReorder: string;
  noTranslations: string;
  enterTextToStart: string;
  variation: string;
  edited: string;
  userCustom: string;
  draftCustom: string;
  save: string;
  cancel: string;
  
  // Language Bar
  from: string;
  to: string;
  english: string;
  french: string;
  arabic: string;
  
  // Toasts
  welcome: string;
  signedInAs: string;
  signedOut: string;
  seeYou: string;
  translationComplete: string;
  variationsGenerated: string;
  translationFailed: string;
  tryAgain: string;
  translationUpdated: string;
  editSaved: string;
  sessionDeleted: string;
  removedFromHistory: string;
  fileUploaded: string;
  textExtracted: string;
  submitPreferences: string;
  preferencesSubmitted: string;
  preferencesSubmittedDesc: string;
}

const translations: Record<AppLanguage, Translations> = {
  en: {
    appName: 'LinguistBridge',
    appSubtitle: 'Professional Translation Tool',
    signIn: 'Sign in',
    signOut: 'Sign out',
    loggedIn: 'Logged in',
    login: 'Login',
    signup: 'Sign Up',
    name: 'Name',
    password: 'Password',
    enterName: 'Enter your name',
    enterPassword: 'Enter your password',
    continueBtn: 'Continue',
    createAccount: 'Create Account',
    noAccount: "Don't have an account?",
    haveAccount: 'Already have an account?',
    history: 'History',
    noHistory: 'No translation history yet',
    settings: 'Settings',
    language: 'Language',
    typeToTranslate: 'Type to translate...',
    words: 'words',
    translateTo: 'Translate to',
    translating: 'Translating...',
    uploadFile: 'Upload file',
    dropFileHere: 'Drop file here or click to upload',
    supportedFormats: 'PDF, DOCX, TXT supported',
    removeFile: 'Remove file',
    translations: 'Translations',
    dragToReorder: 'Drag to reorder by preference',
    noTranslations: 'No translations yet',
    enterTextToStart: 'Enter text and click "Translate" to get started',
    variation: 'Variation',
    edited: 'Edited',
    userCustom: 'User Custom',
    draftCustom: 'Draft Custom Translation from Scratch',
    save: 'Save',
    cancel: 'Cancel',
    from: 'From',
    to: 'To',
    english: 'English',
    french: 'French',
    arabic: 'Arabic',
    welcome: 'Welcome!',
    signedInAs: 'Signed in as',
    signedOut: 'Signed out',
    seeYou: 'See you next time!',
    translationComplete: 'Translation complete',
    variationsGenerated: 'variations generated',
    translationFailed: 'Translation failed',
    tryAgain: 'Please try again',
    translationUpdated: 'Translation updated',
    editSaved: 'Your edit has been saved',
    sessionDeleted: 'Session deleted',
    removedFromHistory: 'Translation removed from history',
    fileUploaded: 'File uploaded',
    textExtracted: 'Text extracted from file',
    submitPreferences: 'Submit Preferences',
    preferencesSubmitted: 'Preferences submitted',
    preferencesSubmittedDesc: 'Your translation rankings have been saved',
  },
  fr: {
    appName: 'LinguistBridge',
    appSubtitle: 'Outil de Traduction Professionnel',
    signIn: 'Se connecter',
    signOut: 'Se déconnecter',
    loggedIn: 'Connecté',
    login: 'Connexion',
    signup: 'Inscription',
    name: 'Nom',
    password: 'Mot de passe',
    enterName: 'Entrez votre nom',
    enterPassword: 'Entrez votre mot de passe',
    continueBtn: 'Continuer',
    createAccount: 'Créer un compte',
    noAccount: "Vous n'avez pas de compte ?",
    haveAccount: 'Vous avez déjà un compte ?',
    history: 'Historique',
    noHistory: "Pas encore d'historique de traduction",
    settings: 'Paramètres',
    language: 'Langue',
    typeToTranslate: 'Tapez pour traduire...',
    words: 'mots',
    translateTo: 'Traduire en',
    translating: 'Traduction...',
    uploadFile: 'Télécharger un fichier',
    dropFileHere: 'Déposez le fichier ici ou cliquez pour télécharger',
    supportedFormats: 'PDF, DOCX, TXT supportés',
    removeFile: 'Supprimer le fichier',
    translations: 'Traductions',
    dragToReorder: 'Glissez pour réorganiser par préférence',
    noTranslations: 'Pas encore de traductions',
    enterTextToStart: 'Entrez du texte et cliquez sur "Traduire" pour commencer',
    variation: 'Variation',
    edited: 'Modifié',
    userCustom: 'Personnalisé',
    draftCustom: 'Rédiger une traduction personnalisée',
    save: 'Enregistrer',
    cancel: 'Annuler',
    from: 'De',
    to: 'Vers',
    english: 'Anglais',
    french: 'Français',
    arabic: 'Arabe',
    welcome: 'Bienvenue !',
    signedInAs: 'Connecté en tant que',
    signedOut: 'Déconnecté',
    seeYou: 'À bientôt !',
    translationComplete: 'Traduction terminée',
    variationsGenerated: 'variations générées',
    translationFailed: 'Échec de la traduction',
    tryAgain: 'Veuillez réessayer',
    translationUpdated: 'Traduction mise à jour',
    editSaved: 'Votre modification a été enregistrée',
    sessionDeleted: 'Session supprimée',
    removedFromHistory: "Traduction supprimée de l'historique",
    fileUploaded: 'Fichier téléchargé',
    textExtracted: 'Texte extrait du fichier',
    submitPreferences: 'Soumettre les préférences',
    preferencesSubmitted: 'Préférences soumises',
    preferencesSubmittedDesc: 'Vos classements de traduction ont été enregistrés',
  },
  ar: {
    appName: 'جسر اللغات',
    appSubtitle: 'أداة ترجمة احترافية',
    signIn: 'تسجيل الدخول',
    signOut: 'تسجيل الخروج',
    loggedIn: 'مسجّل الدخول',
    login: 'دخول',
    signup: 'إنشاء حساب',
    name: 'الاسم',
    password: 'كلمة المرور',
    enterName: 'أدخل اسمك',
    enterPassword: 'أدخل كلمة المرور',
    continueBtn: 'متابعة',
    createAccount: 'إنشاء حساب',
    noAccount: 'ليس لديك حساب؟',
    haveAccount: 'لديك حساب بالفعل؟',
    history: 'السجل',
    noHistory: 'لا يوجد سجل ترجمة بعد',
    settings: 'الإعدادات',
    language: 'اللغة',
    typeToTranslate: 'اكتب للترجمة...',
    words: 'كلمات',
    translateTo: 'ترجم إلى',
    translating: 'جارٍ الترجمة...',
    uploadFile: 'رفع ملف',
    dropFileHere: 'اسحب الملف هنا أو انقر للرفع',
    supportedFormats: 'PDF، DOCX، TXT مدعومة',
    removeFile: 'إزالة الملف',
    translations: 'الترجمات',
    dragToReorder: 'اسحب لإعادة الترتيب حسب التفضيل',
    noTranslations: 'لا توجد ترجمات بعد',
    enterTextToStart: 'أدخل النص وانقر على "ترجم" للبدء',
    variation: 'الاختلاف',
    edited: 'معدّل',
    userCustom: 'مخصص',
    draftCustom: 'صياغة ترجمة مخصصة من الصفر',
    save: 'حفظ',
    cancel: 'إلغاء',
    from: 'من',
    to: 'إلى',
    english: 'الإنجليزية',
    french: 'الفرنسية',
    arabic: 'العربية',
    welcome: 'مرحباً!',
    signedInAs: 'مسجّل الدخول باسم',
    signedOut: 'تم تسجيل الخروج',
    seeYou: 'إلى اللقاء!',
    translationComplete: 'اكتملت الترجمة',
    variationsGenerated: 'اختلافات تم إنشاؤها',
    translationFailed: 'فشلت الترجمة',
    tryAgain: 'يرجى المحاولة مرة أخرى',
    translationUpdated: 'تم تحديث الترجمة',
    editSaved: 'تم حفظ التعديل',
    sessionDeleted: 'تم حذف الجلسة',
    removedFromHistory: 'تمت إزالة الترجمة من السجل',
    fileUploaded: 'تم رفع الملف',
    textExtracted: 'تم استخراج النص من الملف',
    submitPreferences: 'إرسال التفضيلات',
    preferencesSubmitted: 'تم إرسال التفضيلات',
    preferencesSubmittedDesc: 'تم حفظ ترتيب الترجمات الخاص بك',
  },
};

interface I18nContextType {
  appLanguage: AppLanguage;
  setAppLanguage: (lang: AppLanguage) => void;
  t: Translations;
  isRtl: boolean;
}

const I18nContext = createContext<I18nContextType | null>(null);

const APP_LANG_STORAGE_KEY = 'linguist-bridge-app-lang';

export const I18nProvider = ({ children }: { children: ReactNode }) => {
  const [appLanguage, setAppLanguageState] = useState<AppLanguage>(() => {
    try {
      const stored = localStorage.getItem(APP_LANG_STORAGE_KEY);
      if (stored && ['en', 'fr', 'ar'].includes(stored)) {
        return stored as AppLanguage;
      }
    } catch (e) {
      console.error('Failed to load app language', e);
    }
    return 'en';
  });

  const setAppLanguage = (lang: AppLanguage) => {
    setAppLanguageState(lang);
    try {
      localStorage.setItem(APP_LANG_STORAGE_KEY, lang);
    } catch (e) {
      console.error('Failed to save app language', e);
    }
  };

  const isRtl = appLanguage === 'ar';
  const t = translations[appLanguage];

  // Update document direction
  useEffect(() => {
    document.documentElement.dir = isRtl ? 'rtl' : 'ltr';
    document.documentElement.lang = appLanguage;
  }, [isRtl, appLanguage]);

  return (
    <I18nContext.Provider value={{ appLanguage, setAppLanguage, t, isRtl }}>
      {children}
    </I18nContext.Provider>
  );
};

export const useI18n = () => {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
};
