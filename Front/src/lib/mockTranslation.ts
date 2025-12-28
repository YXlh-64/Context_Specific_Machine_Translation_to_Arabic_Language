import { TranslationVariant } from '@/types/translation';

// API base URL from environment
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

// Note: debug logs removed for production cleanliness. Set VITE_DEBUG to enable runtime logging if needed.

// Fallback mock Arabic translations for when API is unavailable
const mockArabicResponses = [
  [
    'مرحباً بكم في أداة الترجمة الاحترافية',
    'أهلاً وسهلاً في أداة الترجمة المتقدمة',
    'نرحب بكم في برنامج الترجمة المهنية',
  ],
  [
    'هذا النص هو مثال على الترجمة العربية',
    'هذه الجملة توضح نموذج الترجمة بالعربية',
    'يُظهر هذا المحتوى كيفية عمل الترجمة',
  ],
  [
    'الترجمة الآلية تساعد في التواصل بين الثقافات',
    'تُسهّل الترجمة التلقائية التفاهم عبر اللغات',
    'يُيسّر النقل الآلي للمعنى الحوار بين الشعوب',
  ],
];

export const detectLanguage = async (text: string): Promise<'en' | 'fr' | 'ar'> => {
  try {
    const response = await fetch(`${API_BASE_URL}/detect-language`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    return data.detected_language as 'en' | 'fr' | 'ar';
  } catch (error) {
    console.warn('Language detection API failed, using fallback:', error);
    // Fallback to mock detection
    const frenchIndicators = ['le', 'la', 'les', 'un', 'une', 'de', 'du', 'des', 'et', 'est', 'sont', 'dans', 'pour', 'avec', 'sur', 'qui', 'que', 'ne', 'pas', 'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes'];
    const words = text.toLowerCase().split(/\s+/);
    const frenchWordCount = words.filter(word => frenchIndicators.includes(word)).length;

    return frenchWordCount >= 2 ? 'fr' : 'en';
  }
};

export const translate = async (
  text: string,
  sourceLanguage: 'en' | 'fr' | 'ar',
  targetLanguage: 'en' | 'fr' | 'ar' = 'ar'
): Promise<TranslationVariant[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        source_language: sourceLanguage,
        target_language: targetLanguage,
      }),
    });

    if (!response.ok) {
      throw new Error(`Translation API request failed: ${response.status}`);
    }

    const data = await response.json();

    // Convert API response to TranslationVariant format
    const variants: TranslationVariant[] = data.variants.map((variantText: string, index: number) => ({
      id: `api-variant-${Date.now()}-${index}`,
      text: variantText,
      rank: index + 1,
      isEdited: false,
    }));

    return variants;
  } catch (error) {
    console.error('Translation API failed, using fallback mock data:', error);
    console.error('Error details:', error);
    // Fallback to mock translation
    return mockTranslateFallback(text, sourceLanguage);
  }
};

// Keep the old function name for backward compatibility, but use the new API-based one
export const mockTranslate = translate;

// Fallback mock translation function
const mockTranslateFallback = async (
  text: string,
  _sourceLanguage: 'en' | 'fr' | 'ar'
): Promise<TranslationVariant[]> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1200));

  // Select random Arabic responses or generate based on text length
  const responseSet = mockArabicResponses[Math.floor(Math.random() * mockArabicResponses.length)];

  // If text is longer, add some context-aware mock response
  const variants: TranslationVariant[] = responseSet.map((arabicText, index) => ({
    id: `fallback-variant-${Date.now()}-${index}`,
    text: text.length > 50
      ? `${arabicText}. ${text.substring(0, 20)}... → النص المترجم`
      : arabicText,
    rank: index + 1,
    isEdited: false,
  }));

  return variants;
};
