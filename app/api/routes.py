
from flask import Blueprint, request, jsonify
import requests
from datetime import datetime
import logging

from app.api.rag_service import get_prompting_service

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

# Mock translation function - kept as fallback
def mock_translate(text, source_lang, target_lang, n=3):
    """Return a list of up to `n` mock translation variants.

    Used as fallback when RAG services are unavailable.
    """
    mock_translations_base = {
        'en-ar': [
            f"ترجمة تجريبية للنص: {text}",
            f"ترجمة بديلة: {text}",
            f"ترجمة مختصرة: {text}"
        ],
        'fr-ar': [
            f"ترجمة تجريبية من الفرنسية: {text}",
            f"ترجمة بديلة من الفرنسية: {text}",
            f"ترجمة مختصرة من الفرنسية: {text}"
        ],
        'ar-en': [
            f"Mock translation to English: {text}",
            f"Alternative English translation: {text}",
            f"Concise English translation: {text}"
        ],
        'ar-fr': [
            f"Traduction simulée vers le français: {text}",
            f"Traduction alternative vers le français: {text}",
            f"Traduction concise vers le français: {text}"
        ],
    }

    key = f"{source_lang}-{target_lang}"
    variants = mock_translations_base.get(key)
    if not variants:
        # Fallback: generate simple numbered variants
        variants = [f"Mock translation ({i+1}): {text}" for i in range(n)]

    # Ensure we only return up to n variants
    return variants[:n]


@api_bp.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Translation API Server',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'translate': '/api/translate',
            'detect_language': '/api/detect-language'
        }
    })

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'translation-api'
    })

@api_bp.route('/translate', methods=['POST'])
def translate():
    """Main translation endpoint - integrates with RAG pipeline and OpenRouter"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        text = data.get('text', '').strip()
        source_language = data.get('source_language', 'en')
        target_language = data.get('target_language', 'ar')
        domain = data.get('domain', 'general')  # Optional domain parameter

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        if len(text) > 10000:  # Reasonable limit
            return jsonify({'error': 'Text too long (max 10000 characters)'}), 400

        # Use the new Prompting translation service
        try:
            translation_service = get_prompting_service()
            result = translation_service.translate(
                text=text,
                source_lang=source_language,
                target_lang=target_language,
                domain=domain,
                num_variants=3  # Generate 3 diverse, high-quality translations
            )
            
            translations = result.get('translations', [])
            translation = translations[0] if translations else ''
            
            # Return the translation response
            response = {
                'original_text': text,
                'translation': translation,
                'translations': translations,
                'source_language': source_language,
                'target_language': target_language,
                'domain': domain,
                'timestamp': datetime.utcnow().isoformat(),
                'variants': translations, # Kept for frontend compatibility
                'metadata': {}
            }
            
            logger.info(f"Translation successful: {len(translations)} variant(s) generated")
            return jsonify(response)
            
        except Exception as service_error:
            logger.error(f"Translation service failed: {service_error}, falling back to mock")
            # Fallback to mock translation if the service is unavailable
            translations = mock_translate(text, source_language, target_language, n=3)
            translation = translations[0] if translations else ''
            
            response = {
                'original_text': text,
                'translation': translation,
                'translations': translations,
                'source_language': source_language,
                'target_language': target_language,
                'timestamp': datetime.utcnow().isoformat(),
                'variants': translations,
                'warning': 'Translation service unavailable, using fallback mock translation.'
            }
            
            return jsonify(response)

    except Exception as e:
        logger.error(f"Translation endpoint error: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@api_bp.route('/upload-file', methods=['POST'])
def upload_file():
    """Accepts a single file upload (multipart/form-data) and extracts text from it.

    Supported types: .txt, .pdf, .docx
    Returns: { filename, content_type, text }
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename
        content_type = file.content_type or ''
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        # Read bytes so we can reuse for different parsers
        file_bytes = file.read()

        text = ''
        if ext in ('txt', 'md', 'csv'):
            try:
                text = file_bytes.decode('utf-8')
            except Exception:
                text = file_bytes.decode('latin-1', errors='ignore')

        elif ext == 'pdf':
            # Quick sanity checks on the uploaded bytes
            if len(file_bytes) < 100 or b'%PDF' not in file_bytes[:1024]:
                logger.warning('Uploaded file does not appear to be a valid PDF (too small or missing %PDF header)')
                return jsonify({'error': 'Invalid or empty PDF file', 'detail': 'Uploaded file is too small or missing a %PDF header; it may be corrupt.'}), 422

            # Try PyMuPDF first (prefer import of 'pymupdf' to avoid naming conflicts with other 'fitz' packages)
            try:
                import pymupdf as fitz  # prefer the proper pymupdf package

                try:
                    # Try opening; catch errors to return clear response
                    doc = fitz.open(stream=file_bytes, filetype='pdf')
                except Exception as open_err:
                    logger.error(f'fitz failed to open PDF: {open_err}', exc_info=True)
                    return jsonify({'error': 'Failed to open PDF', 'detail': str(open_err)}), 400

                pages_text = []
                for i, page in enumerate(doc):
                    try:
                        p_text = page.get_text('text') or ''
                    except Exception as perr:
                        logger.debug(f'page.get_text error on page {i+1}: {perr}')
                        p_text = ''
                    pages_text.append(p_text)
                    logger.debug(f'Page {i+1}: extracted {len(p_text)} chars')

                text = '\n'.join(pages_text)
                logger.info(f'Extracted total {len(text)} characters from PDF using PyMuPDF/fitz (pages: {getattr(doc, "page_count", len(pages_text))})')

                if not text.strip():
                    logger.warning('PDF parsed successfully but no extractable text found (fitz)')
                    return jsonify({
                        'error': 'No extractable text found in PDF',
                        'detail': 'The PDF may be a scanned image without embedded text. Consider performing OCR (e.g., Tesseract) or uploading a text/Word file.'
                    }), 422

            except Exception as fitz_err:
                logger.debug(f'PyMuPDF/fitz not usable or failed: {fitz_err}', exc_info=True)
                # Try pypdf fallback if available
                try:
                    from pypdf import PdfReader
                    import io
                    reader = PdfReader(io.BytesIO(file_bytes))
                    pages = [p.extract_text() or '' for p in reader.pages]
                    text = '\n'.join(pages)

                    if not text.strip():
                        logger.warning('PDF parsed successfully but no extractable text found (pypdf fallback)')
                        return jsonify({
                            'error': 'No extractable text found in PDF',
                            'detail': 'The PDF may be a scanned image without embedded text. Consider performing OCR (e.g., Tesseract) or uploading a text/Word file.'
                        }), 422

                except Exception as pypdf_err:
                    logger.error(f'PDF extraction failed (both PyMuPDF and pypdf attempts failed): fitz_err={fitz_err}, pypdf_err={pypdf_err}', exc_info=True)
                    # If no libraries are available or both failed, return actionable error
                    return jsonify({
                        'error': 'Failed to extract text from PDF',
                        'detail': 'Could not extract text. Ensure PyMuPDF (recommended) or pypdf is installed in the Python environment. See README for setup.'
                    }), 500

        elif ext in ('docx', 'doc'):
            try:
                from docx import Document
                import io
                doc = Document(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text]
                text = '\n'.join(paragraphs)
            except Exception as e:
                logger.error(f'DOCX extraction failed: {e}', exc_info=True)
                return jsonify({'error': 'Failed to extract text from Word document'}), 500

        else:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400

        return jsonify({'filename': filename, 'content_type': content_type, 'text': text}), 200

    except Exception as e:
        logger.error(f'File upload/extract error: {e}', exc_info=True)
        return jsonify({'error': 'Internal server error while extracting file'}), 500

@api_bp.route('/detect-language', methods=['POST'])
def detect_language():
    """Language detection endpoint"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400

        text = data['text'].strip()

        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Simple mock language detection
        # In production, use a proper language detection library
        if any(char in 'abcdefghijklmnopqrstuvwxyz' for char in text.lower()):
            detected_lang = 'en' if text[0].isupper() else 'en'  # Very basic
        elif any(char in 'àâäéèêëïîôöùûüÿç' for char in text.lower()):
            detected_lang = 'fr'
        else:
            detected_lang = 'ar'  # Default to Arabic for this project

        return jsonify({
            'text': text,
            'detected_language': detected_lang,
            'confidence': 0.8  # Mock confidence
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500