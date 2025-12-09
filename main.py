from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import PyPDF2
import pdfplumber
from docx import Document
import openpyxl
from pptx import Presentation
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile
import os
from langdetect import detect
import httpx
from openai import OpenAI
from supabase import create_client, Client
import hashlib
import time

app = FastAPI(title="Knowledge Base Processor Service")

# Configuración de clientes
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://hsoagaoxuaspkdptfgou.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


class ProcessDocumentRequest(BaseModel):
    file_url: str
    library_id: str
    file_name: str
    subject: str
    subject_id: Optional[str] = None
    authors: Optional[List[str]] = None
    title: Optional[str] = None
    publication_date: Optional[str] = None


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "knowledge-base-processor"}


@app.post("/process")
async def process_document(request: ProcessDocumentRequest):
    """
    Procesa un documento completo para la Knowledge Base:
    1. Descarga el documento
    2. Extrae el texto
    3. Crea chunks
    4. Genera embeddings
    5. Almacena en Supabase (documents_v2)
    6. Actualiza AIvantage_library
    """
    try:
        # Validar configuración
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
        if not SUPABASE_SERVICE_KEY:
            raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_KEY no configurada")
        
        # Inicializar clientes
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Paso 1: Descargar el documento
        print(f"Descargando documento: {request.file_url}")
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(request.file_url)
            if response.status_code != 200:
                # Actualizar status a error en AIvantage_library
                supabase.table("AIvantage_library").update({
                    "status": "error"
                }).eq("id", request.library_id).execute()
                raise HTTPException(status_code=400, detail=f"Error descargando archivo: {response.status_code}")
            file_content = response.content
        
        print(f"Documento descargado: {len(file_content)} bytes")
        
        # Actualizar status a processing
        supabase.table("AIvantage_library").update({
            "status": "processing"
        }).eq("id", request.library_id).execute()
        
        # Paso 2: Extraer texto según tipo de archivo
        file_extension = request.file_name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            extraction_result = await extract_pdf(file_content)
        elif file_extension in ['docx', 'doc']:
            extraction_result = await extract_docx(file_content)
        elif file_extension in ['xlsx', 'xls']:
            extraction_result = await extract_excel(file_content)
        elif file_extension in ['pptx', 'ppt']:
            extraction_result = await extract_pptx(file_content)
        elif file_extension in ['png', 'jpg', 'jpeg']:
            extraction_result = await extract_image(file_content)
        elif file_extension in ['txt', 'md']:
            extraction_result = await extract_text(file_content)
        else:
            supabase.table("AIvantage_library").update({
                "status": "error"
            }).eq("id", request.library_id).execute()
            raise HTTPException(status_code=400, detail=f"Formato no soportado: {file_extension}")
        
        extracted_text = extraction_result.get('text', '')
        if not extracted_text or len(extracted_text.strip()) < 50:
            supabase.table("AIvantage_library").update({
                "status": "error"
            }).eq("id", request.library_id).execute()
            raise HTTPException(status_code=400, detail="No se pudo extraer texto suficiente del documento")
        
        print(f"Texto extraído: {len(extracted_text)} caracteres")
        
        # Detectar idioma
        try:
            language = detect(extracted_text[:1000])
        except:
            language = 'unknown'
        
        # Paso 3: Crear chunks
        chunks = create_smart_chunks(extracted_text, chunk_size=4000, overlap=400)
        print(f"Chunks creados: {len(chunks)}")
        
        # Paso 4: Generar hash del documento
        doc_hash = hashlib.md5(file_content).hexdigest()
        
        # Paso 5: Insertar en record_manager_v2
        record_data = {
            "file_path": request.file_url,
            "file_name": request.file_name,
            "content_hash": doc_hash,
            "status": "processing",
            "total_chunks": len(chunks),
            "processed_chunks": 0
        }
        
        record_result = supabase.table("record_manager_v2").insert(record_data).execute()
        
        if not record_result.data:
            supabase.table("AIvantage_library").update({
                "status": "error"
            }).eq("id", request.library_id).execute()
            raise HTTPException(status_code=500, detail="Error creando registro en record_manager_v2")
        
        record_id = record_result.data[0]['id']
        print(f"Record creado: {record_id}")
        
        # Actualizar AIvantage_library con el record_id
        supabase.table("AIvantage_library").update({
            "record_id": record_id
        }).eq("id", request.library_id).execute()
        
        # Paso 6: Procesar chunks en lotes
        batch_size = 20
        total_vectors_inserted = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Generar embeddings para el lote
            texts_to_embed = [chunk['content'] for chunk in batch_chunks]
            
            try:
                embedding_response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts_to_embed
                )
                
                # Preparar datos para insertar en documents_v2
                documents_to_insert = []
                
                for j, (chunk, embedding_data) in enumerate(zip(batch_chunks, embedding_response.data)):
                    chunk_index = i + j
                    
                    doc_record = {
                        "content": chunk['content'],
                        "metadata": {
                            "file_name": request.file_name,
                            "subject": request.subject,
                            "subject_id": request.subject_id,
                            "authors": request.authors,
                            "title": request.title or request.file_name,
                            "publication_date": request.publication_date,
                            "chunk_index": chunk_index,
                            "total_chunks": len(chunks),
                            "language": language,
                            "library_id": request.library_id,
                            "source": "knowledge_base"
                        },
                        "embedding": embedding_data.embedding,
                        "record_id": record_id
                    }
                    documents_to_insert.append(doc_record)
                
                # Insertar en documents_v2
                if documents_to_insert:
                    insert_result = supabase.table("documents_v2").insert(documents_to_insert).execute()
                    total_vectors_inserted += len(documents_to_insert)
                
                # Actualizar progreso
                progress = min(i + batch_size, len(chunks))
                supabase.table("record_manager_v2").update({
                    "processed_chunks": progress
                }).eq("id", record_id).execute()
                
                # Actualizar loading_percentage en AIvantage_library
                loading_pct = round(progress / len(chunks), 2)
                supabase.table("AIvantage_library").update({
                    "loading_percentage": loading_pct
                }).eq("id", request.library_id).execute()
                
                print(f"Lote procesado: {progress}/{len(chunks)} chunks ({loading_pct*100}%)")
                
                # Pequeña pausa para evitar rate limits
                if i + batch_size < len(chunks):
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error procesando lote {i}: {str(e)}")
                continue
        
        # Paso 7: Actualizar estado final
        supabase.table("record_manager_v2").update({
            "status": "completed",
            "processed_chunks": len(chunks)
        }).eq("id", record_id).execute()
        
        supabase.table("AIvantage_library").update({
            "status": "processed",
            "loading_percentage": 1.0
        }).eq("id", request.library_id).execute()
        
        print(f"Procesamiento completado. Record ID: {record_id}, Vectores: {total_vectors_inserted}")
        
        return {
            "success": True,
            "record_id": record_id,
            "library_id": request.library_id,
            "total_chunks": len(chunks),
            "total_vectors": total_vectors_inserted,
            "language": language,
            "file_name": request.file_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en process_document: {str(e)}")
        # Intentar actualizar status a error
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            supabase.table("AIvantage_library").update({
                "status": "error"
            }).eq("id", request.library_id).execute()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FUNCIONES DE EXTRACCIÓN
# ============================================================

def create_smart_chunks(text: str, chunk_size: int = 4000, overlap: int = 400) -> List[Dict[str, Any]]:
    """Crea chunks inteligentes del texto"""
    if not text or len(text) <= chunk_size:
        return [{
            'content': text,
            'chunk_index': 0,
            'total_chunks': 1,
            'start_position': 0,
            'end_position': len(text) if text else 0
        }]
    
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    chunk_index = 0
    start_position = 0
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'start_position': start_position,
                    'end_position': start_position + len(current_chunk)
                })
                chunk_index += 1
                
                lines = current_chunk.split('\n')
                overlap_lines = lines[-5:] if len(lines) > 5 else lines
                overlap_text = '\n'.join(overlap_lines)
                
                start_position = start_position + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                if len(paragraph) > chunk_size:
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 > chunk_size:
                            chunks.append({
                                'content': current_chunk.strip(),
                                'chunk_index': chunk_index,
                                'start_position': start_position,
                                'end_position': start_position + len(current_chunk)
                            })
                            chunk_index += 1
                            start_position += len(current_chunk)
                            current_chunk = sentence + ". "
                        else:
                            current_chunk += sentence + ". "
                else:
                    current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append({
            'content': current_chunk.strip(),
            'chunk_index': chunk_index,
            'start_position': start_position,
            'end_position': len(text)
        })
    
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)
    
    return chunks


async def extract_pdf(content: bytes) -> Dict[str, Any]:
    """Extrae texto de PDF"""
    result = {'text': '', 'pages': 0}
    
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        result['pages'] = len(pdf_reader.pages)
        
        text_parts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(f"--- Página {page_num + 1} ---\n{page_text}")
        
        result['text'] = '\n\n'.join(text_parts)
        
        # Si no hay texto, intentar con pdfplumber
        if not result['text'].strip() or len(result['text']) < 100:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            with pdfplumber.open(tmp_path) as pdf:
                result['pages'] = len(pdf.pages)
                text_parts = []
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Página {i + 1} ---\n{page_text}")
                result['text'] = '\n\n'.join(text_parts)
            
            os.unlink(tmp_path)
        
        # Si aún no hay texto, intentar OCR
        if not result['text'].strip() or len(result['text']) < 100:
            result['text'] = await extract_pdf_with_ocr(content)
            
    except Exception as e:
        print(f"Error extrayendo PDF: {e}")
        result['text'] = await extract_pdf_with_ocr(content)
    
    return result


async def extract_pdf_with_ocr(content: bytes) -> str:
    """Extrae texto de PDF usando OCR"""
    try:
        images = convert_from_bytes(content, dpi=200)
        text_parts = []
        
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='spa+eng')
            text_parts.append(f"--- Página {i + 1} (OCR) ---\n{text}")
        
        return '\n\n'.join(text_parts)
    except Exception as e:
        return f"Error en OCR: {str(e)}"


async def extract_docx(content: bytes) -> Dict[str, Any]:
    """Extrae texto de archivos Word"""
    doc_stream = io.BytesIO(content)
    doc = Document(doc_stream)
    
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    
    return {
        'text': '\n\n'.join(text_parts),
        'paragraphs': len(doc.paragraphs)
    }


async def extract_excel(content: bytes) -> Dict[str, Any]:
    """Extrae datos de archivos Excel"""
    excel_stream = io.BytesIO(content)
    workbook = openpyxl.load_workbook(excel_stream, data_only=True)
    
    all_text = []
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        for row in sheet.iter_rows():
            row_values = [str(cell.value) for cell in row if cell.value is not None]
            if row_values:
                all_text.append(' | '.join(row_values))
    
    return {
        'text': '\n'.join(all_text),
        'total_sheets': len(workbook.sheetnames)
    }


async def extract_pptx(content: bytes) -> Dict[str, Any]:
    """Extrae texto de PowerPoint"""
    pptx_stream = io.BytesIO(content)
    presentation = Presentation(pptx_stream)
    
    all_text = []
    for i, slide in enumerate(presentation.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text.append(shape.text)
        
        if slide_text:
            all_text.append(f"--- Diapositiva {i + 1} ---\n" + '\n'.join(slide_text))
    
    return {
        'text': '\n\n'.join(all_text),
        'total_slides': len(presentation.slides)
    }


async def extract_image(content: bytes) -> Dict[str, Any]:
    """Extrae texto de imágenes usando OCR"""
    image = Image.open(io.BytesIO(content))
    text = pytesseract.image_to_string(image, lang='spa+eng')
    return {'text': text.strip()}


async def extract_text(content: bytes) -> Dict[str, Any]:
    """Extrae texto de archivos de texto plano"""
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('latin-1')
    return {'text': text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
