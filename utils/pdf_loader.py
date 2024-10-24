import PyPDF2
import os

def carregar_pdfs(diretorio):
    textos = ""
    for filename in os.listdir(diretorio):
        if filename.endswith('.pdf'):
            caminho_pdf = os.path.join(diretorio, filename)
            leitor_pdf = PyPDF2.PdfReader(caminho_pdf)
            for pagina in leitor_pdf.pages:
                textos += pagina.extract_text()
    return textos
