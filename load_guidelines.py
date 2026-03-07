"""
MedExpert - Load Clinical Guidelines into RAG
Loads PDF/TXT guidelines into ChromaDB for retrieval.
Each expert has its own guides directory under data/experts/<slug>/guides/.

Usage:
    python load_guidelines.py                    # Load all files from data/guides/
    python load_guidelines.py path/to/guide.pdf  # Load a specific file
    python load_guidelines.py --sample           # Load sample oncology knowledge
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from clinical_brain import RAGEngine

console = Console()


SAMPLE_ONCOLOGY_KNOWLEDGE = [
    {
        "source": "NCCN Breast Cancer v4.2024 - Summary",
        "text": """
Carcinoma ductal infiltrante - Enfoque terapeutico segun estadio:

ESTADIO I-IIA (T1-T2, N0):
- Cirugia conservadora + radioterapia o mastectomia
- Biopsia de ganglio centinela obligatoria
- Si RE/RP positivo, HER2 negativo: terapia endocrina adyuvante (tamoxifeno o inhibidor de aromatasa)
- Considerar Oncotype DX para decision de quimioterapia en N0, RE+, HER2-
- Ki67 <20% y Oncotype RS <26: probable beneficio limitado de quimioterapia

ESTADIO IIB-IIIA:
- Considerar quimioterapia neoadyuvante especialmente si:
  * Triple negativo
  * HER2 positivo
  * Tumor >5cm
  * Ganglios clinicamente positivos
- Esquemas neoadyuvantes: AC-T (doxorrubicina/ciclofosfamida seguido de taxano)
- HER2+: agregar trastuzumab +/- pertuzumab

TERAPIA ENDOCRINA ADYUVANTE:
- Premenopausia: Tamoxifeno 5-10 anos, considerar supresion ovarica + IA si alto riesgo
- Postmenopausia: Inhibidor de aromatasa (letrozol, anastrozol, exemestano) 5-10 anos
- Considerar abemaciclib adyuvante en alto riesgo (N+, Ki67>20%, grado 3)
"""
    },
    {
        "source": "NCCN Colorectal Cancer - Summary",
        "text": """
Cancer colorrectal - Principios de tratamiento:

COLON ESTADIO III (N+):
- Cirugia: colectomia con diseccion ganglionar (minimo 12 ganglios)
- Quimioterapia adyuvante obligatoria:
  * FOLFOX (5-FU/leucovorin/oxaliplatino) x 6 meses - estandar
  * CAPOX (capecitabina/oxaliplatino) alternativa
  * T1-3 N1: considerar 3 meses de CAPOX (estudio IDEA)
  * T4 o N2: 6 meses completos recomendados
- Determinar estatus de MMR/MSI: implicaciones pronosticas y terapeuticas

RECTO LOCALMENTE AVANZADO (T3-4 o N+):
- Neoadyuvancia: TNT (Total Neoadjuvant Therapy) preferido
  * Quimioterapia de induccion (FOLFOX/CAPOX) -> quimioradioterapia -> cirugia
  * O quimioradioterapia -> cirugia -> quimioterapia adyuvante
- Evaluar respuesta clinica completa: considerar watch & wait en casos selectos

ENFERMEDAD METASTASICA:
- Determinar RAS/BRAF/MSI/HER2
- RAS wild-type, lado izquierdo: anti-EGFR (cetuximab/panitumumab) + quimioterapia
- RAS mutado: bevacizumab + quimioterapia
- MSI-H/dMMR: inmunoterapia (pembrolizumab primera linea)
- BRAF V600E: encorafenib + cetuximab (segunda linea+)
"""
    },
    {
        "source": "NCCN Lung Cancer NSCLC - Summary",
        "text": """
Cancer de pulmon de celulas no pequenas (CPCNP):

EVALUACION INICIAL:
- Biopsia con material suficiente para perfil molecular completo
- Estudios moleculares obligatorios: EGFR, ALK, ROS1, BRAF, KRAS G12C, MET, RET, NTRK, PD-L1
- PET-CT y RMN cerebral para estadificacion

ESTADIO I-II RESECABLE:
- Cirugia (lobectomia preferida) + diseccion ganglionar mediastinal
- Estadio IB (>=4cm) - IIIA: quimioterapia adyuvante basada en cisplatino
- Considerar atezolizumab adyuvante si PD-L1 >=1% (estadio II-IIIA post-quimio)
- Osimertinib adyuvante si EGFR mutado (estadio IB-IIIA)

ESTADIO III NO RESECABLE:
- Quimioradioterapia concurrente -> durvalumab de consolidacion x 1 ano
- Esquema: cisplatino/etoposido o carboplatino/paclitaxel + RT 60Gy

ESTADIO IV CON DRIVER ONCOGENICO:
- EGFR mut: osimertinib primera linea
- ALK+: alectinib o lorlatinib primera linea
- ROS1+: crizotinib o entrectinib
- KRAS G12C: sotorasib o adagrasib

ESTADIO IV SIN DRIVER, PD-L1:
- PD-L1 >=50%: pembrolizumab monoterapia o + quimioterapia
- PD-L1 1-49%: pembrolizumab + quimioterapia
- PD-L1 <1%: quimioterapia + pembrolizumab o nivolumab/ipilimumab + quimioterapia
"""
    },
    {
        "source": "Guias mexicanas - Referencia rapida",
        "text": """
Consideraciones especificas para practica oncologica en Mexico:

ACCESO A MEDICAMENTOS:
- Cuadro basico del IMSS/INSABI incluye esquemas estandar de quimioterapia
- Trastuzumab biosimilar disponible
- Inmunoterapia: acceso variable segun institucion
- Terapias dirigidas: considerar disponibilidad antes de recomendar

TAMIZAJE:
- Mama: mastografia anual a partir de los 40 anos (NOM-041-SSA2-2011)
- Cervix: citologia cada 3 anos 25-64 anos, VPH cada 5 anos 35-64 anos
- Colorrectal: colonoscopia cada 10 anos a partir de los 50 anos

REFERENCIA OPORTUNA:
- Sospecha de cancer: referencia a segundo nivel en <10 dias habiles
- Confirmacion diagnostica: inicio de tratamiento en <20 dias habiles
- Programa de Accion Especifico para la Prevencion y Control del Cancer

REGISTRO:
- Obligatorio reportar al Registro Nacional de Cancer (SINBA/DGE)
"""
    },
]


def load_pdf(filepath: str, rag: RAGEngine, category: str = "") -> int:
    """Load a PDF file and chunk it into the RAG store. Returns chunk count."""
    import fitz  # PyMuPDF
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()

    source = Path(filepath).stem
    before = rag.get_total_count()
    rag.load_text(text, source=source, category=category)
    return rag.get_total_count() - before


def load_text_file(filepath: str, rag: RAGEngine, category: str = "") -> int:
    """Load a text file into the RAG store. Returns chunk count."""
    text = Path(filepath).read_text(encoding="utf-8")
    source = Path(filepath).stem
    before = rag.get_total_count()
    rag.load_text(text, source=source, category=category)
    return rag.get_total_count() - before


def load_file(filepath: str, rag: RAGEngine, category: str = "") -> int:
    """Load a single file (PDF or TXT) into the RAG store. Returns chunk count."""
    filepath = str(filepath)
    if filepath.endswith(".pdf"):
        return load_pdf(filepath, rag, category=category)
    elif filepath.endswith((".txt", ".md")):
        return load_text_file(filepath, rag, category=category)
    else:
        console.print(f"[yellow]Unsupported file type: {filepath}[/yellow]")
        return 0


def load_from_directory(guides_dir: str, rag: RAGEngine):
    """Load all PDF/TXT files from a directory, using subdirectories as categories."""
    guides_path = Path(guides_dir)
    if not guides_path.exists():
        console.print(f"[yellow]No guides directory found at {guides_path}[/yellow]")
        return 0

    count = 0
    # Top-level files (no category)
    for f in guides_path.glob("*.pdf"):
        console.print(f"[cyan]Loading {f.name}...[/cyan]")
        load_pdf(str(f), rag)
        count += 1
    for f in guides_path.glob("*.txt"):
        console.print(f"[cyan]Loading {f.name}...[/cyan]")
        load_text_file(str(f), rag)
        count += 1

    # Subdirectories as categories
    for subdir in sorted(guides_path.iterdir()):
        if not subdir.is_dir():
            continue
        category = subdir.name
        for f in sorted(subdir.glob("*")):
            if f.suffix in (".pdf", ".txt", ".md"):
                console.print(f"[cyan]Loading {category}/{f.name}...[/cyan]")
                load_file(str(f), rag, category=category)
                count += 1

    return count


def load_sample(rag: RAGEngine):
    """Load sample oncology knowledge into RAG."""
    console.print("[cyan]Loading sample oncology knowledge base...[/cyan]")

    for entry in SAMPLE_ONCOLOGY_KNOWLEDGE:
        rag.load_text(entry["text"], source=entry["source"])

    console.print(f"[green]✓ Loaded {len(SAMPLE_ONCOLOGY_KNOWLEDGE)} sample guideline summaries[/green]")


def main():
    rag = RAGEngine()
    rag.initialize()

    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        load_sample(rag)
        return

    if len(sys.argv) > 1:
        load_file(sys.argv[1], rag)
        return

    # Load all files from data/guides/ (with subdirectory categories)
    guides_dir = "data/guides"
    guides_path = Path(guides_dir)
    if not guides_path.exists():
        console.print(f"[yellow]No guides directory found at {guides_path}[/yellow]")
        console.print("[cyan]Loading sample knowledge base instead...[/cyan]")
        load_sample(rag)
        return

    count = load_from_directory(guides_dir, rag)

    if count == 0:
        console.print(f"[yellow]No PDF/TXT files found in {guides_path}[/yellow]")
        console.print("[cyan]Loading sample knowledge base instead...[/cyan]")
        load_sample(rag)
        return

    console.print(f"\n[green]✓ All guidelines loaded ({rag.collection.count()} chunks total)[/green]")


if __name__ == "__main__":
    main()
