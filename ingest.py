import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────
DATA_FOLDER    = "data"
CHROMA_FOLDER  = "embeddings"
CHUNK_SIZE     = 1500   # was 500 — larger chunks keep legal sections intact
CHUNK_OVERLAP  = 200    # was 50  — more overlap avoids losing context at boundaries

# ── Load embedding model ──────────────────────────────────
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded.\n")

# ── ChromaDB setup ────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_FOLDER)

# Always start fresh — delete old collection and recreate
# This guarantees no stale 500-char chunks pollute the results
try:
    chroma_client.delete_collection(name="legal_docs")
    print("🗑️  Deleted old ChromaDB collection (stale chunks cleared).")
except Exception:
    pass  # Collection didn't exist yet — that's fine

collection = chroma_client.create_collection(name="legal_docs")
print("✅ Fresh ChromaDB collection created.\n")

# ── Law name display mapping (exact filename stems) ───────
LAW_DISPLAY_NAMES = {
    "CPA2019":                                                           "Consumer Protection Act, 2019",
    "RTI-Act_English":                                                   "Right to Information Act, 2005",
    "THE INDIAN PENAL CODE":                                             "Indian Penal Code, 1860",
    "THE INFORMATION TECHNOLOGY ACT, 2000":                              "Information Technology Act, 2000",
    "THE MOTOR VEHICLES ACT, 1988":                                      "Motor Vehicles Act, 1988",
    "THE INDIAN CONTRACT ACT, 1872":                                     "Indian Contract Act, 1872",
    "THE FOOD SAFETY AND STANDARDS ACT, 2006":                           "Food Safety and Standards Act, 2006",
    "THE PROTECTION OF WOMEN FROM DOMESTIC VIOLENCE ACT, 2005":         "Protection of Women from Domestic Violence Act, 2005",
    "THE PROTECTION OF CHILDREN FROM SEXUAL OFFENCES ACT, 2012":        "POCSO Act, 2012",
    "PROVIDENT FUNDS":                                                       "Employees' Provident Funds Act, 1952",
    "the_industrial_disputes_act":                                       "Industrial Disputes Act, 1947",
    "The Code on Wages, 2019":                                           "Code on Wages, 2019",
    "Model-Tenancy-Act-English-02_06_2021":                              "Model Tenancy Act, 2021",
    "THE MAHARASHTRA RENT CONTROL ACT, 1999":                            "Maharashtra Rent Control Act, 1999",
}

def get_law_display_name(filename):
    base = os.path.splitext(filename)[0]  # strip .pdf
    # Exact match first
    if base in LAW_DISPLAY_NAMES:
        return LAW_DISPLAY_NAMES[base]
    # Case-insensitive exact match
    for key, display in LAW_DISPLAY_NAMES.items():
        if key.lower() == base.lower():
            return display
    # Case-insensitive partial match — check if any key word appears in filename
    base_lower = base.lower()
    for key, display in LAW_DISPLAY_NAMES.items():
        if key.lower() in base_lower or base_lower in key.lower():
            return display
    return base

# ── Helper: Extract text + page numbers from PDF ──────────
def extract_pages(pdf_path):
    """Returns list of (page_number, text) tuples, skipping blank pages."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append((i, text))
    doc.close()
    return pages

# ── Helper: Chunk with page tracking ─────────────────────
def split_into_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Flattens all pages into one string, then splits into overlapping chunks.
    Each chunk remembers which PDF page it started on.
    Returns list of (chunk_text, page_number) tuples.
    """
    full_text = ""
    page_boundaries = []  # list of (char_position, page_number)

    for page_num, text in pages:
        page_boundaries.append((len(full_text), page_num))
        full_text += text + "\n"

    def page_at(pos):
        """Return the page number for a given character position."""
        result = page_boundaries[0][1]
        for char_start, page_num in page_boundaries:
            if pos >= char_start:
                result = page_num
            else:
                break
        return result

    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, page_at(start)))
        start += chunk_size - overlap

    return chunks

# ── Main ingestion pipeline ───────────────────────────────
def ingest_all_pdfs():
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Folder '{DATA_FOLDER}' not found! Create it and add your PDFs.")
        return

    pdf_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")])

    if not pdf_files:
        print(f"❌ No PDF files found in '{DATA_FOLDER}/' folder!")
        return

    print(f"📚 Found {len(pdf_files)} PDF(s) to ingest:\n")
    for f in pdf_files:
        print(f"   • {f}  →  {get_law_display_name(f)}")
    print()

    total_chunks = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        law_name = os.path.splitext(pdf_file)[0]
        law_display = get_law_display_name(pdf_file)

        print(f"📄 Processing: {law_display}")

        # 1. Extract text with page numbers
        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print(f"   ❌ Failed to read PDF: {e}")
            continue

        total_chars = sum(len(t) for _, t in pages)
        print(f"   ✅ {len(pages)} pages, {total_chars:,} characters extracted")

        # 2. Split into overlapping chunks
        chunks = split_into_chunks(pages)
        print(f"   ✅ {len(chunks)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        # 3. Generate embeddings
        print(f"   ⏳ Generating embeddings...")
        texts = [c[0] for c in chunks]
        try:
            embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
        except Exception as e:
            print(f"   ❌ Embedding failed: {e}")
            continue

        # 4. Store in ChromaDB with full metadata
        ids = [f"{pdf_file}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source":      pdf_file,
                "law_name":    law_name,
                "law_display": law_display,
                "page":        chunks[i][1],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        try:
            collection.upsert(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"   ❌ ChromaDB upsert failed: {e}")
            continue

        print(f"   ✅ Stored successfully!\n")
        total_chunks += len(chunks)

    # ── Summary ───────────────────────────────────────────
    print("=" * 55)
    print(f"🎉 Ingestion complete!")
    print(f"   PDFs processed : {len(pdf_files)}")
    print(f"   Total chunks   : {total_chunks:,}")
    print(f"   ChromaDB total : {collection.count():,}")
    print(f"   Chunk size     : {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
    print("=" * 55)
    print("\nLaws now available in LegalMind AI:")
    for f in pdf_files:
        print(f"   ⚖️  {get_law_display_name(f)}")

if __name__ == "__main__":
    ingest_all_pdfs()
