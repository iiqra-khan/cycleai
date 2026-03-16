# =============================================================================
# knowledge_base.py — Clinical knowledge base + ChromaDB vector store
# =============================================================================

import chromadb
from chromadb.utils import embedding_functions

CLINICAL_KNOWLEDGE = [
    {
        "id": "ck_001",
        "text": (
            "Progesterone (PDG) rises sharply after ovulation in the luteal phase, "
            "peaking around day 21 of a 28-day cycle. A sharp PDG drop signals "
            "the end of the luteal phase and triggers menstruation. PDG < 3 ng/mL "
            "mid-luteal may indicate anovulation. (ACOG Practice Bulletin)"
        ),
        "metadata": {"topic": "progesterone", "event": "period_start", "source": "ACOG"}
    },
    {
        "id": "ck_002",
        "text": (
            "The LH surge occurs 24-36 hours before ovulation, with LH typically "
            "rising above 25 IU/L at peak. Detection of the LH surge is the gold "
            "standard for predicting ovulation. Basal LH is usually below 15 IU/L "
            "in the follicular phase. (FIGO Reproductive Endocrinology Guidelines)"
        ),
        "metadata": {"topic": "lh_surge", "event": "lh_surge", "source": "FIGO"}
    },
    {
        "id": "ck_003",
        "text": (
            "Dysmenorrhea is caused by prostaglandin release during menstruation. "
            "Cramp severity peaks on days 1-2 of the period. Cramps beginning 1-2 "
            "days before onset make them a predictive symptom. Severe cramps (>=3) "
            "in the late luteal phase are associated with imminent menstruation. "
            "(Journal of Women's Health, 2021)"
        ),
        "metadata": {"topic": "cramps", "event": "period_start", "source": "JWH"}
    },
    {
        "id": "ck_004",
        "text": (
            "The estrogen-to-progesterone ratio (E2:PDG) is a key hormonal balance "
            "indicator. A rising E2:PDG ratio in the late luteal phase, combined "
            "with falling PDG, is a reliable marker for upcoming menstruation. "
            "(Reproductive Biology and Endocrinology, 2019)"
        ),
        "metadata": {"topic": "e2_pdg_ratio", "event": "period_start", "source": "RBE"}
    },
    {
        "id": "ck_005",
        "text": (
            "Breast tenderness and bloating are common premenstrual symptoms due to "
            "estrogen and progesterone fluctuations in the luteal phase. They resolve "
            "within 24-48 hours after menstruation. Combining multiple PMS symptom "
            "scores improves period prediction over single-symptom models. (AJOG, 2020)"
        ),
        "metadata": {"topic": "symptoms", "event": "period_start", "source": "AJOG"}
    },
    {
        "id": "ck_006",
        "text": (
            "Estrogen peaks just before the LH surge (typically > 200 pg/mL), "
            "triggering LH release via positive feedback on the HPG axis. A rapid "
            "estrogen drop after the LH surge confirms ovulation timing. "
            "(Endocrine Reviews, 2018)"
        ),
        "metadata": {"topic": "estrogen", "event": "lh_surge", "source": "ER"}
    },
    {
        "id": "ck_007",
        "text": (
            "A regular menstrual cycle is 21-35 days. ML models incorporating "
            "cycle day features alongside hormonal markers outperform symptom-only "
            "models in predicting period onset. (npj Digital Medicine, 2022)"
        ),
        "metadata": {"topic": "cycle_day", "event": "period_start", "source": "npjDM"}
    },
]


def build_vector_store(collection_name: str = "menstrual_health",
                        persist: bool = False,
                        persist_path: str = "./chroma_db"):
    """
    Builds ChromaDB vector store from CLINICAL_KNOWLEDGE.

    Args:
        collection_name : name for the ChromaDB collection
        persist         : if True, saves to disk (for production)
        persist_path    : path to save ChromaDB (used when persist=True)

    Returns:
        collection, client
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    if persist:
        client = chromadb.PersistentClient(path=persist_path)
    else:
        client = chromadb.Client()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Only add if collection is empty (avoid duplicates on restart)
    if collection.count() == 0:
        collection.add(
            ids       = [d["id"]       for d in CLINICAL_KNOWLEDGE],
            documents = [d["text"]     for d in CLINICAL_KNOWLEDGE],
            metadatas = [d["metadata"] for d in CLINICAL_KNOWLEDGE]
        )
        print(f"  [KB] {len(CLINICAL_KNOWLEDGE)} documents indexed "
              f"in '{collection_name}'")
    else:
        print(f"  [KB] Collection '{collection_name}' already loaded "
              f"({collection.count()} docs)")

    return collection, client