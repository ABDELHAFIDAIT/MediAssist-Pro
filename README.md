# MediAssist-Pro
Assistant intelligent dédié à la maintenance biomédicale. Grâce à une architecture RAG optimisée sous FastAPI et LangChain, il indexe vos manuels techniques pour fournir des guides de dépannage précis, sourcés et instantanés. Sécurisé par JWT et conteneurisé sous Docker, il réduit les délais d’arrêt technique des laboratoires.


```bash
mediassist-pro/
├── app/
│   ├── api/                # Points d'entrée de l'API (Routes)
│   │   ├── endpoints/
│   │   │   ├── auth.py      # Login, JWT, Registration
│   │   │   ├── chat.py      # Pipeline RAG (Query/Response)
│   │   │   └── documents.py # Upload et Indexation
│   │   └── api.py           # Agrégateur de routes
│   ├── core/               # Configuration globale
│   │   ├── config.py            # Pydantic Settings (.env)
│   │   ├── security.py          # Logique JWT et Hashing
│   │   └── exceptions.py        # Gestionnaire d'erreurs centralisé
│   ├── db/                 # Persistance (PostgreSQL)
│   │   ├── base.py              # Import de tous les modèles pour SQLAlchemy
│   │   ├── session.py           # Connexion Engine/Session
│   │   └── models/              # Modèles SQLAlchemy (User, Query)
│   ├── schemas/            # Validation Pydantic
│   │   ├── user.py
│   │   ├── chat.py
│   │   └── token.py
│   ├── services/           # Logique métier & Pipeline RAG
│   │   ├── rag_service.py       # Coeur LangChain (Chunking, Retrieval, LLM)
│   │   ├── vector_store.py      # Interface avec ChromaDB/FAISS
│   │   └── user_service.py      # CRUD Utilisateurs
│   └── main.py             # Initialisation de l'application FastAPI
├── data/
│   ├── vector_store/      # Données indexées de ChromaDB
│   └── uploads/           # PDF sources (ex: manuels ELISA) [cite: 9, 136]
├── migrations/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/          # Historique des changements PostgreSQL
├── tests/
│   ├── conftest.py        # Fixtures globales
│   ├── test_api/          # Tests d'intégration (FastAPI)
│   └── test_services/     # Tests unitaires (LangChain, Business Logic)                 # Tests unitaires et d'intégration (Pytest)
├── docker-compose.yml      # Orchestration (API, Postgres, VectorDB)
├── Dockerfile              # Image Docker de l'application
├── requirements.txt        # Dépendances Python
└── .env                    # Variables d'environnement (Secret Keys, API Keys)
```