# RoadGuard AI

> **Automated Road Damage Detection, Crowdsourced Reporting, and Inspector Priority Dispatch System**

RoadGuard AI is an intelligent civic infrastructure platform designed to detect road distresses (such as longitudinal, transverse, alligator cracks, and potholes), aggregate citizen/sensor reports, and prioritize inspection queues using a multi-factor risk formula.

---

## Technology Stack

- **Frontend:** React
- **Backend:** FastAPI (Python)
- **Machine Learning / Vision:** Ultralytics YOLOv8 (e.g., D00, D10, D20, D40 damage classifications)
- **Database & Spatial Engine:** PostgreSQL + PostGIS extension
- **Infrastructure & Containerization:** Docker & Docker Compose

---

## Monorepo Directory Structure

```text
RoadGuardAI/
├── .agent/              # Agent rules and project-specific guidelines
├── ml/                  # Model training scripts, dataset preparation, and evaluation notebooks
├── inference-service/   # Standalone microservice for YOLOv8 model loading and inference
├── backend/             # FastAPI REST API (business logic, spatial queries, priority calculations)
├── frontend/            # React user interface for citizens, field inspectors, and administrators
├── db/                  # Database migrations, PostGIS schemas, seeds, and spatial indices
├── docs/                # Architecture diagrams, API specs, and technical documentation
├── docker-compose.yml   # Multi-container orchestration definition
├── .gitignore           # Git ignore rules for Python, Node, models, and environments
└── README.md            # Project overview and directory documentation
```

### Directory Purposes

- **`ml/`**: Machine learning workspace for data annotation pipelines, YOLOv8 model fine-tuning, hyperparameter tuning, model weights management, and validation metrics analysis.
- **`inference-service/`**: Microservice hosting the trained Ultralytics YOLOv8 model to perform high-throughput inference on submitted images without blocking the main API.
- **`backend/`**: FastAPI application handling REST endpoints, human-in-the-loop inspector verification workflows, dynamic priority scoring, and PostGIS geospatial operations.
- **`frontend/`**: React web application containing the public report submission interface, the ranked inspector dispatch queue, and analytical dashboards.
- **`db/`**: SQL schema definitions, spatial indexing (PostGIS `ST_DWithin` cluster scripts), Alembic migrations, and seed data.
- **`docs/`**: Detailed system architecture, data flow diagrams, priority scoring explanations, and API documentation.

---

## Core Principles

1. **Human-in-the-Loop:** AI model outputs serve as suggestions. Only inspector-verified reports populate the dispatch queue.
2. **Deterministic Priority Scoring:** Ranked priority is calculated via \(P = 0.50 \cdot S + 0.30 \cdot R + 0.20 \cdot W\) (Severity, Road context, Waiting time).
3. **Data Immutability:** Raw model detections and inspector verification records reside in decoupled tables.
4. **Auditability:** Complete state transition tracking via a dedicated `StatusHistory` log.
5. **Geospatial De-duplication:** 15-meter radius spatial clustering via PostGIS for identical damage classes.
