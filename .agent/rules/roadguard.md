# RoadGuard AI — Project Rules & Guidelines

## 1. Project Overview & Architecture Stack
- **Project Name:** RoadGuard AI
- **Domain:** Road damage detection, crowdsourced/sensor reporting, and inspector priority dispatch system.
- **Technology Stack:**
  - **Frontend:** React
  - **Backend:** FastAPI (Python)
  - **Machine Learning / Vision:** Ultralytics YOLOv8 (Road damage detection: e.g., D00, D10, D20, D40)
  - **Database & Spatial Engine:** PostgreSQL + PostGIS extension
  - **Containerization & Deployment:** Docker & Docker Compose

---

## 2. Core Operational Rules

### Rule 1: Human-in-the-Loop & Verification Integrity
- AI model inference outputs are strictly **suggestions/recommendations**.
- Automated detections **MUST NOT** directly trigger work orders or bypass human review.
- The **Ranked Inspection Queue** must contain **only inspector-verified reports**. Unverified model detections remain in a triage/review stage until explicitly confirmed or adjusted by a qualified inspector.

---

### Rule 2: Dynamic Priority Scoring Formula
Priority Score \(P\) is normalized between **0 and 100** and computed as:

\[
P = 0.50 \cdot S + 0.30 \cdot R + 0.20 \cdot W
\]

All component scores (\(S\), \(R\), \(W\)) evaluate strictly to the range `[0, 100]`.

#### A. Severity Score (\(S\))
Computed across all detected damage bounding boxes for a report:
\[
S = \max_{\text{detections}} \left[ B_{\text{class}} \cdot (0.7 + 0.3 \cdot a) \right]
\]
Where:
- \(a = \min\left(1.0,\, \frac{\text{box\_area}}{0.25 \cdot \text{image\_area}}\right)\)
- **Base Severity Weights (\(B_{\text{class}}\)):**
  - **D00** (Longitudinal Crack): `30`
  - **D10** (Transverse Crack): `30`
  - **D20** (Alligator / Fatigue Crack): `60`
  - **D40** (Pothole): `85`

#### B. Road Importance & Context Score (\(R\))
\[
R = 0.5 \cdot R_{\text{class}} + 0.5 \cdot R_{\text{sens}}
\]
Where:
- **Road Classification (\(R_{\text{class}}\)):**
  - Local road: `25`
  - Arterial road: `60`
  - Highway / Expressway: `100`
- **Sensitivity Zone (\(R_{\text{sens}}\)):**
  - `100` if within **200 meters** of a school, hospital, or critical transit junction.
  - `0` otherwise.
- **Missing Road Data Fallback:** If spatial or road attribute data is unavailable or cannot be resolved, assign default \(R = 50\) and automatically flag the record with `manual_review_required = true`.

#### C. Aging & Report Frequency Score (\(W\))
\[
W = 100 \cdot \left[ 0.6 \cdot \min\left(1.0, \frac{t}{30}\right) + 0.4 \cdot \min\left(1.0, \frac{n - 1}{4}\right) \right]
\]
Where:
- \(t\) = Elapsed time in days since initial report submission.
- \(n\) = Total number of aggregated/duplicate reports clustered to this damage site (\(n \ge 1\)).

#### D. Priority Classification Bands
- **0 – 39:** Low Priority
- **40 – 59:** Medium Priority
- **60 – 79:** High Priority
- **80 – 100:** Critical Priority

---

### Rule 3: Data Segregation & Immutability
- **Separate Tables:** Machine model inferences and human inspector verifications must be stored in distinct, decoupled tables (e.g., `detections` vs `verifications`).
- **Immutability of Inferences:** Raw AI model outputs (bounding boxes, confidence scores, detected classes, image references) must **never be overwritten, deleted, or mutated** by human verification edits. Inspector overrides are recorded as distinct verification entities referencing the detection.

---

### Rule 4: Comprehensive Audit Trail
- Every status transition (e.g., `SUBMITTED` $\to$ `DETECTED` $\to$ `INSPECTOR_VERIFIED` $\to$ `WORK_SCHEDULED` $\to$ `RESOLVED` $\to$ `REJECTED`) must be captured in a dedicated `StatusHistory` table.
- Each entry must record:
  - Reference to report ID
  - Previous status & new status
  - Acting user ID (or system process ID)
  - UTC timestamp
  - Optional change reason / notes

---

### Rule 5: Geospatial Duplicate Clustering
- Nearby duplicate submissions must be clustered using PostGIS spatial indexing and spatial distance operators (e.g., `ST_DWithin` on `geography`).
- **Clustering Threshold:** Same damage class detected within a **15-meter radius**.
- Aggregated duplicates must increment the cluster count (\(n\)) for the primary parent incident to dynamically update weight score \(W\).

---

### Rule 6: Zero Synthetic / Hardcoded Metrics & Mandatory Testing
- **No Mocked Production Metrics:** Never hardcode metrics, mock model results in production paths, or fabricate benchmark outputs.
- **Test-Driven Rigor:** Every backend and data processing module must be accompanied by comprehensive automated tests (unit and integration tests with pytest, verifying spatial queries, priority calculations, and state transitions).
