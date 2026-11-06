# Prayog

Prayog is an IIT Kanpur research-platform project. This repository currently contains the **FastAPI backend** for user access, research-project management, ML experiment tracking, artifact storage, reproducible rebuilds, and controlled compute-node enrollment.

The main workflow is to create a project and model, track experiments and runs, store verifiable code/weight/log artifacts, record immutable run snapshots, and request isolated rebuilds from those snapshots. The backend also provides role-restricted administrative and professor views, project handover, and guarded event handoff for future AI services.

> **Scope note:** This repository is backend-focused. It does not currently contain a frontend application, a production GPU scheduler, or configured live Steward/optimization services. The AI integration contract and safety gates are implemented, but production service-to-service transport and downstream AI consumers are not configured.

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [Run the API](#run-the-api)
- [Create the first administrator](#create-the-first-administrator)
- [ML Flow quick start](#ml-flow-quick-start)
- [Run the execution worker](#run-the-execution-worker)
- [Artifact storage](#artifact-storage)
- [Node-agent enrollment](#node-agent-enrollment)
- [Tests](#tests)
- [Configuration reference](#configuration-reference)
- [Security and operational notes](#security-and-operational-notes)
- [Detailed documentation](#detailed-documentation)

## Capabilities

### Accounts and access control
- User authentication using bearer JWTs and database-backed roles.
- Role-protected administrative and professor endpoints.
- Access-request and professor-related APIs.
- User-owned ML Flow resources; attempts to access another user's resources are scoped as not found.

### ML Flow experiment lifecycle
- Separate user-owned projects and models, with experiments belonging to both.
- Execution runs, hyperparameters, environment metadata, and metrics.
- Individual metric logging and atomic metric batches.
- Code, weights, and log artifact uploads with server-computed integrity metadata.
- Immutable snapshots with explicit `RECORDED` or `INCOMPLETE` status.
- Run resubmission from a completed run with a verified reusable code artifact.
- Snapshot-based rebuild attempts, status history, and metric comparison.

### Reproducibility and artifact safety
- Content-addressed artifact storage with integrity verification.
- Local storage and S3-compatible object storage modes.
- Project snapshot packaging with file inventory and chunk reuse.
- Archive path, symlink, entrypoint, and output-file validation.
- Private artifact retrieval; artifacts are not exposed as static public files.

### Worker execution
- A separate worker process claims queued first-time runs and rebuild attempts.
- User code executes in a restricted Docker container, not in the API process.
- Worker claims, stale-claim recovery, bounded logs, timeouts, and queue ordering.
- Optional node capability inventory checks to fail fast when no provisioned node can satisfy a job's declared requirements.

### Administration and guarded AI handoff
- Admin-only node-agent enrollment-token issuance, node listing, and credential revocation.
- Project IP handover with artifact inventory verification.
- Professor-facing, read-only experiment metadata views.
- Transactional outbox events with leased delivery, retries, and idempotency safeguards.
- Steward diagnosis eligibility only after a persisted failed rebuild with a verified log; optimization eligibility only after a persisted successful rebuild.

The outbox is a handoff mechanism, not a live AI integration. No configured production Steward client, optimization service, or service-to-service transport is included.

## Architecture

```text
Client / CLI
    |
    | HTTPS + Prayog user JWT
    v
FastAPI application (backend/main.py)
    |
    +-- Routers: authentication, access requests, ML Flow,
    |             professor/admin views, handover, node agents
    |
    +-- Logic layer: ownership checks, lifecycle transitions,
    |                artifact verification, authorization
    |
    +-- SQLAlchemy models + Alembic migrations
    |                 |
    |                 v
    |              PostgreSQL
    |
    +-- Artifact storage: private local volume or S3-compatible bucket
    |
    +-- Transactional outbox
                      |
                      v
             Separate outbox dispatcher
             (requires configured handler module)

Separate ML Flow execution worker
    |
    +-- Claims queued run/rebuild work from PostgreSQL
    +-- Reads verified artifacts from configured storage
    +-- Runs project code in a restricted Docker container
    +-- Persists execution result, logs, and output artifacts
```

The API, execution worker, and outbox dispatcher are separate processes. In local artifact-storage mode, the API and execution worker must share the same private durable artifact directory. In S3 mode, both require the same bucket configuration and authorized credentials.

## Repository layout

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI application, middleware, exception handling, router registration, health endpoint |
| `backend/routers/` | HTTP routes for authentication, access requests, ML Flow, handover, professors, administration, and node agents |
| `backend/logic/` | Business logic, lifecycle rules, authorization, artifact storage, outbox, and diagnostic safeguards |
| `backend/schemas/` | Request and response schemas |
| `backend/models.py`, `backend/mlflow_models.py`, `backend/node_agent_models.py` | Database models |
| `backend/alembic/` | Database migration environment and migration history |
| `backend/mlflow_cli.py` | Command-line client for ML Flow API workflows |
| `backend/mlflow_worker.py` | Separate Docker-backed run/rebuild execution worker |
| `backend/mlflow_outbox_worker.py` | Transactional outbox dispatcher |
| `backend/worker/Dockerfile` | Default Python runtime image for isolated project execution |
| `backend/scripts/create_admin.py` | Initial administrator bootstrap |
| `backend/tests/` | Backend automated tests and test-suite instructions |
| `backend/ML_FLOW.md` | Detailed lifecycle, CLI, worker, artifact, and API guide |
| `backend/AI_INTEGRATION.md` | Steward/optimization eligibility and outbox contract |
| `backend/NODE_AGENT_ENROLLMENT.md` | Node-agent enrollment and authentication guide |
| `docker-compose.yml` | Local PostgreSQL service |

## Requirements

- Python 3.10 or newer (the worker's default runtime image is Python 3.12).
- PostgreSQL 16 for the supplied local Docker Compose configuration.
- Docker Engine for isolated run/rebuild execution.
- Git.
- An S3-compatible object store and AWS SDK credentials only if S3 artifact storage is selected.

The API can be developed and its unit tests can be run without starting the Docker execution worker. PostgreSQL is required for a functional local API and database-backed integration tests. Docker is required to execute submitted project archives through the worker.

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/baymaxisme/Prayog.git
cd Prayog
```

Switch to the branch you intend to work on if it is not the repository's default branch.

### 2. Configure environment variables

Create a root-level `.env` file. Do not commit it.

```dotenv
POSTGRES_USER=prayog
POSTGRES_PASSWORD=replace-with-a-strong-local-password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prayog
JWT_SECRET_KEY=replace-with-a-long-random-secret
```

The database module requires the PostgreSQL username, password, host, database name, and JWT secret. The port defaults to `5432` in the Python database configuration; Docker Compose uses `POSTGRES_PORT` for the host port mapping, so define it in `.env` when using the supplied Compose file.

Generate a strong, unique JWT secret for each environment. Do not reuse a production secret in development or tests.

### 3. Start PostgreSQL

From the repository root:

```bash
docker compose up -d postgres
docker compose ps
```

The Compose service persists database data in the `postgres_data` named volume.

### 4. Install backend dependencies

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Apply database migrations

From the `backend` directory:

```bash
alembic upgrade head
```

Use Alembic migrations to create and evolve the schema. Do not use ad hoc table creation as a substitute for applying the migration history.

## Run the API

Start the development server from the `backend` directory:

```bash
uvicorn main:app --reload
```

Default local URLs:

- API base: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`
- Health check: `http://localhost:8000/health`

The health endpoint checks database connectivity. It returns HTTP 200 when the database is reachable and HTTP 503 when it is not.

The configured development CORS origins are `http://localhost:3000` and `http://localhost:5173`. Update the application configuration deliberately when connecting a different frontend origin; do not use unrestricted origins in production.

## Create the first administrator

Self-registration does not allow a user to assign themselves the `ADMIN` role. After applying migrations, create the initial administrator from the `backend` directory:

```bash
python -m scripts.create_admin admin@iitk.ac.in "replace-with-a-strong-password"
```

Use an authorized administrator account and a secure local environment. This bootstrap command accepts the password as a process argument, so avoid running it on shared machines or where command-line arguments are exposed. Do not include real credentials in shell transcripts or documentation.

## ML Flow quick start

The CLI is an API client. It does not execute submitted project code itself.

Set the API URL and an existing Prayog bearer token in the environment:

Linux/macOS:

```bash
export PRAYOG_API_URL=http://localhost:8000
export PRAYOG_TOKEN="your-user-bearer-token"
```

Windows PowerShell:

```powershell
$env:PRAYOG_API_URL = "http://localhost:8000"
$env:PRAYOG_TOKEN = "your-user-bearer-token"
```

From `backend`, inspect the available commands:

```bash
python mlflow_cli.py --help
```

The CLI supports project/model/experiment creation, run creation and start/finish, artifact upload/download, metric logging, snapshot operations, rebuild requests and status, resubmission, and result synchronization. For a complete command reference and end-to-end examples, see [backend/ML_FLOW.md](backend/ML_FLOW.md).

A typical lifecycle is:

1. Create a project.
2. Create a model within that project.
3. Create an experiment linked to the project and model.
4. Create a run and record its hyperparameters/environment.
5. Upload a code archive and optional input weights.
6. Start the run. Supplying a verified code artifact ID queues it for the separate worker; omitting it retains the caller-managed execution flow.
7. Record metrics while the run is running.
8. Finish the run and inspect its snapshot.
9. Rebuild only from a verified `RECORDED` snapshot.
10. Inspect the rebuild result and compare metrics where required.

A snapshot may be `INCOMPLETE` if required or supplied artifacts or declared expected metrics are missing or fail verification. Incomplete snapshots remain inspectable but cannot be rebuilt. Terminal run state and snapshot completeness are separate concepts.

## Run the execution worker

Build the default runtime image from the repository root:

```bash
docker build -t prayog-ml-runtime:python3.12 -f backend/worker/Dockerfile backend/worker
```

Then, in a separate terminal, run the worker from the `backend` directory with the same database and artifact-storage configuration as the API:

```bash
python mlflow_worker.py
```

To process at most one queued job and exit:

```bash
python mlflow_worker.py --once
```

The worker handles queued first-time runs and snapshot rebuilds. It validates submitted archives and executes them in restricted Docker containers. The API process does not run user-submitted code.

The worker requires access to a Docker daemon. Treat it as a trusted platform service. Do not expose the Docker socket to submitted project containers. The default runtime image does not install arbitrary project dependencies or access the public network; add reviewed dependencies to an administrator-managed runtime image instead.

For worker timeout, image, archive, recovery, node-capability, and security details, see [backend/ML_FLOW.md](backend/ML_FLOW.md).

## Artifact storage

Local storage is the default. Configure `PRAYOG_ARTIFACT_ROOT` to a private, durable directory writable by the API and worker. The default is `backend/artifacts`. In local mode, every API/worker instance that handles the same jobs must share that private volume.

For S3-compatible storage, configure:

- `PRAYOG_ARTIFACT_STORAGE=s3`
- `PRAYOG_S3_BUCKET`
- Optional `PRAYOG_S3_PREFIX` (default: `prayog-artifacts`)
- Optional `PRAYOG_S3_REGION` or `AWS_DEFAULT_REGION`
- Optional `PRAYOG_S3_ENDPOINT_URL` for compatible providers
- Optional `PRAYOG_S3_SERVER_SIDE_ENCRYPTION` (default: `AES256`)

Credentials are resolved through the standard AWS SDK credential chain. Keep the bucket private, block public access, enforce TLS, and restrict access to the API/worker identities. In S3 mode, `PRAYOG_ARTIFACT_ROOT` is a local cache/staging directory, not the source of truth.

After configuring storage, run the storage smoke check from `backend`:

```bash
python mlflow_storage_smoke.py
```

This check exercises temporary upload, download/checksum verification, and cleanup; it does not create ML Flow database records.

## Node-agent enrollment

The backend includes admin-controlled node-agent enrollment and authenticated heartbeat endpoints. It does **not** implement a complete GPU scheduler or hardware allocation service.

The enrollment flow is:

1. An administrator issues a node-specific, one-time enrollment token using `POST /node-agents/enrollment-tokens`.
2. The node exchanges that token at `POST /node-agents/enroll` for a node-scoped credential.
3. The node authenticates each heartbeat request using that credential.
4. An administrator can list enrolled agents with `GET /node-agents` and revoke a credential with `POST /node-agents/{node_id}/revoke`.

Enrollment tokens expire after 15 minutes and are single-use. The server stores hashes of enrollment tokens and node credentials, not their raw values. See [backend/NODE_AGENT_ENROLLMENT.md](backend/NODE_AGENT_ENROLLMENT.md) for request examples and security details.

## Tests

From the `backend` directory, install the requirements and run:

```bash
python -m pytest tests/
```

For a concise summary:

```bash
python -m pytest tests/ -q
```

To run a specific module:

```bash
python -m pytest tests/test_auth_dependencies.py
```

The tests include unit-level checks and selected integration-oriented workflows. A passing unit-test run does not by itself validate a production PostgreSQL deployment, S3 permissions, Docker isolation, real GPU execution, or live downstream AI services. See [backend/tests/README.md](backend/tests/README.md) for the test inventory, commands, and limitations.

## Configuration reference

| Variable | Purpose |
|---|---|
| `POSTGRES_USER` | PostgreSQL username (required) |
| `POSTGRES_PASSWORD` | PostgreSQL password (required) |
| `POSTGRES_HOST` | PostgreSQL host (required) |
| `POSTGRES_PORT` | PostgreSQL port; Python defaults to `5432`, Compose uses this for host mapping |
| `POSTGRES_DB` | PostgreSQL database name (required) |
| `JWT_SECRET_KEY` | Secret used to validate/sign user JWTs (required by authentication logic) |
| `PRAYOG_ARTIFACT_ROOT` | Private artifact/cache directory; defaults to `backend/artifacts` |
| `PRAYOG_ARTIFACT_STORAGE` | Artifact backend: `local` (default) or `s3` |
| `PRAYOG_S3_BUCKET` | S3 bucket when S3 storage is enabled |
| `PRAYOG_S3_PREFIX` | Object-key prefix; defaults to `prayog-artifacts` |
| `PRAYOG_S3_REGION` / `AWS_DEFAULT_REGION` | S3 region |
| `PRAYOG_S3_ENDPOINT_URL` | Optional S3-compatible endpoint |
| `PRAYOG_S3_SERVER_SIDE_ENCRYPTION` | Server-side encryption setting; defaults to `AES256` |
| `PRAYOG_API_URL` | Base URL used by the ML Flow CLI |
| `PRAYOG_TOKEN` | Existing user bearer token used by the ML Flow CLI |
| `PRAYOG_WORKER_IMAGE` | Administrator-managed local Docker runtime image override |
| `PRAYOG_WORKER_TIMEOUT_SECONDS` | Worker execution timeout (documented range: 1–1800 seconds) |
| `PRAYOG_NODE_CAPABILITY_INVENTORY` | JSON inventory of provisioned node capabilities and GPU units; unset disables impossible-job cancellation |
| `PRAYOG_OUTBOX_HANDLER_MODULE` | Python module exporting the required outbox `HANDLERS` mapping |
| `PRAYOG_OUTBOX_POLL_SECONDS` | Outbox worker polling interval (1–60 seconds) |
| `PRAYOG_OUTBOX_WORKER_ID` | Optional operator-defined outbox worker identifier |

The ML Flow guide documents additional worker and storage behavior. Configure only the variables needed for the services being run.

## Security and operational notes

- Never commit `.env`, JWTs, enrollment tokens, node credentials, AWS credentials, source archives, model weights, or diagnostic logs.
- Use HTTPS and strong environment-specific secrets outside local development.
- Keep artifact buckets and artifact volumes private; do not expose them as static web directories.
- User resources are owner-scoped. Administrative and professor access is role-restricted.
- Submitted project code runs only in the separate restricted Docker worker, not inside the API process.
- Do not install arbitrary user dependencies at execution time or enable network access for submitted jobs.
- Diagnostic links are short-lived, single-use retrieval capabilities for verified logs from failed rebuilds. They are not a trigger for AI diagnosis and must be treated as secrets.
- The transactional outbox enforces event eligibility and delivery safeguards, but live AI consumers and production service authentication still need to be configured.
- Back up PostgreSQL and artifact storage together. A database snapshot without its referenced artifact objects may not be sufficient to reproduce a run.

## Detailed documentation

- [ML Flow lifecycle, CLI, artifacts, snapshots, and workers](backend/ML_FLOW.md)
- [Steward diagnosis and optimization integration contract](backend/AI_INTEGRATION.md)
- [Node-agent enrollment and authentication](backend/NODE_AGENT_ENROLLMENT.md)
- [Backend test suite](backend/tests/README.md)
- [Deliverable 2 package and implementation traceability](DELIVERABLE_2_PACKAGE.md)
