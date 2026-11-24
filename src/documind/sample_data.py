"""Generate a realistic 56-page enterprise engineering & product handbook PDF for demos and RAG evaluation."""

from __future__ import annotations

from pathlib import Path
from fpdf import FPDF, XPos, YPos

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1. Identity, OAuth 2.1, and Session Security",
        [
            "NovaCloud enforces OAuth 2.1 with Proof Key for Code Exchange (PKCE) using the S256 code challenge method across all public and confidential client applications.",
            "Short-lived JSON Web Tokens expire after 15 minutes, while rotating refresh tokens remain valid for 14 days before requiring fresh interactive sign-in.",
            "Service-to-service workloads inside private VPC subnets authenticate through short-lived mTLS SPIFFE certificates issued every 4 hours by the workload identity broker.",
            "Administrative console sessions require hardware FIDO2 security keys or WebAuthn passkeys; SMS-based second factors are prohibited for production infrastructure access.",
        ],
    ),
    (
        "2. Vector Indexing and FAISS Tuning Standards",
        [
            "Document retrieval pipelines normalize embedding vectors to unit L2 length prior to indexing so inner-product similarity scores correspond directly to cosine similarity.",
            "Collections containing fewer than 100,000 passages default to IndexFlatL2 for exact nearest-neighbor recall without quantization loss.",
            "Datasets scaling beyond 500,000 vectors migrate to IndexIVFPQ with nlist=1024 and nprobe=32 to keep p95 retrieval latency below 18 milliseconds.",
            "Recommended chunking windows span 750 to 900 characters with a 150-character overlap so multi-sentence definitions stay intact across boundary splits.",
        ],
    ),
    (
        "3. API Gateway Rate Limits and Backoff Policy",
        [
            "Standard workspace tenants receive a sustained quota of 600 requests per minute and burst allowance of 120 concurrent connections per regional edge POP.",
            "Clients receiving HTTP 429 Too Many Requests must honor the Retry-After header and apply jittered exponential backoff starting at 500 milliseconds up to 32 seconds.",
            "Enterprise tier accounts may request dedicated token-bucket isolation with throughput guarantees up to 4,500 requests per minute via the capacity portal.",
            "Repeated unauthenticated bursts exceeding 300 failed calls inside 60 seconds trigger an automatic 10-minute edge WAF cooldown on the offending source CIDR.",
        ],
    ),
    (
        "4. Incident Command and Severity Classification",
        [
            "Severity 1 (SEV-1) incidents represent complete regional service unavailability or verified customer data exposure and require Incident Commander assignment within 5 minutes.",
            "SEV-2 outages denote degraded throughput above 15 percent error rate or broken background job queues and trigger automated PagerDuty escalation within 15 minutes.",
            "Post-incident reviews must be drafted within 48 hours of mitigation, focusing on systemic safeguards and blameless root-cause timelines rather than individual error.",
            "Customer status page updates during active SEV-1 events publish every 20 minutes until three consecutive synthetic probes pass across all availability zones.",
        ],
    ),
    (
        "5. Service Level Agreements and Availability Credits",
        [
            "NovaCloud guarantees 99.95 percent monthly uptime for Enterprise Control Plane APIs across multi-AZ deployments in us-east, eu-central, and ap-south.",
            "Monthly availability falling between 99.90 and 99.95 percent entitles affected enterprise organizations to a 10 percent invoice credit on the impacted SKU.",
            "Availability dropping below 99.0 percent within a calendar month raises the service credit tier to 30 percent when claimed within 30 days of billing close.",
            "Scheduled maintenance windows announced at least 7 days in advance and lasting under 20 minutes are excluded from monthly SLA downtime calculations.",
        ],
    ),
    (
        "6. Data Retention, Archival, and Erasure",
        [
            "Raw application telemetry and structured JSON audit logs remain queryable in hot NVMe storage for 30 days before transitioning to compressed cold object storage for 365 days.",
            "Customer-uploaded PDF documents are encrypted at rest with AES-256-GCM envelope keys and purged within 24 hours of workspace deletion confirmation.",
            "GDPR Article 17 erasure requests propagate across primary PostgreSQL replicas and derived vector caches within 72 hours of ticket verification.",
            "Immutable financial ledger records are retained for 7 years in write-once-read-many (WORM) compliance buckets to satisfy statutory audit obligations.",
        ],
    ),
    (
        "7. Cloud Billing, Usage Quotas, and Spend Guardrails",
        [
            "Token-metered LLM workloads trigger automated budget alert webhooks at 70 percent, 90 percent, and 100 percent of the configured monthly spending ceiling.",
            "Idle GPU inference endpoints automatically scale to zero after 12 minutes without incoming requests, cutting non-production compute spend by roughly 62 percent.",
            "Annual committed-use reservations provide a 28 percent discount on baseline CPU clusters and 35 percent savings on dedicated A100/H100 inference nodes.",
            "Overdue invoices past 21 days enter a 7-day grace state with read-only dashboard access before non-critical batch pipelines are suspended.",
        ],
    ),
    (
        "8. Kubernetes Autoscaling and Workload Scheduling",
        [
            "Horizontal Pod Autoscalers target 65 percent sustained CPU utilization for stateless web APIs and custom queue-depth metrics for asynchronous ingestion workers.",
            "Stateful vector database pods use anti-affinity topology spread constraints across at least three availability zones to survive single-zone power interruptions.",
            "Every production deployment defines explicit memory limits equal to memory requests so the Linux kernel OOM killer never evicts co-located latency-sensitive pods.",
            "Cluster Autoscaler provisions fresh compute nodes within 90 seconds using warm pre-pulled container images cached on regional registry mirrors.",
        ],
    ),
    (
        "9. Observability, Distributed Tracing, and SLOs",
        [
            "All microservices emit OpenTelemetry traces sampled at 10 percent for nominal traffic and 100 percent tail-based sampling whenever latency exceeds 1.5 seconds or HTTP 5xx occurs.",
            "The Retrieval Gateway SLO enforces p99 response time under 450 milliseconds and a 99.9 percent success ratio measured over rolling 28-day error budgets.",
            "When an error budget burn rate exceeds 14.4x over a 1-hour window, automated release freezes block non-hotfix merges across the affected repository.",
            "Structured logs omit bearer tokens, email addresses, and raw customer payloads through an enforced edge PII redaction filter before ingestion.",
        ],
    ),
    (
        "10. Disaster Recovery and Backup Verification",
        [
            "Production relational databases maintain synchronous cross-AZ replication for zero data loss (RPO = 0) and automated failover completing in under 45 seconds (RTO < 1 min).",
            "Cross-region asynchronous snapshots replicate every 15 minutes to a secondary geographic region, guaranteeing a regional disaster RPO under 15 minutes and RTO under 20 minutes.",
            "Quarterly chaos-engineering game days restore encrypted snapshots into an isolated staging VPC and validate row checksums and schema migrations end to end.",
            "Vector index snapshots pair with manifest version hashes so stale embeddings are automatically re-indexed if schema upgrades alter tokenization.",
        ],
    ),
    (
        "11. Secret Management and Key Rotation",
        [
            "Static API keys and database passwords are prohibited in git repositories; pre-commit hooks and CI secret scanners reject any commit matching high-entropy key patterns.",
            "Envelope Data Encryption Keys rotate automatically every 90 days via KMS without requiring downtime or re-encrypting historical cold archives synchronously.",
            "External vendor API credentials reside exclusively in HashiCorp Vault dynamic secrets engines with 24-hour lease TTLs and per-service IAM role scoping.",
            "Compromised credential alerts trigger immediate token revocation across the edge gateway within 3 seconds through Redis pub/sub invalidation channels.",
        ],
    ),
    (
        "12. Production Release and Canary Rollout Checklist",
        [
            "Every production change advances through a canary stage receiving 5 percent of live traffic for 15 minutes while automated metric analyzers compare p95 latency and error rates.",
            "Database schema migrations strictly follow expand-and-contract patterns so previous container versions remain backwards-compatible during zero-downtime rollbacks.",
            "Feature flags decouple code deployment from customer feature exposure, allowing instant kill-switch toggling without triggering a full container rebuild.",
            "Release artifacts carry SLSA Level 3 provenance attestations signed in CI/CD before admission controllers permit scheduling on production clusters.",
        ],
    ),
    (
        "13. Remote Work, Equipment, and Travel Expense Policy",
        [
            "Full-time engineering staff receive an annual $1,200 home-office ergonomics and peripheral stipend reimbursable within 30 days of purchase receipts.",
            "Conference attendance including airfare, lodging, and workshop registration is covered up to $2,800 per calendar year with manager pre-approval.",
            "Team offsites prioritize rail travel for trips under 4 hours and reimburse economy-class flights plus reasonable meal allowances up to $85 per day.",
            "Company laptops enforce full-disk FileVault encryption, 5-minute screen lock timeouts, and automatic OS security patch installation within 72 hours of release.",
        ],
    ),
    (
        "14. Compliance, SOC 2 Type II, and Vendor Audits",
        [
            "NovaCloud completes annual SOC 2 Type II and ISO 27001 external audits covering Security, Availability, and Confidentiality trust service criteria.",
            "New third-party SaaS subprocessors undergo a mandatory Data Protection Impact Assessment and legal DPA review prior to ingesting customer metadata.",
            "Production SSH bastion access is disabled by default; break-glass debugging sessions require dual-party approval and record full terminal session transcripts.",
            "Quarterly access reviews revoke stale repository and cloud console permissions for contractors and role-transferring employees within 24 hours.",
        ],
    ),
]


def _sanitize_latin1(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_handbook_pdf(output_path: Path, target_pages: int = 56) -> Path:
    """Create a 56-page structured PDF handbook with unique per-page operational specifics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)

    # Cover page (Page 1)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(30)
    pdf.cell(0, 14, "NovaCloud Engineering & Operations Handbook", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 13)
    pdf.ln(6)
    pdf.cell(0, 10, "Reference Manual v4.2 - Architecture, Security, RAG, and Reliability", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(16)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        7,
        _sanitize_latin1(
            "This 56-page reference manual documents NovaCloud's production standards across "
            "identity and authentication, FAISS vector indexing, API rate limits, incident management, "
            "SLA guarantees, data retention, cloud cost controls, Kubernetes scheduling, observability, "
            "disaster recovery, key rotation, canary rollouts, workplace policies, and compliance."
        ),
    )

    # Pages 2..target_pages: 4 pages per section across 14 sections = 56 pages total.
    page_num = 1
    for sec_idx, (title, paragraphs) in enumerate(SECTIONS):
        for sub_idx in range(4):
            if page_num >= target_pages:
                break
            pdf.add_page()
            page_num += 1
            pdf.set_font("Helvetica", "B", 15)
            pdf.cell(0, 10, _sanitize_latin1(f"{title} (Part {sub_idx + 1}/4)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 11)

            # Rotate lead paragraph and add concrete operational guidance per page.
            rotated = paragraphs[sub_idx:] + paragraphs[:sub_idx]
            for p_i, para in enumerate(rotated, start=1):
                pdf.multi_cell(0, 6.5, _sanitize_latin1(f"Clause {sec_idx + 1}.{sub_idx + 1}.{p_i}: {para}"))
                pdf.ln(2.5)

            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(
                0,
                6,
                _sanitize_latin1(
                    f"Operational Verification Note (Page {page_num}): Platform owners review "
                    f"{title.split('.', 1)[1].strip()} controls prior to every quarterly readiness sign-off."
                ),
            )

    pdf.output(str(output_path))
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "samples" / "nova_cloud_handbook_56p.pdf"
    generated = generate_handbook_pdf(out)
    print(f"Generated {generated}")
