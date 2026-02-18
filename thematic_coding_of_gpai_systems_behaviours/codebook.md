# RQ3 Codebook (Lexicon)

This file contains the codebook/lexicon used for **RQ3 thematic coding**. Each code corresponds to a lexical feature that is matched in model outputs.

## How matching works

- Codes defined as **regex patterns** are combined into a single case-insensitive regex (OR-ed together).
- Codes defined as **token sets** are matched as word-bounded tokens (case-insensitive); `n't` is matched as `n['’]t`.
- Feature counts are computed as the number of matches of each compiled pattern.

## Feature groups (as used in the analysis)

### Process & planning

#### `requirements_terms` — Requirements & specification language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `requirements?`
  - `acceptance criteria`
  - `user stor(?:y|ies)`
  - `spec(?:ification)?s?`
  - `feature request`
  - `change request`
  - `backlog`
  - `groom(?:ing)?`
  - `refinement`
  - `epic(s)?`
  - `definition of done|(?:\b|_)dod\b`
  - `\bmvp\b`
  - `prototype`
  - `non[- ]functional requirements|(?:\b|_)nfrs?\b`
  - `use cases?`
  - `spike(?:s)?`
  - `story mapping|story map`

#### `pm_process_terms` — Product management / planning rituals

- **Type:** regex patterns
- **Patterns/tokens:**
  - `retro(?:spective)?`
  - `stand[- ]?up`
  - `demo`
  - `kanban`
  - `scrum`
  - `story map`
  - `\bokrs?\b|\bkpis?\b`
  - `\bmoscow\b`
  - `\brace\b|\bice\b scoring`
  - `pi planning|safe\b`
  - `\b3 amigos\b`
  - `gantt|critical path`

#### `process_governance_terms` — Process & governance (PRs, reviews, standards)

- **Type:** regex patterns
- **Patterns/tokens:**
  - `code review`
  - `pull request`
  - `\bprs?\b`
  - `merge`
  - `rebase`
  - `cherry[- ]pick`
  - `branch`
  - `\bmain\b`
  - `fast[- ]forward`
  - `standards?`
  - `lint(?:er|ing)`
  - `style guide`
  - `compliance`
  - `policy`
  - `static analysis`
  - `sonarqube`
  - `\brfc\b`
  - `codeowners?`
  - `branch protection`
  - `trunk[- ]based`
  - `dora metrics?|deployment frequency|lead time for changes|change failure rate|\bmttr\b`

#### `estimation_uncertainty` — Estimation & uncertainty language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `estimate(?:s|d|ing)?`
  - `story points?`
  - `t[- ]?shirt sizing`
  - `rough(?:ly)?`
  - `\bapproximately\b`
  - `\babout\b`
  - `confidence interval`
  - `\btbd\b`
  - `\btba\b`
  - `ballpark`
  - `guesstimate`
  - `at (?:most|least)`

#### `time_pressure_deadline` — Time pressure, deadlines, and prioritization

- **Type:** regex patterns
- **Patterns/tokens:**
  - `deadlines?`
  - `\bsprint\b`
  - `iteration`
  - `blocker`
  - `priority`
  - `\bp0\b`
  - `\bp1\b`
  - `roadmap`
  - `milestone`
  - `time[- ]pressure`
  - `capacity`
  - `burn[- ]down`
  - `burn[- ]up`
  - `crunch|overtime`
  - `hard deadline`
  - `slip(?:page)?|overrun`
  - `stretch goal`
  - `cut scope`


### Quality & reliability

#### `bug_failure_terms` — Bugs, failures, crashes, and incidents

- **Type:** regex patterns
- **Patterns/tokens:**
  - `bugs?`
  - `defects?`
  - `crash(?:es)?`
  - `stack trace|traceback|backtrace`
  - `exception`
  - `null pointer|npe|nre`
  - `seg(?:mentation)? fault|segfault|sigsegv|sigabrt`
  - `regression(?:s)?`
  - `fail(?:ed|ure|ing)`
  - `outage`
  - `downtime`
  - `incident`
  - `oom|out[- ]of[- ]memory|oom[- ]?killed`
  - `memory leak`
  - `race condition`
  - `deadlock`
  - `hang|freeze`
  - `time[- ]?out`
  - `rollback`
  - `panic`
  - `assert(?:ion)? failure`
  - `core dump`

#### `testing_quality_terms` — Testing & quality assurance language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `tests?`
  - `testing`
  - `unit test(?:s)?`
  - `integration test(?:s)?`
  - `(?:end|e)[- ]?to[- ]?(?:end|e)`
  - `\be2e\b`
  - `coverage|code coverage`
  - `\bqa\b`
  - `flaky`
  - `repro(?:duce|duction)`
  - `mock(?:s|ing)?`
  - `fixture(?:s)?`
  - `test plan`
  - `linter(?:s)?`
  - `fuzz(?:er|ing)?|property[- ]?based`
  - `mutation testing`
  - `\bpytest\b|\bjunit\b|\btestng\b|\bjest\b|\bmocha\b|\bjasmine\b|\bvitest\b`
  - `selenium|cypress|playwright|testcontainers?`
  - `coverage.py|istanbul|nyc|cobertura`

#### `incident_reliability` — Reliability engineering & incidents (SRE)

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\bon[- ]?call\b`
  - `\bsev[0-4]\b|\bsev[- ]?[0-4]\b`
  - `pagerduty`
  - `blameless`
  - `slo|sla|sli`
  - `error budget`
  - `resilien[ct]e?`
  - `failover`
  - `drill`
  - `\bmttr\b|\bmttd\b|\bmtbf\b`
  - `war room|major incident`
  - `chaos (?:test|engineering)|game day`
  - `circuit breaker`
  - `\bsre\b`
  - `toil`

#### `observability_terms` — Observability (metrics, logs, tracing)

- **Type:** regex patterns
- **Patterns/tokens:**
  - `metrics?`
  - `traces?`
  - `logs?`
  - `prometheus`
  - `grafana`
  - `datadog`
  - `opentelemetry|otel`
  - `alert(?:s|ing)?`
  - `dashboard`
  - `sentry|new relic|honeycomb|jaeger|zipkin|tempo`
  - `elasticsearch|kibana|loki`
  - `trace[- ]?id|span[- ]?id|correlation id`

#### `performance_scalability` — Performance & scalability language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\bperformance\b`
  - `latenc(?:y|ies)`
  - `throughput`
  - `optim(?:ize|ised?|isation)?`
  - `benchmark(?:s)?`
  - `\bscale\b|scalab(?:le|ility)`
  - `\bmemory\b`
  - `\bcpu\b`
  - `\bgc\b`
  - `\bjit\b`
  - `profil(?:e|ing|er)`
  - `concurren(?:t|cy)`
  - `parallel(?:ism|ize)`
  - `\bp95\b|\bp99\b|tail latency`
  - `cold start`
  - `throttling|back[- ]?pressure|rate[- ]?limit(?:ing)?`
  - `heap dump|flame ?graph`

#### `security_privacy_terms` — Security & privacy language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\bsecurity\b`
  - `vulnerabilit(?:y|ies)`
  - `\bxss\b`
  - `sql injection|sqli`
  - `\bcve\b`
  - `encryption|cryptograph(?:y|ic)`
  - `secret(?:s)?`
  - `token`
  - `credential(?:s)?`
  - `\bpii\b`
  - `\bprivacy\b`
  - `\brbac\b`
  - `\boauth\b`
  - `\bjwt\b`
  - `\bmfa\b|\b2fa\b`
  - `csrf`
  - `ssrf`
  - `rce`
  - `sso`
  - `kms`
  - `\btls\b|m[- ]?tls`
  - `\bcors\b`
  - `\bcsp\b|clickjacking`
  - `\bowasp\b|\bcwe\b`
  - `sast|dast|iast|rasp`
  - `sbom|supply chain|code signing|sigstore|cosign`
  - `key rotation|least privilege|secrets? scan(?:ning)?|vault|hsm|fips`


### Engineering debt & coordination

#### `tooling_infra_terms` — Tooling, CI/CD, and infrastructure language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `c[iI][-_/ ]?c[dD]|cicd`
  - `pipeline(?:s)?`
  - `\bbuild\b`
  - `deploy(?:ment)?`
  - `container(?:s)?`
  - `docker`
  - `kubernetes|k8s`
  - `helm`
  - `terraform`
  - `runner(?:s)?`
  - `artifact(?:s)?`
  - `ansible`
  - `packer`
  - `bazel|cmake|ninja|gradle|maven|meson|\bmake(file)?\b`
  - `jenkins|github actions?|gitlab[-_ ]?ci|circleci|travis`
  - `argo[- ]?cd|spinnaker|fluxcd|skaffold`
  - `poetry|pipenv|tox|pre[- ]?commit|nix|conan|vcpkg`
  - `\bnpm\b|\byarn\b|\bpnpm\b|\bnvm\b|turborepo|lerna|\bnx\b|\bvite\b|webpack|rollup|esbuild`

#### `legacy_tech_debt` — Legacy systems & technical debt language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `legacy`
  - `technical debt|\btech debt\b`
  - `monolith(?:ic)?`
  - `refactor(?:ing|ed)?`
  - `rewrite`
  - `workaround`
  - `hack(?:y)?`
  - `deprecat(?:e|ed|ion)`
  - `spaghetti code|bit rot|stop[- ]?gap|shim`

#### `refactoring_design_terms` — Refactoring, design, and architecture language

- **Type:** regex patterns
- **Patterns/tokens:**
  - `refactor(?:ing|ed)?`
  - `design pattern(?:s)?`
  - `\bsolid\b`
  - `\bdry\b`
  - `\bkiss\b`
  - `\byagni\b`
  - `architecture`
  - `module(?:s)?`
  - `interface(?:s)?`
  - `abstraction(?:s)?`
  - `cohesion`
  - `coupling`
  - `\bddd\b|hexagonal|onion architecture|clean architecture|cqrs|event sourcing`

#### `documentation_terms` — Documentation & knowledge management

- **Type:** regex patterns
- **Patterns/tokens:**
  - `doc(?:s|umentation)?`
  - `readme`
  - `\badr\b`
  - `runbook|playbook`
  - `comment(?:s)?`
  - `diagram(?:s)?`
  - `postmortem`
  - `\brca\b`
  - `api reference|swagger|openapi`
  - `sphinx|javadoc|docstring`
  - `release notes|changelog|contributing\.md|faq|how[- ]?to|tutorial`

#### `collaboration_conflict` — Collaboration, stakeholders, and conflict

- **Type:** regex patterns
- **Patterns/tokens:**
  - `stakeholders?`
  - `\bproduct\b`
  - `\bpm\b`
  - `lead`
  - `disagree(?:s|ment)?`
  - `consensus`
  - `conflict`
  - `handoff|handover`
  - `alignment|misaligned`
  - `ownership`
  - `escalat(?:e|ion)s?`
  - `\braci\b|decision log|bikeshedding`


### Product surface

#### `data_db_terms` — Data & databases / pipelines

- **Type:** regex patterns
- **Patterns/tokens:**
  - `schema`
  - `migration(?:s)?|backfill`
  - `index(?:es|ing)?`
  - `replica(?:s|tion)?|read replica`
  - `shard(?:s|ing)?`
  - `vacuum|autovacuum|analyz(e|e)`
  - `explain plan|query plan|slow query`
  - `cache(?:s|ing)?`
  - `queue(?:s|ing)?`
  - `\bacid\b|transaction(?:s)?|isolation|mvcc`
  - `prepared statement|connection pool|orm`
  - `eventual consistency|strong consistency`
  - `kafka|kinesis|pulsar|consumer group|partition|offset|compaction`
  - `spark|flink|beam|airflow|dbt`
  - `snowflake|redshift|bigquery`
  - `parquet|delta lake|iceberg|hudi`
  - `\bpostgres(?:ql)?\b|\bmysql\b|\bsqlite\b|\bredis\b|\bcassandra\b|\bmongo(?:db)?\b|\bcockroach\b`

#### `frontend_perf_accessibility` — Front-end performance & accessibility

- **Type:** regex patterns
- **Patterns/tokens:**
  - `bundle size|tree[- ]?shak(?:e|ing)|code[- ]?splitting|lazy[- ]?load(?:ing)?`
  - `\bcore web vitals?\b`
  - `\bttfb\b|\bcls\b|\blcp\b|\binp\b|\btti\b|\btbt\b`
  - `ssr|csr|ssg|isr|hydration|server[- ]?components?`
  - `lighthouse`
  - `\ba11y\b|accessibilit(y|ies)`
  - `aria[-: ]`
  - `screen reader`
  - `wcag|contrast ratio|tabindex|focus (?:trap|visible)`
  - `service worker|pwa|web worker`

#### `mobile_release` — Mobile engineering & release

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\banr\b`
  - `crashlytics`
  - `testflight`
  - `play store`
  - `\bapk\b`
  - `\bipa\b`
  - `\baab\b|android app bundle`
  - `proguard|r8`
  - `minsdk|targetsdk`
  - `fastlane|code ?push|ota updates?`
  - `keystore|signing`
  - `deeplink|universal link`


### Platform Scope

#### `api_platform_terms` — API & platform integration

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\brest\b|graphq[lL]|g[- ]?rpc|websocket(?:s)?`
  - `openapi|swagger`
  - `pagination|rate[- ]?limit(?:ing)?|idempotent`
  - `version(?:ing)?|semver|conventional commits?`

#### `cloud_platforms` — Cloud platforms & IAM

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\baws\b|\bgcp\b|\bazure\b`
  - `\biam\b`
  - `\bvpc\b`
  - `s3|gcs|blob storage`
  - `lambda|cloud[- ]functions|azure functions`
  - `iam role|assume[- ]?role|\bsts\b|oidc`
  - `ec2|gce|aks|eks|gke|cloud run|fargate|ecs`
  - `sqs|sns|pub/sub`
  - `rds|aurora|dynamodb|cosmos db|bigquery`
  - `step functions|dataflow|event hubs|service bus|api gateway`
  - `kms|key vault|cloud kms`


### Emerging themes

#### `cost_finops` — Cost management / FinOps

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\bcost(?:s)?\b|\bbudget(?:s|ary)?\b|\bspend\b|\bfinops\b`
  - `reserved instances?|savings plans?|spot instances?`
  - `cost allocation|chargeback|showback`

#### `ml_ai_terms` — ML/AI engineering

- **Type:** regex patterns
- **Patterns/tokens:**
  - `\bmlops?\b|\bai\b|\bml\b|\bdl\b`
  - `model(?:s)?|training|inference|serv(?:e|ing)`
  - `fine[- ]?tune|prompt(?:ing)?|guardrails?`
  - `\bllm\b|embeddings?|vector (?:db|store)`
  - `\brag\b|retrieval[- ]?augmented`
  - `transformers?|pytorch|tensorflow|onnx`
  - `latency budget|token limit`


### General language markers

#### `negations` — Negations

- **Type:** token set
- **Patterns/tokens:**
  - `neither`
  - `never`
  - `nor`
  - `not`
  - `n't`
  - `without`

#### `comparatives` — Comparatives

- **Type:** token set
- **Patterns/tokens:**
  - `better`
  - `fewer`
  - `greater`
  - `higher`
  - `larger`
  - `less`
  - `lower`
  - `more`
  - `older`
  - `smaller`
  - `worse`
  - `younger`

#### `superlatives` — Superlatives

- **Type:** token set
- **Patterns/tokens:**
  - `best`
  - `highest`
  - `largest`
  - `least`
  - `lowest`
  - `most`
  - `oldest`
  - `smallest`
  - `worst`
  - `youngest`

#### `modals` — Modal verbs

- **Type:** token set
- **Patterns/tokens:**
  - `can`
  - `could`
  - `may`
  - `might`
  - `must`
  - `shall`
  - `should`
  - `will`
  - `would`

#### `hedges` — Hedges

- **Type:** token set
- **Patterns/tokens:**
  - `apparently`
  - `appears`
  - `approximately`
  - `arguably`
  - `around`
  - `likely`
  - `maybe`
  - `perhaps`
  - `roughly`
  - `seems`
  - `somewhat`
  - `suggests`
  - `unlikely`

#### `intensifiers` — Intensifiers

- **Type:** token set
- **Patterns/tokens:**
  - `clearly`
  - `especially`
  - `extremely`
  - `highly`
  - `incredibly`
  - `obviously`
  - `particularly`
  - `remarkably`
  - `significantly`
  - `strongly`
  - `undeniably`
  - `very`

#### `emotion_words` — Emotion words

- **Type:** token set
- **Patterns/tokens:**
  - `afraid`
  - `angry`
  - `anxious`
  - `concerned`
  - `confident`
  - `disappointed`
  - `fear`
  - `frustrated`
  - `happy`
  - `pleased`
  - `proud`
  - `sad`
  - `upset`
  - `worried`

#### `risk_liability` — Risk & liability language

- **Type:** token set
- **Patterns/tokens:**
  - `concern`
  - `hazard`
  - `liability`
  - `risk`
  - `risky`
  - `safety`
  - `unsafe`

#### `performance_judgment` — Judgments about competence/capability

- **Type:** token set
- **Patterns/tokens:**
  - `capability`
  - `capable`
  - `competence`
  - `competent`
  - `experienced`
  - `inexperienced`
  - `overqualified`
  - `qualified`
  - `skillful`
  - `talented`
  - `underqualified`
  - `unqualified`
