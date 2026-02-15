# se_bias_sensitivity_analysis.py
import argparse, math, re, zipfile
from pathlib import Path

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import textwrap
from matplotlib import patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from wordcloud import WordCloud
import statsmodels.api as sm
from statsmodels.stats.rates import test_poisson_2indep



SE_LEXICON = {
	# ==== Core SE areas (expanded & de-duplicated where possible) ====
	"requirements_terms": [
		r"requirements?", r"acceptance criteria", r"user stor(?:y|ies)",
		r"spec(?:ification)?s?", r"feature request", r"change request",
		r"backlog", r"groom(?:ing)?", r"refinement", r"epic(s)?",
		r"definition of done|(?:\b|_)dod\b", r"\bmvp\b", r"prototype",
		r"non[- ]functional requirements|(?:\b|_)nfrs?\b",
		r"use cases?", r"spike(?:s)?", r"story mapping|story map",
	],
	"bug_failure_terms": [
		r"bugs?", r"defects?", r"crash(?:es)?", r"stack trace|traceback|backtrace",
		r"exception", r"null pointer|npe|nre", r"seg(?:mentation)? fault|segfault|sigsegv|sigabrt",
		r"regression(?:s)?", r"fail(?:ed|ure|ing)", r"outage", r"downtime", r"incident",
		r"oom|out[- ]of[- ]memory|oom[- ]?killed", r"memory leak", r"race condition",
		r"deadlock", r"hang|freeze", r"time[- ]?out", r"rollback",
		r"panic", r"assert(?:ion)? failure", r"core dump",
	],
	"testing_quality_terms": [
		r"tests?", r"testing", r"unit test(?:s)?", r"integration test(?:s)?",
		r"(?:end|e)[- ]?to[- ]?(?:end|e)", r"\be2e\b", r"coverage|code coverage",
		r"\bqa\b", r"flaky", r"repro(?:duce|duction)",
		r"mock(?:s|ing)?", r"fixture(?:s)?", r"test plan", r"linter(?:s)?",
		r"fuzz(?:er|ing)?|property[- ]?based", r"mutation testing",
		r"\bpytest\b|\bjunit\b|\btestng\b|\bjest\b|\bmocha\b|\bjasmine\b|\bvitest\b",
		r"selenium|cypress|playwright|testcontainers?",
		r"coverage.py|istanbul|nyc|cobertura",
	],
	"time_pressure_deadline": [
		r"deadlines?", r"\bsprint\b", r"iteration",
		r"blocker", r"priority", r"\bp0\b", r"\bp1\b", r"roadmap", r"milestone",
		r"time[- ]pressure", r"capacity", r"burn[- ]down", r"burn[- ]up",
		r"crunch|overtime", r"hard deadline", r"slip(?:page)?|overrun",
		r"stretch goal", r"cut scope",
	],
	"process_governance_terms": [
		r"code review", r"pull request", r"\bprs?\b", r"merge", r"rebase",
		r"cherry[- ]pick", r"branch", r"\bmain\b", r"fast[- ]forward",
		r"standards?", r"lint(?:er|ing)", r"style guide", r"compliance", r"policy",
		r"static analysis", r"sonarqube",
		r"\brfc\b", r"codeowners?", r"branch protection",
		r"trunk[- ]based", r"dora metrics?|deployment frequency|lead time for changes|change failure rate|\bmttr\b",
	],
	"estimation_uncertainty": [
		r"estimate(?:s|d|ing)?", r"story points?", r"t[- ]?shirt sizing",
		r"rough(?:ly)?", r"\bapproximately\b", r"\babout\b",
		r"confidence interval", r"\btbd\b", r"\btba\b", r"ballpark", r"guesstimate", r"at (?:most|least)",
	],
	"performance_scalability": [
		r"\bperformance\b", r"latenc(?:y|ies)", r"throughput", r"optim(?:ize|ised?|isation)?",
		r"benchmark(?:s)?", r"\bscale\b|scalab(?:le|ility)",
		r"\bmemory\b", r"\bcpu\b", r"\bgc\b", r"\bjit\b", r"profil(?:e|ing|er)",
		r"concurren(?:t|cy)", r"parallel(?:ism|ize)",
		r"\bp95\b|\bp99\b|tail latency", r"cold start", r"throttling|back[- ]?pressure|rate[- ]?limit(?:ing)?",
		r"heap dump|flame ?graph",
	],
	"security_privacy_terms": [
		r"\bsecurity\b", r"vulnerabilit(?:y|ies)", r"\bxss\b", r"sql injection|sqli",
		r"\bcve\b", r"encryption|cryptograph(?:y|ic)", r"secret(?:s)?", r"token", r"credential(?:s)?",
		r"\bpii\b", r"\bprivacy\b", r"\brbac\b", r"\boauth\b", r"\bjwt\b", r"\bmfa\b|\b2fa\b",
		r"csrf", r"ssrf", r"rce", r"sso", r"kms", r"\btls\b|m[- ]?tls", r"\bcors\b", r"\bcsp\b|clickjacking",
		r"\bowasp\b|\bcwe\b", r"sast|dast|iast|rasp", r"sbom|supply chain|code signing|sigstore|cosign",
		r"key rotation|least privilege|secrets? scan(?:ning)?|vault|hsm|fips",
	],
	"tooling_infra_terms": [
		r"c[iI][-_/ ]?c[dD]|cicd", r"pipeline(?:s)?", r"\bbuild\b", r"deploy(?:ment)?",
		r"container(?:s)?", r"docker", r"kubernetes|k8s", r"helm", r"terraform",
		r"runner(?:s)?", r"artifact(?:s)?", r"ansible", r"packer",
		r"bazel|cmake|ninja|gradle|maven|meson|\bmake(file)?\b",
		r"jenkins|github actions?|gitlab[-_ ]?ci|circleci|travis",
		r"argo[- ]?cd|spinnaker|fluxcd|skaffold",
		r"poetry|pipenv|tox|pre[- ]?commit|nix|conan|vcpkg",
		r"\bnpm\b|\byarn\b|\bpnpm\b|\bnvm\b|turborepo|lerna|\bnx\b|\bvite\b|webpack|rollup|esbuild",
	],
	"legacy_tech_debt": [
		r"legacy", r"technical debt|\btech debt\b", r"monolith(?:ic)?",
		r"refactor(?:ing|ed)?", r"rewrite", r"workaround", r"hack(?:y)?",
		r"deprecat(?:e|ed|ion)", r"spaghetti code|bit rot|stop[- ]?gap|shim",
	],
	"collaboration_conflict": [
		r"stakeholders?", r"\bproduct\b", r"\bpm\b", r"lead",
		r"disagree(?:s|ment)?", r"consensus", r"conflict",
		r"handoff|handover", r"alignment|misaligned", r"ownership",
		r"escalat(?:e|ion)s?", r"\braci\b|decision log|bikeshedding",
	],
	"refactoring_design_terms": [
		r"refactor(?:ing|ed)?", r"design pattern(?:s)?", r"\bsolid\b", r"\bdry\b",
		r"\bkiss\b", r"\byagni\b", r"architecture", r"module(?:s)?",
		r"interface(?:s)?", r"abstraction(?:s)?", r"cohesion", r"coupling",
		r"\bddd\b|hexagonal|onion architecture|clean architecture|cqrs|event sourcing",
	],
	"documentation_terms": [
		r"doc(?:s|umentation)?", r"readme", r"\badr\b", r"runbook|playbook",
		r"comment(?:s)?", r"diagram(?:s)?", r"postmortem", r"\brca\b",
		r"api reference|swagger|openapi", r"sphinx|javadoc|docstring",
		r"release notes|changelog|contributing\.md|faq|how[- ]?to|tutorial",
	],

	# ==== SRE/Incidents + Observability + Data/DB + FE/Mobile + Cloud/IAM + PM ====
	"incident_reliability": [
		r"\bon[- ]?call\b", r"\bsev[0-4]\b|\bsev[- ]?[0-4]\b", r"pagerduty", r"blameless",
		r"slo|sla|sli", r"error budget", r"resilien[ct]e?", r"failover", r"drill",
		r"\bmttr\b|\bmttd\b|\bmtbf\b", r"war room|major incident", r"chaos (?:test|engineering)|game day",
		r"circuit breaker", r"\bsre\b", r"toil",
	],
	"observability_terms": [
		r"metrics?", r"traces?", r"logs?", r"prometheus", r"grafana", r"datadog",
		r"opentelemetry|otel", r"alert(?:s|ing)?", r"dashboard",
		r"sentry|new relic|honeycomb|jaeger|zipkin|tempo",
		r"elasticsearch|kibana|loki", r"trace[- ]?id|span[- ]?id|correlation id",
	],
	"data_db_terms": [
		r"schema", r"migration(?:s)?|backfill", r"index(?:es|ing)?", r"replica(?:s|tion)?|read replica",
		r"shard(?:s|ing)?", r"vacuum|autovacuum|analyz(e|e)", r"explain plan|query plan|slow query",
		r"cache(?:s|ing)?", r"queue(?:s|ing)?", r"\bacid\b|transaction(?:s)?|isolation|mvcc",
		r"prepared statement|connection pool|orm",
		r"eventual consistency|strong consistency",
		# data/streaming/warehouse
		r"kafka|kinesis|pulsar|consumer group|partition|offset|compaction",
		r"spark|flink|beam|airflow|dbt",
		r"snowflake|redshift|bigquery",
		r"parquet|delta lake|iceberg|hudi",
		# common stores (word bounded)
		r"\bpostgres(?:ql)?\b|\bmysql\b|\bsqlite\b|\bredis\b|\bcassandra\b|\bmongo(?:db)?\b|\bcockroach\b",
	],
	"frontend_perf_accessibility": [
		r"bundle size|tree[- ]?shak(?:e|ing)|code[- ]?splitting|lazy[- ]?load(?:ing)?",
		r"\bcore web vitals?\b", r"\bttfb\b|\bcls\b|\blcp\b|\binp\b|\btti\b|\btbt\b",
		r"ssr|csr|ssg|isr|hydration|server[- ]?components?",
		r"lighthouse", r"\ba11y\b|accessibilit(y|ies)", r"aria[-: ]", r"screen reader",
		r"wcag|contrast ratio|tabindex|focus (?:trap|visible)",
		r"service worker|pwa|web worker",
	],
	"mobile_release": [
		r"\banr\b", r"crashlytics", r"testflight", r"play store", r"\bapk\b", r"\bipa\b",
		r"\baab\b|android app bundle", r"proguard|r8", r"minsdk|targetsdk",
		r"fastlane|code ?push|ota updates?", r"keystore|signing",
		r"deeplink|universal link",
	],
	"cloud_platforms": [
		r"\baws\b|\bgcp\b|\bazure\b", r"\biam\b", r"\bvpc\b",
		r"s3|gcs|blob storage", r"lambda|cloud[- ]functions|azure functions",
		r"iam role|assume[- ]?role|\bsts\b|oidc",
		r"ec2|gce|aks|eks|gke|cloud run|fargate|ecs",
		r"sqs|sns|pub/sub", r"rds|aurora|dynamodb|cosmos db|bigquery",
		r"step functions|dataflow|event hubs|service bus|api gateway",
		r"kms|key vault|cloud kms",
	],
	"pm_process_terms": [
		r"retro(?:spective)?", r"stand[- ]?up", r"demo", r"kanban", r"scrum", r"story map",
		r"\bokrs?\b|\bkpis?\b", r"\bmoscow\b", r"\brace\b|\bice\b scoring",
		r"pi planning|safe\b", r"\b3 amigos\b", r"gantt|critical path",
	],

	# ==== New: API/platform, FinOps/cost, AI/ML engineering (common in modern SE) ====
	"api_platform_terms": [
		r"\brest\b|graphq[lL]|g[- ]?rpc|websocket(?:s)?",
		r"openapi|swagger", r"pagination|rate[- ]?limit(?:ing)?|idempotent",
		r"version(?:ing)?|semver|conventional commits?",
	],
	"cost_finops": [
		r"\bcost(?:s)?\b|\bbudget(?:s|ary)?\b|\bspend\b|\bfinops\b",
		r"reserved instances?|savings plans?|spot instances?",
		r"cost allocation|chargeback|showback",
	],
	"ml_ai_terms": [
		r"\bmlops?\b|\bai\b|\bml\b|\bdl\b", r"model(?:s)?|training|inference|serv(?:e|ing)",
		r"fine[- ]?tune|prompt(?:ing)?|guardrails?", r"\bllm\b|embeddings?|vector (?:db|store)",
		r"\brag\b|retrieval[- ]?augmented", r"transformers?|pytorch|tensorflow|onnx",
		r"latency budget|token limit",
	],

	# ==== General markers — word-anchored automatically (kept as sets) ====
	"negations": {"not","never","neither","nor","without","n't"},
	"modals": {"can","could","may","might","must","should","would","will","shall"},
	"intensifiers": {"very","extremely","highly","significantly","strongly","particularly","especially","remarkably","incredibly","clearly","obviously","undeniably"},
	"hedges": {"maybe","perhaps","somewhat","likely","unlikely","approximately","roughly","around","suggests","appears","seems","apparently","arguably"},
	"comparatives": {"more","less","greater","smaller","higher","lower","older","younger","better","worse","fewer","larger"},
	"superlatives": {"most","least","best","worst","largest","smallest","highest","lowest","oldest","youngest"},
	"emotion_words": {"angry","happy","sad","fear","afraid","proud","disappointed","concerned","worried","anxious","confident","upset","frustrated","pleased"},
	"risk_liability": {"risk","risky","liability","hazard","unsafe","safety","concern"},
	"performance_judgment": {"capable","capability","competent","competence","qualified","unqualified","underqualified","overqualified","talented","skillful","experienced","inexperienced"},
}

def load_data(zip_path):
	"""Concatenate all CSVs having the required columns."""
	required = {"bias_name", "sensitive_to_bias", "prompt_with_bias"}
	frames = []
	with zipfile.ZipFile(zip_path) as z:
		for n in z.namelist():
			if not n.endswith(".csv"):
				continue
			with z.open(n) as f:
				df = pd.read_csv(f)
			if required.issubset(df.columns):
				frames.append(df[list(required)])
	if not frames:
		raise ValueError("No CSV with required columns found.")
	return pd.concat(frames, ignore_index=True).dropna(subset=["prompt_with_bias"])


def compile_lexicon(lex=None):
	lex = SE_LEXICON if lex is None else lex
	compiled = {}
	for k, v in lex.items():
		if isinstance(v, (set, frozenset)):
			# Treat as literal tokens; anchor to words; allow simple plural 's'
			alts = []
			for w in sorted(v):
				# keep n't as-is but still word-boundary around the t
				if w == "n't":
					alts.append(r"n['’]t")
				else:
					alts.append(rf"\b{re.escape(w)}\b")
			pat = r"(?:%s)" % "|".join(alts)
		else:
			# List of (possibly complex) regex patterns
			pat = r"(?:%s)" % "|".join(v)
		try:
			compiled[k] = re.compile(pat, re.IGNORECASE)
		except re.error as e:
			raise ValueError(f"Invalid pattern in '{k}': {e}\nPattern: {pat}")
	return compiled


def count_features(df, pats):
	df = df.copy()
	txt = df["prompt_with_bias"].astype(str).str.lower()
	for name, pat in pats.items():
		df[f"feat_{name}"] = txt.str.count(pat)
	df["tokens"] = txt.str.findall(r"\b\w+\b").str.len()
	return df


def _bh_fdr(pvals):
	# Benjamini–Hochberg (two-sided) returns adjusted p-values
	m = len(pvals)
	order = np.argsort(pvals)
	ranked = np.empty_like(order)
	ranked[order] = np.arange(1, m+1)
	adj = pvals * m / ranked
	# enforce monotonicity
	adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
	out = np.empty_like(adj)
	out[order] = np.clip(adj_sorted, 0, 1)
	return out

def poisson_glm_effect(df_doc, feat_col):
	y = df_doc[feat_col].astype(int)                 # counts per prompt
	X = sm.add_constant(df_doc["sensitive_to_bias"].astype(int))
	off = np.log(df_doc["tokens"].clip(lower=1))
	m = sm.GLM(y, X, family=sm.families.Poisson(), offset=off)
	res = m.fit(cov_type="HC1")  # or cov_type="cluster", cov_kwds={"groups": df_doc["prompt_id"]}
	b = res.params["sensitive_to_bias"]             # log rate ratio
	se = res.bse["sensitive_to_bias"]
	lo, hi = b - 1.96*se, b + 1.96*se
	z = b / se
	p = 2*sm.stats.normsf(abs(z))
	return b, se, z, p, lo, hi

def _poisson_glm_effect_per_prompt(df_bias, feat_col, prefer_quasi_if_overdisp=True):
	"""
	Returns: effect (log rate ratio), se, z, p, lo, hi, overdisp
	Effect = coefficient on 'sensitive_to_bias' from a Poisson GLM with log offset(tokens).
	"""
	# y: counts per prompt for this feature
	y = df_bias[feat_col].astype(int).to_numpy()

	# If the feature never appears, bail out early
	if y.sum() == 0 or df_bias["tokens"].sum() <= 0:
		return None

	X = pd.DataFrame({
		"const": 1.0,
		"sensitive": df_bias["sensitive_to_bias"].astype(int).to_numpy()
	})
	offset = np.log(df_bias["tokens"].clip(lower=1).astype(float).to_numpy())

	m = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)

	# ML fit to gauge overdispersion
	res_ml = m.fit()
	df_resid = max(res_ml.df_resid, 1)
	overdisp = float(res_ml.pearson_chi2 / df_resid)

	# Robust HC1 as sturdy default
	res_hc = m.fit(cov_type="HC1")
	b = float(res_hc.params["sensitive"])
	se = float(res_hc.bse["sensitive"])

	# If strongly overdispersed, use quasi-Poisson SEs (Pearson scaling)
	if prefer_quasi_if_overdisp and overdisp > 1.5:
		res_q = m.fit(scale="X2")  # same coef; inflated SEs
		se = float(res_q.bse["sensitive"])

	z = b / se if se > 0 else np.nan
	p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
	lo, hi = b - 1.96 * se, b + 1.96 * se
	return b, se, z, p, lo, hi, overdisp


def compute_effects(df, min_tokens=10, method="auto"):
	"""
	method:
	  - "poisson" : aggregated Wald log rate ratio (your original)
	  - "logit"   : aggregated smoothed log-odds diff (your original alternative)
	  - "glm"     : per-prompt Poisson GLM + offset, robust SEs
	  - "auto"    : use score/exact for tiny counts; else GLM (robust; quasi if overdispersed)
	"""
	feat_cols = [c for c in df.columns if c.startswith("feat_")]
	rows = []

	# Precompute group totals for rate displays
	agg = (
		df.groupby(["bias_name", "sensitive_to_bias"])[feat_cols + ["tokens"]]
		  .sum()
		  .reset_index()
	)

	for bias, df_bias in df.groupby("bias_name"):
		# Need both groups present with enough tokens
		g1 = agg[(agg["bias_name"] == bias) & (agg["sensitive_to_bias"] == True)]
		g0 = agg[(agg["bias_name"] == bias) & (agg["sensitive_to_bias"] == False)]
		if g1.empty or g0.empty:
			continue
		N1, N0 = int(g1["tokens"].iloc[0]), int(g0["tokens"].iloc[0])
		if N1 < min_tokens or N0 < min_tokens:
			continue

		for col in feat_cols:
			x1, x0 = int(g1[col].iloc[0]), int(g0[col].iloc[0])

			# Skip degenerate all-zero case early
			if x1 + x0 == 0:
				continue

			# Rates for display
			r1k1 = 1000.0 * x1 / max(N1, 1)
			r1k0 = 1000.0 * x0 / max(N0, 1)

			if method == "poisson":
				# Original: Wald LRR with 0.5 add
				lrr = math.log((x1 + 0.5) / N1) - math.log((x0 + 0.5) / N0)
				se = math.sqrt(1.0 / (x1 + 0.5) + 1.0 / (x0 + 0.5))
				z = lrr / se
				p = 2 * norm.sf(abs(z))
				lo = lrr - 1.96 * se
				hi = lrr + 1.96 * se
				effect = lrr

			elif method == "logit":
				# Original alternative: smoothed logit
				p1 = max(min((x1 + 1) / (N1 + 2), 1 - 1e-6), 1e-6)
				p0 = max(min((x0 + 1) / (N0 + 2), 1 - 1e-6), 1e-6)
				effect = math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0))
				var1 = 1.0 / max(N1 * p1 * (1 - p1), 1e-9)
				var0 = 1.0 / max(N0 * p0 * (1 - p0), 1e-9)
				se = math.sqrt(var1 + var0)
				z = effect / se
				p = 2 * norm.sf(abs(z))
				lo, hi = effect - 1.96 * se, effect + 1.96 * se

			elif method in ("glm", "auto"):
				# Tiny-count guardrails (switch to score/exact)
				small_total = (x1 + x0) < 20
				if method == "auto" and small_total:
					# Score test (great all-round), exact for ultra-tiny
					meth = "exact-cond" if (x1 + x0) <= 5 else "score"
					try:
						res_tp = test_poisson_2indep(x1, N1, x0, N0, method=meth)
						rr = float(res_tp.rate_ratio)
						lo_rr, hi_rr = res_tp.confint_ratio()
						effect = math.log(rr)
						ci_low, ci_high = math.log(lo_rr), math.log(hi_rr)
						# Wald-ish SE from CI half-width (only for display)
						se = (ci_high - ci_low) / (2 * 1.96)
						z = effect / se if se > 0 else np.nan
						p = float(res_tp.pvalue)
						lo, hi = ci_low, ci_high
					except Exception:
						# Safe fallback: Haldane LRR
						lrr = math.log((x1 + 0.5) / N1) - math.log((x0 + 0.5) / N0)
						se = math.sqrt(1.0 / (x1 + 0.5) + 1.0 / (x0 + 0.5))
						z = lrr / se
						p = 2 * norm.sf(abs(z))
						lo, hi = lrr - 1.96 * se, lrr + 1.96 * se
						effect = lrr
				else:
					# Per-prompt GLM with offset
					out = _poisson_glm_effect_per_prompt(df_bias, col, prefer_quasi_if_overdisp=(method=="auto"))
					if out is None:
						continue
					b, se, z, p, lo, hi, overdisp = out
					effect = b

			else:
				raise ValueError(f"Unknown method '{method}'")

			rows.append(dict(
				bias_name=bias,
				feature=col.replace("feat_", ""),
				effect=effect,
				se=se, z=z, p_value=p, ci_low=lo, ci_high=hi,
				rate_sensitive_per_1k_tokens=r1k1,
				rate_not_sensitive_per_1k_tokens=r1k0,
				count_sensitive=x1, count_not_sensitive=x0,
				tokens_sensitive=N1, tokens_not_sensitive=N0,
			))

	res = pd.DataFrame(rows)
	if not res.empty:
		res["p_fdr"] = _bh_fdr(res["p_value"].fillna(1.0).to_numpy())
		res["significant"] = res["p_fdr"] < 0.05
		res["direction"] = np.where(res["effect"] > 0, "sensitive>not", "not>sensitive")
		res = res.sort_values(["bias_name", "p_fdr", "effect"], ascending=[True, True, False])
	return res

def plot_overview_heatmap(
	res, out,
	cluster_rows=True, cluster_cols=True,
	signif_style="star",          # "star" | "alpha" | "both"
	max_xticklabels=36,           # max number of x labels to show
	star_levels=(0.001, 0.01, 0.05),  # thresholds for ***, **, *
	annotate_values=False,        # keep False to reduce clutter
	# --- NEW ---
	rowwise_top_k=3,                          # how many to highlight per row
	rowwise_metric="effect",              # "abs_effect" | "effect" | "display"
	rowwise_require_significant=True,        # restrict to significant cells?
	rowwise_box_kw=None                       # dict for Rectangle styling
):

	def _wrap_labels(labels, width=32):
		return [textwrap.fill(str(lbl).replace("_", " ").replace(" ", "\n"), width=width) for lbl in labels]

	def _auto_fontsizes(nrows, ncols):
		xfs = 10 if ncols <= 22 else 9 if ncols <= 34 else 8 if ncols <= 48 else 7
		yfs = 11 if nrows <= 20 else 10 if nrows <= 35 else 9 if nrows <= 55 else 8
		tfs = 13 if max(nrows, ncols) <= 30 else 12
		return xfs, yfs-1, tfs

	d = res.copy()
	if d.empty:
		return

	piv = d.pivot(index="bias_name", columns="feature", values="effect").fillna(0.0)
	sig = d.pivot(index="bias_name", columns="feature", values="significant").reindex_like(piv).fillna(False)
	# p_fdr may not exist; handle gracefully
	pf = None
	if "p_fdr" in d.columns:
		pf = d.pivot(index="bias_name", columns="feature", values="p_fdr").reindex_like(piv)

	# Optional clustering
	mat = piv.values
	row_idx = np.arange(piv.shape[0])
	col_idx = np.arange(piv.shape[1])
	try:
		# if cluster_rows and piv.shape[0] > 2:
		# 	row_idx = cluster_order_from_matrix(mat, axis="rows")
		if cluster_cols and piv.shape[1] > 2:
			col_idx = cluster_order_from_matrix(mat, axis="columns")
	except Exception:
		# row_idx = np.arange(piv.shape[0])
		col_idx = np.arange(piv.shape[1])

	piv = piv.iloc[row_idx, :].iloc[:, col_idx]
	sig = sig.iloc[row_idx, :].iloc[:, col_idx]
	if pf is not None:
		pf = pf.iloc[row_idx, :].iloc[:, col_idx]

	# Keep general markers at end
	tail = ["negations","comparatives","superlatives","modals","hedges",
			"intensifiers","emotion_words","risk_liability","performance_judgment"]
	cols = list(piv.columns)
	head = [c for c in cols if c not in tail]
	tail_present = [c for c in tail if c in cols]
	piv = piv[head + tail_present]
	sig = sig[head + tail_present]
	if pf is not None:
		pf = pf[head + tail_present]

	data = piv.values
	if data.size == 0:
		return

	# Asymmetric scaling
	vmax = np.nanpercentile(data, 99)
	vmin = np.nanpercentile(data, 5)
	norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

	# Sizing
	nrows, ncols = data.shape
	fig_w = 2.0 + 0.325 * ncols
	fig_h = 1.2 + 0.4 * max(4, nrows)
	fig, ax = plt.subplots(figsize=(fig_w, fig_h))

	im = ax.imshow(data, aspect="auto", norm=norm, cmap="RdBu_r")

	# Significance styling
	alpha_arr = np.where(sig.values, 1.0, 0.35)
	if signif_style in ("alpha", "both"):
		im.set_alpha(alpha_arr)
	else:
		im.set_alpha(1.0)

	# Ticks & de-cluttered x-labels
	xfs, yfs, tfs = _auto_fontsizes(nrows, ncols)
	ax.set_yticks(np.arange(nrows))
	ax.set_yticklabels(_wrap_labels(piv.index, width=11), fontsize=yfs)

	ax.set_xticks(np.arange(ncols))
	show = set(np.linspace(0, ncols - 1, num=min(ncols, max_xticklabels), dtype=int).tolist())
	xlabels_full = _wrap_labels(piv.columns, width=13)
	xlabels = [lbl if j in show else "" for j, lbl in enumerate(xlabels_full)]
	ax.set_xticklabels(xlabels, rotation=90, ha="center", fontsize=xfs)

	# Subtle gridlines
	ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
	ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
	ax.grid(which="minor", color="k", linestyle="-", linewidth=0.3, alpha=0.12)
	ax.tick_params(which="minor", bottom=False, left=False)

	# Visual separator for tail group
	if len(tail_present) > 0 and len(head) > 0:
		split_at = len(head) - 0.5
		ax.axvline(split_at, color="k", lw=2.0, alpha=0.25)
		ax.axvspan(split_at, ncols - 0.5, color="k", alpha=0.04)

	# Title + colorbar
	# ax.set_title("SE feature impact on bias sensitivity (log rate ratio)", fontsize=tfs)
	cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.005, format="%.1f")
	cb.set_ticks([vmin, vmin/2, 0.0, vmax/2, vmax])
	for label in cb.ax.get_yticklabels():
		label.set_rotation(90)
		label.set_horizontalalignment("center")
	cb.set_label("Effect (log rate ratio)", fontsize=8)

	# ----- Row-wise top-k highlighting -----
	if rowwise_box_kw is None:
		rowwise_box_kw = dict(edgecolor="black", linewidth=1.8, fill=False, alpha=0.95)

	# Which cells are eligible?
	if rowwise_require_significant:
		if pf is not None:
			consider = (pf.values < star_levels[-1])  # FDR < last threshold (default .05)
		else:
			consider = sig.values
	else:
		consider = np.ones_like(data, dtype=bool)

	# How to rank "impact"?
	if rowwise_metric == "display":
		# match perceived intensity on the heatmap (accounts for global norm + alpha)
		normed = im.norm(data)
		center = im.norm(0.0)
		scores_all = np.abs(normed - center) * (alpha_arr if signif_style in ("alpha","both") else 1.0)
	elif rowwise_metric == "effect":
		scores_all = data.copy()              # largest positive effects
	else:
		scores_all = np.abs(data)             # default: largest magnitude

	for i in range(nrows):
		scores = scores_all[i, :].copy()
		scores[~consider[i, :]] = -np.inf
		k = min(rowwise_top_k, np.isfinite(scores).sum())
		if k <= 0:
			continue
		top_idx = np.argpartition(scores, -k)[-k:]
		for j in top_idx:
			ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, **rowwise_box_kw))

	# Significance stars (***, **, *, or single * fallback)
	if signif_style in ("star", "both"):
		star_fs = max(8, min(12, int(12 - 0.02 * max(nrows, ncols))))
		for i in range(nrows):
			for j in range(ncols):
				s = ""
				if pf is not None and np.isfinite(pf.iat[i, j]):
					p = pf.iat[i, j]
					if p < star_levels[0]:
						s = "***"
					elif p < star_levels[1]:
						s = "**"
					elif p < star_levels[2]:
						s = "*"
				else:
					if bool(sig.iat[i, j]):
						s = "*"
				if s:
					val = data[i, j]
					on_dark = abs(val) > 0.6 * vmax
					txt = ax.text(j, i, s, ha="center", va="center",
								  fontsize=star_fs, color=("white" if on_dark else "black"),
								  zorder=5, alpha=1.0)
					txt.set_path_effects([pe.withStroke(linewidth=2, foreground=("black" if on_dark else "white"), alpha=0.6)])

	# Optional numeric annotations
	if annotate_values and (nrows * ncols <= 700):
		ann_fs = max(7, min(9, int(11 - 0.02 * max(nrows, ncols))))
		for (i, j), val in np.ndenumerate(data):
			if np.isfinite(val):
				on_dark = abs(val) > 0.6 * vmax
				txt = ax.text(j, i, f"{val:.2f}", ha="center", va="center",
							  fontsize=ann_fs, color=("white" if on_dark else "black"),
							  alpha=(1.0 if sig.iat[i, j] else 0.6), zorder=4)
				txt.set_path_effects([pe.withStroke(linewidth=2, foreground=("black" if on_dark else "white"), alpha=0.6)])

	# Legend (if helpful)
	legend_handles = []
	if signif_style in ("alpha","both"):
		legend_handles += [
			Line2D([0],[0], marker="s", linestyle="none", markersize=10, markerfacecolor="gray", alpha=1.0, label="Significant"),
			Line2D([0],[0], marker="s", linestyle="none", markersize=10, markerfacecolor="gray", alpha=0.35, label="Not significant"),
		]
	if rowwise_top_k:
		legend_handles.append(Rectangle((0,0), 1, 1, **{**rowwise_box_kw, "label": f"Top-{rowwise_top_k} per row"}))
	if legend_handles:
		ax.legend(
			handles=legend_handles,
			loc="upper left",
			bbox_to_anchor=(.88, 1.1),
			borderaxespad=0,
			frameon=True,
			fontsize=max(8, xfs-1)
		)

	# Clean look
	for spine in ax.spines.values():
		spine.set_visible(False)

	plt.tight_layout()
	fig.savefig(out, dpi=240, bbox_inches="tight")
	plt.close(fig)



def plot_dotmap(res, bias, out, top_n=12):
	d = res[res["bias_name"] == bias].copy()
	if d.empty:
		return
	d["abs"] = d["effect"].abs()
	d = d.sort_values(["significant", "abs"], ascending=[False, False]).head(top_n).reset_index(drop=True)

	y = np.arange(len(d))
	x = d["effect"].to_numpy()
	xerr_low = (d["effect"] - d["ci_low"]).to_numpy()
	xerr_high = (d["ci_high"] - d["effect"]).to_numpy()
	sig_mask = d["significant"].to_numpy()

	fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(d) + 1))
	ax.axvline(0, linestyle="--", linewidth=1)

	# plot significant points (solid)
	if sig_mask.any():
		ax.errorbar(
			x[sig_mask], y[sig_mask],
			xerr=[xerr_low[sig_mask], xerr_high[sig_mask]],
			fmt="o", capsize=3, elinewidth=1.2, alpha=1.0,
		)

	# plot non-significant points (faded)
	if (~sig_mask).any():
		ax.errorbar(
			x[~sig_mask], y[~sig_mask],
			xerr=[xerr_low[~sig_mask], xerr_high[~sig_mask]],
			fmt="o", capsize=3, elinewidth=1.2, alpha=0.35,
		)

	# guideline lines to the origin, with matching alpha
	for yi, xi, sig in zip(y, x, sig_mask):
		ax.plot([0, xi], [yi, yi], linewidth=2, alpha=0.6 if sig else 0.21)

	# y tick labels with rates
	labels = (
		d["feature"] + "  (" +
		d["rate_sensitive_per_1k_tokens"].round(2).astype(str) +
		" vs " +
		d["rate_not_sensitive_per_1k_tokens"].round(2).astype(str) +
		" /1k)"
	).tolist()
	ax.set_yticks(y)
	ax.set_yticklabels(labels)

	ax.set_xlabel("Effect (log rate ratio: Sensitive vs Not)  [points=95% CI]")
	ax.set_title(f"{bias}: features most associated with bias sensitivity")
	plt.tight_layout()
	fig.savefig(out, dpi=220)
	plt.close(fig)

def _safe_corrcoef(M, rowvar=False):
	"""Correlation with NaN-safe fallback (replaces NaNs/const columns)."""
	# Remove all-NaN columns/rows by replacing with zeros (correlation will be 0)
	M = np.asarray(M, float)
	if M.ndim != 2: M = np.atleast_2d(M)
	# If a col/row is constant, corr is NaN; replace with 0 later.
	C = np.corrcoef(M, rowvar=rowvar)
	C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
	# clip numerical noise
	np.fill_diagonal(C, 1.0)
	return C

def _spectral_order_from_similarity(S):
	"""Return an index order using the Fiedler vector of graph Laplacian."""
	n = S.shape[0]
	if n <= 2:
		return np.arange(n)
	# ensure non-negative similarity
	S = np.maximum(S, 0.0)
	np.fill_diagonal(S, 0.0)
	d = S.sum(axis=1)
	if np.allclose(d, 0.0):
		return np.arange(n)
	L = np.diag(d) - S
	# eigen-decomposition
	vals, vecs = np.linalg.eigh(L)
	# second-smallest eigenvector (Fiedler)
	order = np.argsort(vals)
	if len(order) < 2:
		return np.arange(n)
	fiedler = vecs[:, order[1]]
	return np.argsort(fiedler)

def cluster_order_from_matrix(X, axis="columns"):
	"""
	X is a 2D matrix of effects (bias x feature). Returns an index order for the chosen axis.
	axis="columns" → order features; axis="rows" → order biases.
	"""
	if axis == "columns":
		# similarity between columns (features) via correlation across biases
		C = _safe_corrcoef(X, rowvar=False)
	else:
		# similarity between rows (biases) via correlation across features
		C = _safe_corrcoef(X, rowvar=True)
	# map correlation [-1,1] → similarity [0,1]
	S = 0.5 * (C + 1.0)
	return _spectral_order_from_similarity(S)

def compute_word_effects(df, min_tokens=10, method="auto", top_n=50):
	"""
	Word-level log rate ratios using the same inference upgrades.
	Returns columns: bias_name, word, effect, count_sensitive, count_not_sensitive, p_value, ci_low, ci_high
	"""
	rows = []
	for bias, g in df.groupby("bias_name"):
		txt1 = g[g["sensitive_to_bias"] == True]["prompt_with_bias"].astype(str).str.lower()
		txt0 = g[g["sensitive_to_bias"] == False]["prompt_with_bias"].astype(str).str.lower()

		w1 = pd.Series([w for lst in txt1.str.findall(r"\b\w+\b") for w in lst])
		w0 = pd.Series([w for lst in txt0.str.findall(r"\b\w+\b") for w in lst])

		N1, N0 = int(w1.size), int(w0.size)
		if N1 < min_tokens or N0 < min_tokens:
			continue

		counts1 = w1.value_counts()
		counts0 = w0.value_counts()
		all_words = set(counts1.index) | set(counts0.index)

		for w in all_words:
			x1 = int(counts1.get(w, 0))
			x0 = int(counts0.get(w, 0))
			if x1 + x0 < 3:
				continue

			try:
				meth = "exact-cond" if (x1 + x0) <= 5 else "score"
				res_tp = test_poisson_2indep(x1, N1, x0, N0, method=meth)
				rr = float(res_tp.rate_ratio)
				lo_rr, hi_rr = res_tp.confint_ratio()
				if rr <= 0 or not np.isfinite(rr):
					raise ValueError("bad rr")
				effect = math.log(rr)
				ci_low, ci_high = math.log(lo_rr), math.log(hi_rr)
				p = float(res_tp.pvalue)
			except Exception:
				# fallback to Haldane-smoothed LRR
				effect = math.log((x1 + 0.5) / N1) - math.log((x0 + 0.5) / N0)
				# no reliable p/CI under fallback
				p = np.nan
				ci_low = np.nan
				ci_high = np.nan

			rows.append(dict(
				bias_name=bias, word=w, effect=effect,
				count_sensitive=x1, count_not_sensitive=x0,
				p_value=p, ci_low=ci_low, ci_high=ci_high
			))

	return pd.DataFrame(rows)



def plot_wordclouds_tokens(res_words, out_dir, top_n=50):
	from wordcloud import WordCloud
	for bias, g in res_words.groupby("bias_name"):
		if g.empty:
			continue
		g["abs_effect"] = g["effect"].abs()
		top = g.sort_values("abs_effect", ascending=False).head(top_n)

		freqs = dict(zip(top["word"], top["abs_effect"]))

		wc = WordCloud(
			width=1000, height=600,
			background_color="white",
			colormap="RdBu_r",
		).generate_from_frequencies(freqs)

		out = out_dir / f"wordcloud_words_{bias.replace(' ', '_')}.pdf"
		wc.to_file(out)
		print(f"Saved token-level wordcloud: {out}")



if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--zip", type=Path, required=True, help="Path to to_analyze.zip")
	# p.add_argument("--max-per-group", type=int, default=None)
	p.add_argument("--method", 
		choices=["auto","poisson","glm","logit"], 
		default="auto",
		help="Effect metric: 'auto' = tiny-count exact/score + per-prompt Poisson GLM (robust/quasi); 'poisson' = aggregated Wald LRR; 'glm' = GLM+offset; 'logit' = smoothed log-odds"
	)
	p.add_argument("--top-n", type=int, default=12)
	args = p.parse_args()

	zip_path=args.zip
	max_per_group=None
	
	out_dir = Path("bias_se_analysis")
	out_dir.mkdir(exist_ok=True)

	df = load_data(zip_path)

	# Optional: stratified sampling for speed
	if max_per_group is not None and max_per_group > 0:
		df = (
			df.groupby(["bias_name", "sensitive_to_bias"], group_keys=False)
			.apply(lambda g: g.sample(min(len(g), max_per_group), random_state=7))
			.reset_index(drop=True)
		)

	pats = compile_lexicon()
	df = count_features(df, pats)
	res = compute_effects(df, method=args.method)
	csv_path = out_dir / "se_feature_differences_by_bias.csv"
	res.to_csv(csv_path, index=False)

	# res_words = compute_word_effects(df, method=args.method)
	# if not res_words.empty:
	# 	plot_wordclouds_tokens(res_words, out_dir, top_n=args.top_n)


	top = (res[res["significant"]]
		   .sort_values(["bias_name","p_fdr","effect"], ascending=[True, True, False])
		   .groupby("bias_name")
		   .head(args.top_n))
	top.to_csv(out_dir / "top_significant_by_bias.csv", index=False)

	if not res.empty:
		plot_overview_heatmap(res, out_dir / "overview_heatmap.pdf")
		for bias in sorted(res["bias_name"].unique()):
			plot_dotmap(res, bias, out_dir / f"dotmap_{bias.replace(' ', '_')}.pdf", top_n=args.top_n)

	print(f"Saved: {csv_path}")
	print("Also wrote: overview_heatmap.pdf and dotmap_<bias>.pdf (one per bias)")
