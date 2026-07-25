# Databricks MLflow — Dev-to-Prod Lifecycle Guides

Two practical, reference-style guides for taking machine learning from development to production on Databricks with MLflow. Written to be useful from **beginner to level-400** — read top-to-bottom to learn the whole lifecycle, or jump to any section as a standalone reference.

Both guides share the same backbone (**MLflow tracks, Unity Catalog governs, deploy code — not models**) and diverge where the lifecycles genuinely differ: **evaluation** and **serving**.

## Contents

| Guide | Use case | What it covers |
|---|---|---|
| [`mlflow-classic-ml-churn-lifecycle.md`](./mlflow-classic-ml-churn-lifecycle.md) | Customer churn (tabular classification) | Feature Engineering tables, MLflow autolog, `mlflow.evaluate`, UC Model Registry with `@champion`/`@challenger` aliases, batch + real-time serving, Lakehouse Monitoring |
| [`mlflow-genai-rag-lifecycle.md`](./mlflow-genai-rag-lifecycle.md) | RAG knowledge assistant (GenAI) | Vector Search, RAG chain as "model-from-code", MLflow tracing, LLM-as-judge evaluation, prompt registry, Agent serving, production trace monitoring |

Both guides include, per lifecycle stage: a **Why it matters** callout, the **key code**, and a **✅ Best practices** list — plus a one-page checklist, common pitfalls, and a glossary at the end.

## Architecture diagrams

Each guide has an end-to-end architecture diagram (also available standalone):

| Diagram | Vector (SVG) | High-res (PNG) |
|---|---|---|
| Classic ML — Churn | [`diagram-classic-ml-churn.svg`](./img/diagram-classic-ml-churn.svg) | [`diagram-classic-ml-churn@3x.png`](./img/diagram-classic-ml-churn@3x.png) |
| GenAI — RAG | [`diagram-genai-rag.svg`](./img/diagram-genai-rag.svg) | [`diagram-genai-rag@3x.png`](./img/diagram-genai-rag@3x.png) |

Use **SVG** for slides/print (sharp at any zoom); use the **@3x PNG** anywhere raster is needed.

## The one idea to take away

**Deploy code, not models.** Instead of copying a trained model binary between environments, you promote the reviewed *training pipeline / chain code* through **dev → staging → prod** (via Databricks Asset Bundles + CI/CD), and each environment produces its own model with full lineage. Promotion between versions is a governed **alias** move (`@champion`), gated on an automated quality check — an accuracy gate for classic ML, an LLM-judge score gate for GenAI.

## How to use these

1. Pick the guide matching your project (classic ML or GenAI).
2. Skim §0 (the mental model) and the one-page checklist at the end.
3. Follow the sections in order for a first project; return to individual sections as a reference later.
4. Adapt the code blocks — they're faithful, teaching-oriented patterns; verify the newest API signatures against current Databricks docs before running.

## Also available as PDF Docs

- **GenAI RAG :** [`Databricks MLflow — GenAI RAG Lifecycle (Knowledge Assistant).pdf`](./pdf/Databricks%20MLflow%20—%20GenAI%20RAG%20Lifecycle%20%28Knowledge%20Assistant%29.pdf)
- **Classic ML:** [`Databricks MLflow — Classic ML Lifecycle (Customer Churn).pdf`](./pdf/Databricks%20MLflow%20—%20Classic%20ML%20Lifecycle%20%28Customer%20Churn%29.pdf)
