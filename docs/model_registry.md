# Model Registry and Dataset/Split Ledgers

FiberTypeQC keeps one machine-readable registry for active, tracked model artifacts. Exact
biological cohort inventories and split ledgers remain private and outside Git:

- [`manifests/model_registry.v1.yaml`](../manifests/model_registry.v1.yaml)
- [`examples/reference/dataset_split_ledger.example.yaml`](../examples/reference/dataset_split_ledger.example.yaml)
- [`examples/reference/dataset_evidence_inventory.example.yaml`](../examples/reference/dataset_evidence_inventory.example.yaml)

The registry avoids using “candidate” as a generic label. Each entry declares its task, required
observed markers, feature schema, status, artifact digest, development/evaluation scope,
comparator, evidence record, and limitations. Historical model files do not enter the registry
until they are needed for a named comparison or reproducibility target.

An `artifact: null` entry is intentional: it identifies an active internal policy or candidate
whose artifact/evidence bundle is private or not yet frozen. Such an entry is useful for clear
status reporting, but is not runnable from this repository and cannot support an independently
reproducible claim.

The synthetic examples document and test the private ledger schemas without publishing real cohort,
mouse, image, genotype, split, or validation metadata. In a private ledger, a group can have exactly
one role in a version: `development`, `evaluation`, `reserve`, or `excluded`. A mouse is the
preferred group where it has multiple related images; otherwise use a source-image group and record
that limitation privately.

Private ledger versions are immutable: never edit a frozen assignment in place. Create a new
version, retain the prior file, and record why an assignment changed. Before using a cohort for
model, gate, or threshold selection, assign every eligible group to a role in that private ledger.
Its checksum may be recorded in private experiment provenance, but neither the ledger nor its exact
group metadata belongs in the public repository.

The private inventory records dataset-specific artifact completeness, while the private split ledger
records evidence roles. These are separate contracts and neither is inferred from the other.

Label authority is recorded separately from split role:

- `manual_gold`: independently reviewed biological label.
- `reviewed_myosight`: expert-curated MyoSight label—curated, accepted, or corrected by an experienced reviewer; it may support supervised development but is not `manual_gold` unless independently designated as such.
- `myosight_derived`: unreviewed historical-workflow label; suitable only for declared agreement or exploratory work.
- `model_prediction`: unreviewed FiberTypeQC output; never a biological ground-truth label.
- `mixed`: a cohort contains more than one of the above; each selected row must still declare its own authority.

Repository checks validate the synthetic ledger examples and every registered public artifact
digest. Private experiment checks validate exact private ledgers in their controlled environment.
This separation tests the reproducibility mechanism without claiming that private validation data
are publicly rerunnable.
