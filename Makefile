LOGS ?= logs/matrix_20251021-180517
TASKS ?= tasks.jsonl
SHARDS ?= 4
LOGDIR ?= logs/matrix_fullrerun_$$(date +%Y%m%d-%H%M%S)
WORKERS ?= 8

.PHONY: build-tasks split-tasks rotate-logs dry-run submit-shards

build-tasks:
	python tools/build_tasks_from_logs.py --logs $(LOGS) --out $(TASKS)

split-tasks:
	python tools/split_tasks.py --tasks $(TASKS) --parts $(SHARDS) --out_prefix tasks_shard

rotate-logs:
	bash scripts/rotate_logs.sh logs --archive

dry-run:
	python scripts/run_tasks.py --tasks tasks_shard0.jsonl --logdir $(LOGDIR) --max_workers 2 --fresh --dry_run

submit-shards:
	@mkdir -p slurm_logs
	@i=0; \
	while [ $$i -lt $(SHARDS) ]; do \
	  sbatch --export=ALL, \
	    TASKS=tasks_shard$$i.jsonl, \
	    LOGDIR=$(LOGDIR)/shard$$i, \
	    WORKERS=$(WORKERS) \
	    slurm/run_matrix_shard.sbatch; \
	  i=$$((i+1)); \
	done
	@echo "Submitted $(SHARDS) shards to Slurm. Logroot = $(LOGDIR)"
