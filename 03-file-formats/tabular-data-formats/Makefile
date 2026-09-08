ACCOUNT ?= project_xxxxxxxxx
SIF := /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
MODULES := module purge && module use /appl/local/laifs/modules && module load lumi-aif-singularity-bindings


download:
	./download_taxi_data.sh 2024-01



venv:
	$(MODULES) && singularity exec $(SIF) bash -c "python3 -m venv --system-site-packages venv-tables && source venv-tables/bin/activate && pip install tables --quiet"

convert:
	$(MODULES) && srun --account=$(ACCOUNT) --partition=small --time=00:10:00 --cpus-per-task=4 \
	  singularity exec $(SIF) bash -c "source venv-tables/bin/activate && python3 convert_tabular_formats.py --input data-formats-tabular/raw/yellow_tripdata_2024-01.parquet"

bench-scan:
	$(MODULES) && srun --account=$(ACCOUNT) --partition=small --time=00:10:00 --cpus-per-task=4 \
	  singularity exec $(SIF) bash -c "source venv-tables/bin/activate && python3 bench_tabular_scan.py --datadir data-formats-tabular"

bench-dataloader:
	for fmt in parquet csv hdf5; do \
	  $(MODULES) && srun --account=$(ACCOUNT) --partition=small --time=00:10:00 --cpus-per-task=4 \
	    singularity exec $(SIF) bash -c "source venv-tables/bin/activate && python3 bench_tabular_dataloader.py -ff $$fmt -N 10000"; \
	done
