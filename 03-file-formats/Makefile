build:
	./install_venv.sh && ./get_data.sh

convert:
	sbatch convert.sh "squashfs" && sbatch convert.sh "lmdb" && sbatch convert.sh "hdf5"

bench-seq:
	sbatch run-scripts/simple-benchmarks/run-comp-tiny-seq.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-tiny-seq.sh "lmdb" && sbatch run-scripts/simple-benchmarks/run-comp-tiny-seq.sh "hdf5"

bench-par:
	sbatch run-scripts/simple-benchmarks/run-comp-tiny-par.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-tiny-par.sh "lmdb" && sbatch run-scripts/simple-benchmarks/run-comp-tiny-par.sh "hdf5"

bench-large:
	sbatch run-scripts/simple-benchmarks/run-comp-large.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-large.sh "lmdb"

bench-full:
	sbatch run-scripts/simple-benchmarks/run-comp-large-full.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-large-full.sh "lmdb"

post-seq:
	python3 run-scripts/simple-benchmarks/post-process.py -d 'tiny-seq'

post-par:
	python3 run-scripts/simple-benchmarks/post-process.py -d 'tiny-par'

post-large:
	python3 run-scripts/simple-benchmarks/post-process.py -d 'large'

post-large-full:
	python3 run-scripts/simple-benchmarks/post-process.py -d 'large-full'

bench-vit:
	sbatch run-scripts/training-benchmarks/run-comp-vision-transformer.sh "squashfs" && sbatch run-scripts/training-benchmarks/run-comp-vision-transformer.sh "lmdb" && sbatch run-scripts/training-benchmarks/run-comp-vision-transformer.sh "hdf5"