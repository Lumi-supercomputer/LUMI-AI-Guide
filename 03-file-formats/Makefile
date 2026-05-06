build:
	./install_venv.sh && ./get_data.sh

convert:
	sbatch convert.sh "squashfs" && sbatch convert.sh "lmdb" && sbatch convert.sh "hdf5"

clean:
	rm -rf venv-extension data-formats

bench-seq:
	sbatch run-scripts/simple-benchmarks/run-comp-seq.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-seq.sh "lmdb" && sbatch run-scripts/simple-benchmarks/run-comp-seq.sh "hdf5"

bench-par:
	sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh "lmdb" && sbatch run-scripts/simple-benchmarks/run-comp-tiny.sh "hdf5"

bench-large:
	sbatch run-scripts/simple-benchmarks/run-comp-large.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-large.sh "lmdb"

bench-full:
	sbatch run-scripts/simple-benchmarks/run-comp-large-full.sh "squashfs" && sbatch run-scripts/simple-benchmarks/run-comp-large-full.sh "lmdb"

post-seq:
	python run-scripts/simple-benchmarks/post-process.py -d 'seq'

post-par:
	python run-scripts/simple-benchmarks/post-process.py -d 'tiny'

post-large:
	python run-scripts/simple-benchmarks/post-process.py -d 'large'

post-large-full:
	python run-scripts/simple-benchmarks/post-process.py -d 'large-full'
