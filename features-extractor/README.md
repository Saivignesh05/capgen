# features-extractor

Hindi/Tamil clips in → measured acoustic features + one tool description per clip.
How it works and what every number means: [EXPLANATION.md](EXPLANATION.md).

```
DIR/tool_descriptions.json                      ← copied from the latest run into the folder you point at
features-extractor/                             ← this project
features-extractor-results/<DIR name>/<run>/    ← one folder per run, never overwritten:
                                                  features/<clip>.json, tool_descriptions.json, rejected, run.json
```

## Setup (conda, once, on a machine with internet)

```bash
cd features-extractor
conda env create -f environment.yml          # env "fx", exact versions
conda activate fx
python -m src fetch-models                   # frozen models -> models/ (~3 GB, needs internet; re-run adds new ones)
pytest ../tests                              # model-free tests
FX_MODELS=1 pytest ../tests                  # + the real models
```

## Run

Always from inside `features-extractor/` (no install step: `python -m src` runs the `src` folder).

```bash
conda activate fx
cd features-extractor
python -m src run /data/hi                 # every .wav under /data/hi, as a new run
python -m src runs /data/hi                # list runs: name, status, counts
python -m src run /data/hi --resume 2026-11-08_23-20-05   # continue that run ('latest' works too)
```

Result: `/data/hi/tool_descriptions.json`

```json
[
  {"audio_path": "hi/clip_001.wav", "tool_description": "Hindi, 9.7 s. Recording: ..."}
]
```

Each run gets its own folder named by its start time, e.g.
`../features-extractor-results/hi/2026-11-08_23-20-05/`, with the per-clip features, the
rejected list and `run.json`. Nothing is deleted; remove old runs yourself when you want.
`--resume RUN` continues that run instead of starting a new one.

Copy the descriptions somewhere else later (run does this for DIR automatically):

```bash
python copy_descriptions.py ../features-extractor-results/hi/2026-11-08_23-20-05 /some/other/dir
```

Check the syllable counter once on a clip whose transcript you know (Ak2_sample17: ~47 syllables
in ~7 s of speech, about 6.7 per second):

```bash
python -m src clip ../devset/ta/clips/Ak2_sample17.wav | grep -m2 '"syllables'   # clip totals
```

If it is far off, set `syllables: {impl: envelope}` in `config.yaml` (about 15-20 % low, no model).

To get exact syllable counts, fillers and word times, point `transcript:` in `config.yaml` at your
ASR stage's output (a JSON map of audio path to text, or one `<clip>.txt` per clip).

## Madhava (GPU, no internet)

On the laptop: `conda pack -n fx -o fx.tar.gz`, then
`rsync -avP fx.tar.gz features-extractor tests <user>@<madhava>:~/`. On Madhava:

```bash
mkdir -p ~/envs/fx && tar -xzf ~/fx.tar.gz -C ~/envs/fx
source ~/envs/fx/bin/activate && conda-unpack
cd ~/features-extractor && mkdir -p logs
sbatch slurm/extract.sbatch /data/hi                      # set your GPU partition in the file first
python -m src merge /data/hi --run job<JOBID>    # after the jobs finish
sbatch slurm/extract.sbatch /data/hi job<JOBID>           # resume a job that hit its time limit
```
