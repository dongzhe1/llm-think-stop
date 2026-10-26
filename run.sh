set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$ROOT/scripts"
PY="${PYTHON:-python3}"
SPLITS="${SPLITS:-50}"
RESULT_SETS="analysis analysis_natref extra seeds train label fine"

run() { echo "--- $*"; if [ -z "${DRY:-}" ]; then "$PY" "$@"; fi; }
abspath() { mkdir -p "$1" && (cd "$1" && pwd); }

analyze() {
    local probes="$1" out="$2" a="$2/analyses" p0="$1/seed0" seeds="$1/seed0" s
    for s in 1 2; do
        if [ -d "$probes/seed$s" ]; then seeds="$seeds,$probes/seed$s"; fi
    done
    cd "$out"
    run "$S/analyze_v2.py" --root "$p0" --out "$a/analysis" --splits "$SPLITS"
    run "$S/analyze_v2.py" --root "$p0" --out "$a/analysis_natref" --ref natural --accts line --splits "$SPLITS"
    run "$S/analyze_v2_extra.py" --root "$p0" --out "$a/extra" --splits "$SPLITS"
    run "$S/analyze_v2_extra.py" --root "$p0" --out "$a/extra" --natural_curves DeepSeek-R1-Distill-Qwen-7B \
        --splits "$SPLITS"
    run "$S/analyze_v2_extra.py" --root "$p0" --out "$a/train" --train_curve --splits "$SPLITS"
    run "$S/analyze_v2_seeds.py" --roots "$seeds" --out "$a/seeds" --splits "$SPLITS"
    run "$S/analyze_v2_extra.py" --root "$p0" --out "$a/label" --label_check --splits "$SPLITS"
    run "$S/analyze_v2_extra.py" --root "$p0" --out "$a/fine" --fine_static --splits "$SPLITS"
}

collect() {
    local out="$1" dest="$2" d f
    for d in $RESULT_SETS; do
        [ -d "$out/analyses/$d" ] || continue
        mkdir -p "$dest/$d"
        for f in "$out/analyses/$d"/*; do
            case "$f" in
                *.md) ;;
                *) cp "$f" "$dest/$d/" ;;
            esac
        done
    done
    echo "results -> $dest"
}

generate() {
    local task="$1" model="$2" seed="$3" mt=16384 tp=1
    case "$task" in
        gsm8k|aqua|bbh_free) mt=8192 ;;
        mmlu_pro) mt=12288 ;;
    esac
    if [ "$model" = Qwen3-32B ]; then tp="${TP32:-2}"; fi
    run "$S/reason_learnstop_v2.py" --model "$MODELS_DIR/$model" \
        --data "data/${task}_eval.jsonl,data/${task}_train.jsonl" --max_think "$mt" --tp "$tp" \
        --out_root "probes/seed$seed" --seed "$seed"
}

fine() {
    local model="$1" stems="$2" tp=1
    if [ "$model" = Qwen3-32B ]; then tp="${TP32:-2}"; fi
    run "$S/probe_fine_grid.py" --model "$MODELS_DIR/$model" --stems "$stems" --tp "$tp" --root probes/seed0
}

fake() {
    local d; d="$(abspath "${1:-$ROOT/fake}")"
    [ -e "$d/probes" ] || run "$S/make_fake_probes.py" "$d/probes"
    (analyze "$d/probes" "$d")
    collect "$d" "$d/results"
}

full() {
    : "${MODELS_DIR:?set MODELS_DIR}" "${OUT_DIR:?set OUT_DIR}"
    local out; out="$(abspath "$OUT_DIR")"
    MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"
    export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
    cd "$out"
    run "$S/prepare_reason_v2.py" --tasks gsm8k,math500,mmlu_pro,gpqa,aqua,bbh_free --out_dir data
    local task model seed
    for task in gsm8k math500 mmlu_pro; do
        for model in Qwen3-8B DeepSeek-R1-Distill-Qwen-7B DeepSeek-R1-Distill-Llama-8B AdaptThink-7B-delta0.05 \
                     Qwen3-32B; do
            generate "$task" "$model" 0
        done
    done
    for task in gpqa aqua bbh_free; do
        generate "$task" Qwen3-8B 0
    done
    generate gpqa Qwen3-32B 0
    for seed in 1 2; do
        generate gsm8k Qwen3-32B "$seed"
        generate mmlu_pro Qwen3-8B "$seed"
    done
    fine Qwen3-8B gsm8k_eval,math500_eval,mmlu_pro_eval,aqua_eval,gpqa_eval
    fine Qwen3-32B gsm8k_eval,math500_eval,mmlu_pro_eval,gpqa_eval
    fine DeepSeek-R1-Distill-Qwen-7B gsm8k_eval,math500_eval,mmlu_pro_eval
    fine DeepSeek-R1-Distill-Llama-8B gsm8k_eval,math500_eval,mmlu_pro_eval
    (analyze "$out/probes" "$out")
    collect "$out" "$out/bundle"
}

analyze_only() {
    : "${PROBES_DIR:?set PROBES_DIR}" "${OUT_DIR:?set OUT_DIR}"
    local out probes; out="$(abspath "$OUT_DIR")"; probes="$(cd "$PROBES_DIR" && pwd)"
    if [ -n "${DATA_DIR:-}" ]; then mkdir -p "$out/data" && cp "$DATA_DIR"/*.jsonl "$out/data/"; fi
    (analyze "$probes" "$out")
    collect "$out" "$out/bundle"
}

restore() {
    local list="$ROOT/LARGE_FILES.sha256" h f
    [ -f "$list" ] || { echo "nothing to restore"; return 0; }
    while read -r h f; do
        if [ -f "$ROOT/$f.part00" ]; then cat "$ROOT/$f".part* > "$ROOT/$f"; rm "$ROOT/$f".part*; fi
        if [ -f "$ROOT/$f.gz.part00" ]; then cat "$ROOT/$f".gz.part* > "$ROOT/$f.gz"; rm "$ROOT/$f".gz.part*; fi
        if [ -f "$ROOT/$f.gz" ] && [ ! -f "$ROOT/$f" ]; then gunzip "$ROOT/$f.gz"; fi
    done < "$list"
    if command -v sha256sum > /dev/null; then (cd "$ROOT" && sha256sum -c "$list")
    else (cd "$ROOT" && shasum -a 256 -c "$list"); fi
}

case "${1:-}" in
    fake) fake "${2:-}" ;;
    analyze) analyze_only ;;
    full) full ;;
    collect) : "${OUT_DIR:?set OUT_DIR}"; collect "$(abspath "$OUT_DIR")" "$(abspath "$OUT_DIR")/bundle" ;;
    restore) restore ;;
    *) echo "usage: bash run.sh fake [DIR] | analyze | full | collect | restore"; exit 2 ;;
esac
