#!/bin/bash
set -e

# Compilar tudo
echo "Compilando..."
make all

SIZES=(16 32 64 128 256 512 1024 2048 4096)
SMOOTHERS_CPU=(jacobi_amortecido gauss_seidel gauss_seidel_rb sor)
SMOOTHERS_CUDA=(jacobi_amortecido gauss_seidel_rb)
TOL=1e-15
MAX_ITERS_SG=100000
MAX_ITERS_MG=100000
HEADER="metodo,plataforma,n,smoother,iteracoes,erro,residuo,tempo_ms"

# Binarios
SG_CPU="./single-grid/cpu/sg_cpu"
SG_CUDA="./single-grid/cuda/sg_cuda"
MG_CPU="./multigrid/cpu/mg_cpu"
MG_CUDA="./multigrid/cuda/mg_cuda"

RESULTS="results"
SUMMARY="$RESULTS/summary.csv"

rm -rf "$RESULTS"

run_bench() {
    local bin="$1"
    local n="$2"
    local smoother="$3"
    local method="$4"
    local plat="$5"
    local dir="$6"

    local resfile="$dir/${smoother}_residuos.csv"
    local outfile="$dir/${smoother}.csv"

    # Limite de iteracoes depende do metodo:
    local max_iters
    if [ "$method" = "sg" ]; then
        max_iters="$MAX_ITERS_SG"
    else
        max_iters="$MAX_ITERS_MG"
    fi

    # Rodar uma vez so, sem --csv
    local output
    output=$("$bin" "$n" "$smoother" "$TOL" "$max_iters")

    # Extrair historico de residuos (erro vazio quando o binario nao imprime)
    echo "iteracao,residuo,erro" > "$resfile"
    echo "$output" | grep -E "^(iter|v-cycle) " \
        | sed -E 's/^(iter|v-cycle) ([0-9]+)  residuo = ([^ ]+)( +erro = ([^ ]+))?$/\2,\3,\5/' >> "$resfile"

    # Extrair resumo final
    local iters residuo erro tempo
    iters=$(echo "$output" | grep -E "^(iter|v-cycle) " | tail -1 | grep -oE '[0-9]+' | head -1)
    residuo=$(echo "$output" | grep "residuo final:" | grep -oE '[0-9]+\.?[0-9eE.+-]+')
    erro=$(echo "$output" | grep "erro maximo:" | grep -oE '[0-9]+\.?[0-9eE.+-]+')
    tempo=$(echo "$output" | grep "tempo total:" | grep -oE '[0-9]+\.?[0-9eE.+-]+')

    echo "$HEADER" > "$outfile"
    echo "$method,$plat,$n,$smoother,$iters,$erro,$residuo,$tempo" >> "$outfile"
}

# Contar total de testes
total=0
for method in mg sg; do
    for plat in cuda cpu; do
        if [ "$plat" = "cpu" ]; then
            smoothers=("${SMOOTHERS_CPU[@]}")
        else
            smoothers=("${SMOOTHERS_CUDA[@]}")
        fi
        for n in "${SIZES[@]}"; do
            total=$((total + ${#smoothers[@]}))
        done
    done
done

current=0

for method in mg sg; do
    for plat in cuda cpu; do
        # Selecionar binario
        if [ "$method" = "sg" ] && [ "$plat" = "cpu" ]; then
            bin="$SG_CPU"
        elif [ "$method" = "sg" ] && [ "$plat" = "cuda" ]; then
            bin="$SG_CUDA"
        elif [ "$method" = "mg" ] && [ "$plat" = "cpu" ]; then
            bin="$MG_CPU"
        elif [ "$method" = "mg" ] && [ "$plat" = "cuda" ]; then
            bin="$MG_CUDA"
        fi

        if [ ! -f "$bin" ]; then
            echo "AVISO: $bin nao encontrado, pulando $method/$plat"
            continue
        fi

        # Selecionar smoothers
        if [ "$plat" = "cpu" ]; then
            smoothers=("${SMOOTHERS_CPU[@]}")
        else
            smoothers=("${SMOOTHERS_CUDA[@]}")
        fi

        for n in "${SIZES[@]}"; do
            dir="$RESULTS/$method/$plat/${n}x${n}"
            mkdir -p "$dir"

            for smoother in "${smoothers[@]}"; do
                current=$((current + 1))
                echo "[$current/$total] $method / $plat / ${n}x${n} / $smoother"

                run_bench "$bin" "$n" "$smoother" "$method" "$plat" "$dir"
            done
        done
    done
done

# Gerar summary.csv
echo "$HEADER" > "$SUMMARY"
find "$RESULTS" -name "*.csv" ! -name "summary.csv" ! -name "*_residuos.csv" -exec tail -n +2 {} \; >> "$SUMMARY"

echo ""
echo "Benchmark completo! Resultados em $RESULTS/"
echo "Resumo geral em $SUMMARY"
