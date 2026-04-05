#!/bin/bash
set -e

# Compilar tudo
echo "Compilando..."
make all

SIZES=(64 128 256 512 1024 2048 4096)
SMOOTHERS_CPU=(jacobi jacobi_amortecido gauss_seidel gauss_seidel_rb sor)
SMOOTHERS_CUDA=(jacobi jacobi_amortecido gauss_seidel_rb)
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

    # Rodar uma vez so, sem --csv
    local output
    output=$("$bin" "$n" "$smoother")

    # Extrair historico de residuos
    echo "iteracao,residuo" > "$resfile"
    echo "$output" | grep -E "^(iter|v-cycle) " \
        | sed -E 's/^(iter|v-cycle) ([0-9]+)  residuo = (.+)/\2,\3/' >> "$resfile"

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
for method in sg mg; do
    for plat in cpu cuda; do
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

for method in sg mg; do
    for plat in cpu cuda; do
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
