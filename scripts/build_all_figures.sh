#!/usr/bin/env bash
# Пересобрать ВСЕ фигуры лекции 19 (двойственные методы) из исходных скриптов.
# Каждая фигура воспроизводима: скрипт -> files/<figure>.pdf. Запуск из корня репо или откуда угодно.
# Использование: bash scripts/build_all_figures.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"

# скрипт -> ожидаемый выходной файл(ы) в files/
declare -A FIGS=(
  [exp_svm_dual.py]="exp_svm_dual.pdf"
  [exp_svm_geometry.py]="exp_svm_geometry.pdf"
  [exp_waterfilling.py]="exp_waterfilling.pdf"
  [exp_conj_support_smoothness.py]="exp_conj_support_smoothness.pdf exp_conj_smoothness_strong.pdf"
  [exp_conj_duality.py]="exp_conj_duality.pdf"
  [exp_conj_biconjugate.py]="exp_conj_biconjugate.pdf"
  [exp_dual_rates.py]="exp1_dual_rates.pdf"
  [exp_dual_stepsize_sensitivity.py]="exp_dual_stepsize_sensitivity.pdf"
  [exp_dual_decomposition.py]="exp4_decomposition.pdf"
  [exp_market_tatonnement.py]="exp_market_tatonnement.pdf"
  [exp_alm_penalty_landscape.py]="exp_alm_penalty_landscape.pdf"
  [exp_alm_vs_dual.py]="exp_alm_vs_dual.pdf"
  [exp_alm_rho_tradeoff.py]="exp_alm_rho_tradeoff.pdf"
  [exp_admm_consensus.py]="exp_admm_consensus.pdf"
  [exp_admm_adaptive_rho.py]="exp_admm_adaptive_rho.pdf"
  [exp_admm_fused_lasso.py]="exp_admm_fused_lasso.pdf"
  [fig_decoupling.py]="decoupling.png"
  [exp_admm_intersection.py]="exp_admm_intersection.pdf"
  [exp_admm_tv_deblur.py]="exp_admm_tv_deblur.pdf"
  [exp_admm_rpca.py]="exp_admm_rpca.pdf"
  [exp_lasso_admm.py]="exp3_lasso_admm.pdf"
  [exp_methods_evolution.py]="exp_methods_evolution.pdf"
  [fig_convex_intersection.py]="convex_intersection.png"
)

ok=0; fail=0; failed=()
for s in "${!FIGS[@]}"; do
  if [ ! -f "scripts/$s" ]; then echo "SKIP (нет скрипта): $s"; continue; fi
  if $PY "scripts/$s" >/dev/null 2>&1; then
    missing=""
    for out in ${FIGS[$s]}; do [ -f "files/$out" ] || missing="$missing $out"; done
    if [ -z "$missing" ]; then echo "OK   $s -> ${FIGS[$s]}"; ok=$((ok+1));
    else echo "WARN $s: запустился, но нет:$missing"; fail=$((fail+1)); failed+=("$s"); fi
  else
    echo "FAIL $s (ошибка выполнения)"; fail=$((fail+1)); failed+=("$s")
  fi
done

echo "----"
echo "Готово: $ok OK, $fail проблем."
[ $fail -gt 0 ] && { echo "Проблемные: ${failed[*]}"; exit 1; }
echo "Все фигуры лекции 19 пересобраны. Теперь: cd lectures && quarto render 19.md --to beamer"
exit 0
