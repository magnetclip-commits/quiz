#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# 날짜: MMDD (문자열) → 10진수 강제 변환(앞자리 0로 인한 8진수 해석 방지)
mmdd="$(date +%m%d)"      # 예: "0815"
mmdd10=$((10#$mmdd))      # 예: 815
year="$(date +%Y)"        # 예: 2026

arg2=""

# 02/15 ~ 03/31 => 10
if [[ $mmdd10 -ge 215 && $mmdd10 -le 331 ]]; then
  arg2="10"
# 08/15 ~ 09/30 => 20
elif [[ $mmdd10 -ge 815 && $mmdd10 -le 930 ]]; then
  arg2="20"
else
  # 해당 기간 아니면 조용히 종료
  exit 0
fi

log "START cls_mst migrate (year=${year}, arg2=${arg2})"

/usr/bin/docker exec hlta-api bash -c "
  cd /app && \
  python3 -m aita.oracle_to_postgres_cls_mst_migrate ${year} ${arg2}
"

log "DONE cls_mst migrate"

log "START cls_enr migrate (year=${year}, arg2=${arg2})"

/usr/bin/docker exec hlta-api bash -c "
  cd /app && \
  python3 -m aita.oracle_to_postgres_cls_enr_migrate ${year} ${arg2}
"

log "DONE cls_enr migrate"
log "ALL DONE"
