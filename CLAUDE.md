# CLAUDE.md

이 파일은 Claude Code (claude.ai/code) 가 이 저장소에서 작업할 때 참고하는 가이드.

## 프로젝트 목적

**한국투자증권(한투) 해외선물옵션** 거래 가능 universe (CME / EUREX / ICE / HKEX / SGX) 를
**확률분포로 다루는** 분석/백테스트 플레이그라운드.

자매 프로젝트:
- `c:\Projects\Quant\ProbabilityDistribution` — 암호화폐 (현물 + funding 차익)
- `c:\Projects\Quant\EquityDistribution` — 글로벌 주식/ETF (한투 해외주식 universe)

세 프로젝트 모두 **같은 방법론** (CVaR + GMM + HMM + walk-forward) 을 다른 자산군에 적용.

## 핵심 가설

- **LLM 안 씀** — 전부 수치 모델 (CVaR, GED, GMM, HMM, 분위수)
- **선물은 주식과 다른 동학** — trend-following이 mean-reversion보다 잘 작동 (학술 정설)
- **연속 컨트랙트 (continuous contract) 사용** — yfinance `=F` suffix가 자동 롤오버 처리
- **레버리지 1x로 운용** — 선물 본질이 5~20x 레버리지지만 백테스트는 notional sizing (현물처럼)
- **Walk-forward만 신뢰** — 자매 프로젝트 교훈 (single split selection bias)

## 데이터 출처

- **Yahoo Finance** (`yfinance`) — `CL=F`, `GC=F`, `ES=F` 등 연속 컨트랙트 (front-month auto-roll)
- **FRED** (`pandas_datareader.fred`) — 매크로 (yield curve, FFR, VIX, DXY)
- **Stooq** — yfinance 실패 시 폴백

전부 무료. API 키 불필요. 한투 API는 **실거래 시점**에만 사용 (백테스트 단계 X).

## 한투 거래 가능 universe

### 거래소 / 카테고리

| 카테고리 | 주요 상품 | yfinance 심볼 |
|---------|---------|--------------|
| **지수선물** | E-mini S&P 500, E-mini Nasdaq, E-mini Dow, E-mini Russell, Nikkei225, Hang Seng, DAX, EuroStoxx50 | `ES=F`, `NQ=F`, `YM=F`, `RTY=F`, `NIY=F`, `^HSI`, `^GDAXI`, `^STOXX50E` |
| **통화선물** | EUR, JPY, GBP, AUD, CAD, CHF, USD Index | `6E=F`, `6J=F`, `6B=F`, `6A=F`, `6C=F`, `6S=F`, `DX=F` |
| **금리선물** | US 2/5/10/30Y Treasury, German Bund | `ZT=F`, `ZF=F`, `ZN=F`, `ZB=F` |
| **에너지선물** | WTI Crude, Brent, Natural Gas, Heating Oil, Gasoline | `CL=F`, `BZ=F`, `NG=F`, `HO=F`, `RB=F` |
| **금속선물** | Gold, Silver, Copper (HG), Platinum, Palladium | `GC=F`, `SI=F`, `HG=F`, `PL=F`, `PA=F` |
| **농산물선물** | Corn, Wheat, Soybean, Coffee, Sugar, Cotton | `ZC=F`, `ZW=F`, `ZS=F`, `KC=F`, `SB=F`, `CT=F` |

총 ~30~40개 주요 선물.

### Mini / Micro 선물 (소액증거금 거래용)

| 풀 사이즈 | Mini (1/2 ~ 1/5) | Micro (1/10) |
|----------|-----------------|--------------|
| S&P 500 | E-mini ES=F | Micro MES=F |
| Nasdaq | E-mini NQ=F | Micro MNQ=F |
| Gold | GC=F | Micro MGC=F |
| Crude Oil | CL=F | Micro MCL=F |

**Retail 권장**: Micro 선물 (증거금 1/10) 로 시작 → $10k 계좌에서 운용 가능.

### 옵션 (별도 framework 필요)

- 지수옵션: ES 옵션, DAX 옵션, ES50 옵션
- 금리옵션: US 10Y 옵션
- 에너지옵션: CL 옵션, NG 옵션
- 금속옵션: GC 옵션
- 농산물옵션: ZC, ZS, ZW 옵션

옵션은 **현재 프로젝트 범위 외**. 별도 IV surface modeling 필요.

## 폴더 구조

```
FuturesDistribution/
├── src/                    # 코어 (data_loader, futures_universe, config, walk_forward)
├── iters/                  # iter*.py 백테스트 스크립트
├── tools/                  # 분석/유틸 (analyze, fetch_universe, summarize_iters)
├── results/                # 백테스트 산출물 JSON (commit)
├── logs/                   # 실행 로그 *.log (commit, 재현성)
├── dashboard/              # 챔피언 대시보드 HTML
├── archive/                # 사용 안 하는 옛 파일
├── data/                   # 캐시 (gitignored)
├── CLAUDE.md / README.md / RESULTS.md / requirements.txt
```

**중요**: `iters/` 와 `tools/` 의 .py 파일들은 `from src import ...` 사용 → **항상 root 에서 실행**.

## 주요 명령

Windows 콘솔에서 한글 출력. **항상** `PYTHONIOENCODING=utf-8` 설정.

```bash
# 1) Universe 다운로드 (parquet 캐시)
PYTHONIOENCODING=utf-8 python tools/fetch_universe.py

# 2) 단일 자산 분포 리포트
PYTHONIOENCODING=utf-8 python tools/analyze.py CL=F
PYTHONIOENCODING=utf-8 python tools/analyze.py ES=F

# 3) iter*.py 백테스트
PYTHONIOENCODING=utf-8 python iters/iter01_baseline.py
```

## 아키텍처

### 데이터 흐름

```
yfinance(=F suffix) ── load_ticker() ──┐
FRED fetch_fred() ────────────────────┤
                                      ▼
                            src/features.py
                            build_daily_features()
                                      ▼
                            Bot.decide() / signal computation
                                      ▼
                            backtester (fee 5bps + slippage 5bps)
                                      ▼
                            walk_forward.py (rolling train/test)
```

### 비용 모델 (선물 vs 주식 차이)

| 항목 | 주식 (EquityDist) | 선물 (이 프로젝트) |
|------|-------------------|-----------------|
| Commission | 1bp (US 0커미션) | **5bps** (한투 선물 수수료 + 거래소 비용) |
| Slippage | 2bps | **5bps** (호가 1tick) |
| Spread | 0.5bp | **2~3bps** (e-mini/micro) |
| Roll cost | 없음 (현물) | **3~5bps/quarter** (롤오버) |
| **합산** | **3~4bps** | **15~25bps** (raw) ~ **30~50bps** (보수적) |

**규칙**: 선물 백테스트는 `fee_per_change` 기본 50bps, stress test는 **75/100/150bps** 까지.

### 봇 계층 (계획)

```
src/strategies.py
    BaseBot
    ├── BuyAndHoldBot                # 단순 long 비교군
    ├── MomentumBot                  # 트렌드 (선물 정통)
    ├── CarryBot                     # 통화/채권/원자재 carry
    ├── TermStructureBot             # contango/backwardation 활용
    ├── CVaRRiskParityBot            # CVaR fat-tail 페널티
    ├── HMMRegimeBot                 # 4-year regime detection
    └── R70AdaptedBot                # EquityDist R70 logic 이식
```

### 선물 특화 시그널 (주식 X)

1. **Term structure**: 만기 다른 컨트랙트 가격 차이로 contango/backwardation 측정
2. **Open Interest**: CFTC COT 리포트 (commercial vs speculator 포지션)
3. **Carry**: 만기 기간 가격 변화율 (FX/원자재 정통 시그널)
4. **Inventory data**: 원유 EIA, 가스 Storage Report (energy 한정)

**현재 단계**: 우선 EquityDist R70 logic 이식 (composite + CVaR + HMM) → baseline 확인 → 선물 특화 시그널 추가.

## 자산별 특성 (가설)

- **ES/NQ/YM**: 지수선물 = drift 강함 + 트렌드. 주식과 비슷.
- **CL/NG**: 에너지 = 변동성 폭발 + 계절성 + 지정학.
- **GC/SI**: 금/은 = 안전자산 / 인플레 헤지. trend 약함, regime-dependent.
- **6E/6J**: 통화 = mean-reverting + carry trade. 금리차 중요.
- **ZB/ZN**: 미국채 = 매크로 정책 driven. mean-reverting on yield level.
- **ZC/ZW/ZS**: 농산물 = 계절성 + 날씨. 비효율 시장 (alpha 가능).

## Critical conventions

- **항상 `PYTHONIOENCODING=utf-8`** — Windows 한글 print
- **레버리지 1x notional** — 백테스트는 현물처럼 (계약수 ÷ 자산 × 가격 ≤ 자본)
- **선물 데이터 결측 주의** — `=F` 연속 컨트랙트는 갭 적지만, micro 선물 (`MES=F`, `MGC=F`) 데이터는 2019~ 만 있음
- **24시간 거래** — 시그널은 종가 (오후 5시 NY) 기준이지만 주문은 24h 가능
- **만기 직전 reduce position** — FND (First Notice Day) 전일까지 청산 강제 (한투 정책)
- **Selection bias 차단**: walk-forward 만 신뢰
- **자매 프로젝트 교훈**: ML 회귀/HMM/QRF 모두 walk-forward 음수 (`fitting < raw stat`)

## Memory / iteration history

새 iter 결과는 `RESULTS.md` (생성 시) 에 한 줄 추가. iter*.json + iter*.log 은 `results/` / `logs/` 에 commit.

iter 번호는 자매 프로젝트와 충돌하지 않게 `iter01_*` 부터 시작.

## 실거래 안전 규칙 (한투 해외선물 적용 시)

1. **Micro 선물만 사용** (시작 단계, $10k 계좌)
2. **포지션 1개당 자본 5% 이하** (margin call 방지)
3. **STOP 주문 항상 동시 입력** (한투 IF Done OCO 활용)
4. **추가증거금 발생 시 즉시 강제청산** (-15% drawdown 도달 시)
5. **롤오버 자동 알림 셋업** (FND 5영업일 전)
6. **24시간 모니터링 자동화** (직접 보지 말고 봇이 처리)
