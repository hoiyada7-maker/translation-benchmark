# translation-benchmark

동일 원문(SRT 자막)을 여러 번역 엔진으로 각각 번역해 접미사만 다른 파일로 저장하고, 싱크 비교 뷰어로 나란히 놓고 품질을 직접 비교하는 프로젝트.

## 사용법

1. 원문 SRT를 준비한다.
2. `scripts/` 안의 번역 스크립트로 엔진별 결과를 생성한다. 모두 `translations/원본이름.<engine>.srt` 형식으로 저장하며, 원본과 동일한 블록 번호·타임스탬프를 유지한다.
   - `translate_ollama.py <model> <src.srt> <out.srt> [logfile] [workers]` — 로컬 Ollama LLM, 블록 단위 병렬 번역(정렬 보장).
   - `translate_simple.py <model> <src.srt> <out.srt> [logfile] [workers]` — 최소 프롬프트 버전(문맥 프롬프트를 그대로 따라 하지 못하는 번역특화 모델용).
   - `translate_nllb.py`, `translate_madlad.py` — transformers 기반 NMT, GPU 배치 처리.
   - `translate_argos.py` — argos-translate(오프라인 CPU baseline).
   - `assemble_gold.py <src.srt> <gold_ko.txt> <out.srt>` — 직접 번역한 정답지를 `<블록번호>\t<한국어>` 텍스트에서 SRT로 조립.
   - `normalize_srt.py <src.srt> <engine.srt> [out.srt]` — 엔진 출력이 원문 블록 수·정렬과 어긋났을 때 원문 기준으로 재정렬.
3. `viewer.html`을 브라우저로 연다.
4. 상단 「📂 SRT 파일들 열기」로 정답지(`*.gold.srt`)와 비교할 엔진 SRT를 함께 선택(드래그도 가능)한다.
   - 정답지는 왼쪽에 고정, 엔진은 최대 3개까지 선택해 비교한다.
   - 「원문(EN) 행 표시」 체크박스로 원문을 함께 볼 수 있다(원문은 뷰어에 내장돼 있어 별도 로드 불필요).
   - 셀 클릭으로 오역·누락 플래그(빨강 표시), 하단 채점표에 엔진별 점수·메모 입력 가능(자동 저장).

## 실측 — 엔진별 번역 소요 시간

동일 문서(`caption_2026-09-10_09-26-10.srt`, 611블록·약 47분 영어 연설, en→ko)를 각 엔진으로 전량 번역한 실측치. 정렬 보장을 위해 로컬 LLM은 1블록=1요청(병렬 6워커)으로 처리했다. 자세한 내용은 [`2026-08-30 083717 번역품질개선-모델비교.md`](2026-08-30%20083717%20번역품질개선-모델비교.md)의 §3-1 참고.

| 엔진 | 실제 모델 태그 | 아키텍처 / 파라미터 / 양자화 | 방식 / 장치 | 소요 시간 | 비고 |
|---|---|---|---|---|---|
| **NLLB-200** | `facebook/nllb-200-distilled-600M` | NLLB / 600M / fp32 | transformers, GPU, batch16 | **0.9분** | 최속. 진짜 배치(텐서) 처리 |
| **argos-translate** | argos en→ko 패키지(내부 OpenNMT 소형) | 통계/소형 NMT | CPU, 블록단위 | **1.1분** | baseline, 직역체 |
| **TranslateGemma:4b** | `translategemma:4b` | gemma3 / 4.3B / Q4_K_M | Ollama, GPU, 블록단위×6 | **4.9분** | 번역특화. 최소 프롬프트(전/후 문맥 없음)로 측정 |
| **gemma4** | `gemma4:latest` | gemma4 / 8.0B / Q4_K_M | Ollama, GPU, 블록단위×6, 전/후 문맥 프롬프트 | **7.0분** | |
| **gemma4-e4b** | `gemma4:e4b` | gemma4 / 8.0B / Q4_K_M | Ollama, GPU, 블록단위×6, 전/후 문맥 프롬프트 | **7.2분** | `gemma4:latest`와 동일 스펙 태그, 시간·번역문 모두 사실상 동일 |
| **EXAONE 3.5** | `exaone3.5:latest` | exaone / 7.8B / Q4_K_M | Ollama, GPU, 블록단위×6, 전/후 문맥 프롬프트 | **7.4분** | 한국어 특화 |
| **Llama3.1** | `llama3.1:latest` | llama / 8.0B / Q4_K_M | Ollama, GPU, 블록단위×6, 전/후 문맥 프롬프트 | **7.7분** | |
| **MADLAD-400** | `google/madlad400-3b-mt` | T5 계열 / 3B / fp16 | transformers, GPU, batch8 | **12.5분** | |
| **qwen3:8b** | `qwen3:8b` | qwen3 / 8.2B / Q4_K_M | Ollama, GPU, 블록단위×6, 전/후 문맥 프롬프트 | **19.4분** | 문맥 프롬프트에서 장문 생성 경향 |
| **Claude Opus** | opencodex `ocx-claude-opus-4-8` | `claude-opus-4-8` | 클라우드 서브에이전트 | **33.6분** | 네트워크 왕복 포함. 보너스 열 |
| **정답지(Gold)** | 없음(모델 아님) | 메인 에이전트(`claude-opus-4-8`)가 611블록 수기 번역 | 수기 번역 | — | opus 엔진 열과 모델 계열은 같지만 방식이 다름 |

관찰:
- 로컬 소형 NMT(NLLB·argos)가 압도적으로 빠르다(1분 내외). 품질 상한은 낮지만 속도·오프라인은 최고.
- 로컬 LLM(Ollama)은 5~19분대. 같은 8B급이라도 생성 길이·양자화에 따라 편차가 크다(gemma4 7분대 vs qwen3 19분).
- TranslateGemma는 전/후 문맥 프롬프트를 지시가 아니라 입력으로 착각해 그대로 출력해버리는 문제가 있어(186/611블록 오염) 최소 프롬프트로 재측정했다. 다른 Ollama LLM(문맥 있음)과 1:1 비교는 아니다.
- 클라우드 LLM(opus)은 33분으로 가장 느리다 — 블록당 네트워크 왕복이 지배적.
