# MATH-500 Eval 캘리브레이션 결과

**날짜**: 2026-06-24
**하드웨어**: 2× NVIDIA H200 (143GB)
**모델**: GSAI-ML/LLaDA-8B-Base + LoRA(PEFT) 어댑터
**데이터셋**: HuggingFaceH4/MATH-500 (test)
**체크포인트** (각 run dir의 `checkpoint-1896`):
- shortCoT: `checkpoints/AM-DeepSeek-R1-CoT-4k-4k/LLaDA-8B-Base/BS16_math_ff_4096_SFT_shortcot_default_ep8_20260618_053832`
- longCoT: `checkpoints/AM-DeepSeek-R1-CoT-4k-4k/LLaDA-8B-Base/BS16_math_ff_4096_SFT_longcot_default_ep8_20260618_033051`

**공통 설정**: block_length=32, max_context_length=4096, prompt_style=default, seed=42(모든 run 동일 subsample).
정확도 = 정답수/N (strict = `<answer>` 영역만 / fallback = `\boxed` 폴백 포함). 괄호값 = `\displaystyle` 스코어러 버그 수정 적용치.

> 참고: `max_context_length=4096` 때문에 gen=4096는 실제로 `(4096 - prompt_len)` ≈ 약 3872 토큰만 생성됨 (총 시퀀스를 4096으로 캡). 학습이 ff_4096이라 의도에 부합.

---

## 결과 표

### shortCoT (gen=1024, N=32, batch=16)
| run | steps | earlystop | newline_later | strict | fallback | 시간 |
|---|---|---|---|---|---|---|
| s_base | 512 | ✓ | ✓ | 15.6% | 15.6% | 14m |
| s_steps | 1024 | ✓ | ✓ | 15.6% | 15.6% | 30m |
| s_es | 512 | ✗ | ✓ | 15.6% | 15.6% | 14m |
| s_nl | 512 | ✓ | ✗ | 12.5% | 12.5% | 14m |

### longCoT (gen=4096, N=16, batch=8)
| run | gen | steps | earlystop | newline_later | strict | fallback | 시간 |
|---|---|---|---|---|---|---|---|
| l_base | 4096 | 2048 | ✓ | ✓ | 6.2% | 6.2% | 102m |
| **l_steps** (기준) | 4096 | 4096 | ✓ | ✓ | 31.2% (37.5%) | 37.5% (43.8%) | 210m |
| l_es | 4096 | 4096 | ✗ | ✓ | 31.2% (37.5%) | 37.5% (43.8%) | 201m |
| l_nl | 4096 | 4096 | ✓ | ✗ | 31.2% | 37.5% | 210m |
| l_gen3k | 3072 | 3072 | ✓ | ✓ | 18.8% | 31.2% | 128m |

---

## 질문별 결론

### Q1. steps = gen_len vs gen_len/2 (정확도 + 속도)
- **shortCoT: 절반으로 줄여도 무손실** (15.6% = 15.6%), 시간 30m→14m로 **2배 빠름**. → `steps=gen_len/2` 채택 가능.
- **longCoT: 절대 안 됨.** steps=2048이면 생성 붕괴(`</answer>` 무한반복, 깨진 루프) → 6.2%. steps=4096이면 37.5%. → `steps=gen_len` 필수.

### Q2. 최대 batch_size
peak memory가 batch에 선형 (H200 143GB 기준):
- longCoT gen4096: **≤16** (B=8 → 60GB, B=16 → ~110GB, B=32 → OOM)
- shortCoT gen1024: **≤32** (B=16 → 44GB, B=32 → ~135GB, B=64 → OOM)
- ⚠️ GPU가 compute-bound라 **batch를 키워도 총 wall-time은 거의 안 줄어듦** (per-forward 시간이 batch에 선형으로 증가). batch는 OOM 회피용 적당값이면 충분.

### Q3. earlystop True vs False (정확도 + 속도)
- **양쪽 모두 효과 없음** (정확도·시간 동일: short 14m≈14m/15.6%=15.6%, long 210m≈201m/동일).
- 이유: 이 체크포인트들이 `<|eot_id|>` stop 토큰을 안 내보내고, longCoT는 `</answer>` 뒤에 다시 `<think>`를 이어가 끝까지 생성함. → **earlystop은 이 모델들에선 무의미.**

### Q4. newline_later True vs False (정확도)
- **off가 약간 손해** (short 15.6→12.5%, long fb 37.5→31.2%). N이 작아 각 1개 차이라 노이즈 가능성 있으나 방향은 일관되게 on 유리. → **newline_later ON 유지 권장.**

### Q5. longCoT gen 4096 vs 3072 (둘 다 full steps)
- gen3072가 **더 나쁨** (18.8% vs 37.5% strict), 시간은 128m vs 210m로 ~40% 빠름.
- 일부 문제는 3072 토큰으로 CoT가 안 끝남. N=16이라 노이즈는 있으나 steps 결과와 같은 방향(긴 예산 필요). → **longCoT는 gen=4096 유지 권장.**

---

## 부가 발견

- ⚠️ **스코어러 `\displaystyle` 미처리 버그**: longCoT 출력에 `\displaystyle\frac{...}` 형태가 자주 나오는데 정답과 매칭 실패 → 16개 중 1개 과소평가 (l_steps/l_es 31.2→37.5% strict). 수정 위치: `eval/math_scorer.py` 정규화 또는 `eval/eval.py`의 `_preprocess_math_answer`에 `\displaystyle` 제거 추가.
- 🔴 **본 실험 시간 경고**: longCoT full-steps는 **예제당 ~13분** (210m/16). MATH-500 전체(500개)면 단일 GPU ~109h, **2 GPU ~55h**. shortCoT는 예제당 ~26s → 500개 2 GPU ~1.8h. longCoT 전체 평가는 시간 예산을 미리 확보하거나 subsample/분할 권장.

---

## 권장 최종 config

| 항목 | shortCoT | longCoT |
|---|---|---|
| gen_length | 1024 | 4096 |
| diffusion_steps | **512** (gen/2) | **4096** (=gen) |
| block_length | 32 | 32 |
| batch_size | 16~32 | 8~16 |
| earlystop | 무관 (off) | 무관 (off) |
| newline_later | **on** | **on** |
| max_context_length | 4096 | 4096 |

---

## 측정 메모 (재현용)

per-forward 시간 (forward 1회 = prompt+gen 전체 시퀀스 재계산, KV 캐시 없음):
- longCoT gen4096: 1.58s @ B=8, 3.19s @ B=16 (≈0.198 s/예제·forward)
- shortCoT gen1024: 0.889s @ B=16 (≈0.056 s/예제·forward)

총 forward 횟수 ≈ steps (`steps_per_block = steps // num_blocks`, `num_blocks = gen_len/block`).
총시간 ≈ N × steps × (예제당 forward 시간) — batch와 거의 무관.
