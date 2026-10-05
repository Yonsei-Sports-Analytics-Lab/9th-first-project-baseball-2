# GBM을 활용한 장타 허용 이후 구종 회피 요인 분석

## 1. 모델 설계

### 1.1 분석 목적

앞선 분석에서는 투수가 장타를 허용한 구종을 이후 같은 타자와의 시즌 내 재대결에서 평소보다 적게 사용하는 경향을 확인하였다.

본 분석에서는 한 단계 더 나아가 다음 질문을 확인하고자 하였다.

> **어떤 상황에서 투수는 장타를 허용한 구종을 더 강하게 회피하는가?**

이를 위해 장타 이벤트별 이후 구종 사용 변화율(`usage_change`)을 종속변수로 설정하고, **Gradient Boosting Machine(GBM)**을 이용하여 장타 당시의 결과, 타구 특성, 경기 상황 등이 이후 회피 강도와 어떠한 연관성을 가지는지 분석하였다.


### 1.2 종속변수

종속변수 `usage_change`는 다음과 같이 정의하였다.

**usage_change = post_usage - baseline_usage**

- `post_usage`: 장타 허용 이후 같은 시즌 내 동일 투수-타자 재대결에서 장타 허용 구종의 구사율
- `baseline_usage`: 해당 투수의 해당 시즌 해당 구종 평균 구사율

따라서,

- `usage_change < 0` : 시즌 평균보다 해당 구종을 적게 사용 → **회피**
- `usage_change > 0` : 시즌 평균보다 해당 구종을 많이 사용
- `usage_change`가 작을수록 장타 허용 구종에 대한 회피가 강한 것으로 해석하였다.


### 1.3 사용 모델

Histogram-based Gradient Boosting Regressor를 사용하였다.

GBM은 독립변수와 종속변수 사이의 관계를 선형으로 가정하지 않으므로, 점수차나 타구 특성과 같이 비선형적인 관계가 존재할 수 있는 변수들을 함께 분석하는 데 적합하다고 판단하였다.

모델 학습 후에는 **SHAP(SHapley Additive exPlanations)**을 이용하여 각 독립변수가 GBM의 예측에 미친 상대적 영향의 크기와 방향을 분석하였다.

SHAP 값의 방향은 다음과 같이 해석하였다.

- **SHAP < 0** → 예측 `usage_change` 감소 → 회피 강화 방향
- **SHAP > 0** → 예측 `usage_change` 증가 → 회피 약화 방향


### 1.4 모델 성능

총 **38,012개의 장타-재대결 이벤트**를 분석에 사용하였으며, 데이터를 80:20으로 분할하였다. (80=학습 데이터, 20=test 데이터)

| 구분 | 결과 |
| 전체 이벤트 | 38,012 |
| Train | 30,409 |
| Test | 7,603 |
| MAE | 0.1841 |
| RMSE | 0.2447 |
| R² | 0.0352 |

모델의 R²는 **0.0352**로 높지 않았다. 즉, 본 모델에 포함된 변수들만으로 개별 장타 이벤트 이후의 구종 사용 변화를 충분히 설명하기는 어려웠다.

따라서 본 분석에서는 GBM을 높은 정확도의 예측 모델로 활용하기보다, **SHAP을 통해 `usage_change`와 상대적으로 강한 예측적 연관성을 보이는 요인과 그 방향성을 탐색하는 데 중점을 두었다.**


---

## 2. 독립변수 및 SHAP Feature Importance

### 2.1 독립변수

모델에는 장타 발생 당시의 정보를 중심으로 다음 변수들을 포함하였다.

| 구분 | 변수 |
|---|---|
| 장타 결과 및 타구 특성 | Home Run 여부, Launch Speed, Launch Angle, Estimated wOBA |
| 장타 허용 구종 | FF, SI, SL, CH, FC, CU, ST, FS, KC 등 |
| 투타 Matchup | L-L, L-R, R-L, R-R |
| Count State | 0-0 ~ 3-2 |
| 경기 상황 | Score Difference, 주자 상황, Out State |
| 투구 특성 | Release Speed, Horizontal/Vertical Movement, Plate Location |


### 2.2 SHAP Feature Importance

각 변수의 상대적 중요도는 **Mean Absolute SHAP Value**를 이용하여 비교하였다.

![SHAP Feature Importance](./images/final_gbm_shap_importance_clean.png)

Mean Absolute SHAP Value가 클수록 해당 변수가 GBM의 `usage_change` 예측값을 변화시키는 평균적인 영향의 크기가 크다는 것을 의미한다.

단, 절댓값을 사용하므로 이 그래프 자체는 해당 변수가 회피를 강화하는지 또는 약화하는지에 대한 **방향을 나타내지 않는다.**

최종 모델에서 가장 높은 중요도를 보인 변수는 다음과 같았다.

| 순위 | 변수 | Mean \|SHAP\| |
|---:|---|---:|
| 1 | Home Run | 0.0177 |
| 2 | Score Difference | 0.0085 |
| 3 | Launch Speed | 0.0076 |
| 4 | SI | 0.0058 |
| 5 | Release Speed | 0.0057 |
| 6 | RHP-LHB Matchup | 0.0055 |
| 7 | Vertical Movement | 0.0052 |
| 8 | LHP-RHB Matchup | 0.0040 |
| 9 | Launch Angle | 0.0036 |
| 10 | Horizontal Movement | 0.0035 |
| 11 | Estimated wOBA | 0.0030 |

특히 **Home Run 여부가 가장 높은 중요도**를 보였으며, Score Difference와 Launch Speed가 그 뒤를 이었다.

반면 Count State, 주자 상황, Out State의 상대적 중요도는 비교적 낮게 나타났다.


### 2.3 SHAP Summary

![SHAP Summary](./images/final_gbm_shap_summary.png)

SHAP Summary Plot을 통해 각 변수의 상대적 중요도뿐만 아니라 변수 값에 따른 예측 방향을 함께 확인하였다.

이를 바탕으로 중요도가 높게 나타난 주요 변수의 영향을 개별적으로 분석하였다.


---

## 3. 주요 독립변수 세부 분석

### 3.1 Home Run

Home Run 여부는 최종 모델에서 **Mean |SHAP| = 0.0177**로 가장 높은 중요도를 보였다.

Home Run 여부의 SHAP 분포를 살펴보면, Home Run에 해당하는 이벤트는 주로 음의 SHAP 값을 나타냈다.

즉,

**Home Run 허용 → 예측 usage_change 감소 → 장타 허용 구종 회피 강화**

의 패턴이 나타났다.

따라서 같은 장타라 하더라도 2루타·3루타보다 **홈런을 허용한 경우 이후 같은 타자와의 재대결에서 해당 구종을 더 강하게 회피하는 방향**이 모델에서 포착되었다.


### 3.2 Score Difference

Score Difference는 **Mean |SHAP| = 0.0085**로 두 번째로 높은 중요도를 보였다.

![Score Difference SHAP](./images/gbm_score_diff_shap.png)

Score Difference와 SHAP 값 사이에서는 단순한 선형관계가 아닌 **비선형적 관계**가 나타났다.

접전에 가까운 상황에서는 상대적으로 음의 SHAP 값이 나타나 회피 강화 방향을 보였으며, 점수차가 크게 벌어진 상황에서는 양의 SHAP 값이 증가하는 패턴이 관찰되었다.

이는 장타의 결과나 타구 특성뿐만 아니라 **장타가 발생한 당시의 경기 맥락 역시 이후 구종 사용 변화와 연관되어 있음을 보여준다.**


### 3.3 Launch Speed

Launch Speed는 **Mean |SHAP| = 0.0076**으로 세 번째로 높은 중요도를 보였다.

![Launch Speed SHAP](./images/gbm_launch_speed_shap.png)

Launch Speed가 증가할수록 SHAP 값이 대체로 감소하는 패턴이 나타났다.

즉,

**Launch Speed 증가 → 예측 usage_change 감소 → 장타 허용 구종 회피 강화**

의 관계가 관찰되었다.

따라서 같은 장타를 허용하더라도 **더 높은 타구속도의 장타를 허용한 경우 이후 해당 구종을 더 강하게 회피하는 방향**이 나타났다.


### 3.4 장타 허용 구종

장타를 허용한 구종 역시 GBM의 예측에 일정한 영향을 미쳤다.

특히 `SI`는 **Mean |SHAP| = 0.0058**로 구종 변수 가운데 가장 높은 중요도를 보였다.

구종별 SHAP 값의 차이는 장타를 허용한 구종에 따라 이후 구종 사용 변화의 패턴이 동일하지 않을 가능성을 보여준다.

다만 구종별 중요도는 해당 구종을 사용하는 투수의 특성, 구종별 기본 구사율 및 레퍼토리 구성 등의 영향을 함께 반영할 수 있으므로, 특정 구종 자체의 인과효과로 해석하지 않았다.


### 3.5 투타 Matchup

투수와 타자의 손잡이를 각각 독립적으로 사용하기보다 다음 네 가지 투타 조합으로 구성하였다.

- LHP-LHB
- LHP-RHB
- RHP-LHB
- RHP-RHB

이 가운데 RHP-LHB와 LHP-RHB matchup이 비교적 높은 SHAP 중요도를 보였다.

이는 장타 이후의 구종 회피가 투수 또는 타자의 손잡이 하나만의 문제가 아니라 **투수-타자의 조합에 따라서도 달라질 수 있음**을 보여준다.

다만 matchup은 투수의 레퍼토리 및 구종별 platoon 특성과 연관될 수 있으므로 인과적으로 해석하지 않았다.


### 3.6 Count / 주자 / Out State

Count를 balls와 strikes로 각각 처리하지 않고 실제 투구 상황을 반영하는 `count_state`로 구성하였다.

이 가운데 `2-2`, `3-2` count가 다른 count에 비해 상대적으로 높은 중요도를 보였지만 전체 변수 중에서는 각각 17위, 19위 수준이었다.

주자 상황 역시 상대적으로 낮은 중요도를 보였으며, 3루 주자 존재 여부가 상위 20개 변수 중 가장 낮은 수준이었다.

Out State는 0/1/2아웃의 범주형 변수로 처리하였으나 최종 SHAP 중요도에서 각각 **28위와 38위** 수준으로 나타났다.

따라서 **모든 경기 상황 변수가 동일하게 중요한 것은 아니었으며**, 특히 Score Difference가 높은 중요도를 보인 것과 달리 Count, 주자, Out State의 상대적 예측 중요도는 낮았다.


---

## 4. Robustness Check: Season Baseline Usage

본 연구의 종속변수는 다음과 같이 정의되어 있다.

**usage_change = post_usage - baseline_usage**

따라서 `baseline_usage`는 이미 종속변수 계산에 직접 포함되어 있으므로 주 분석 모델에서는 독립변수로 사용하지 않았다.

다만 baseline 구사율에 따라 `usage_change`가 가질 수 있는 범위가 구조적으로 달라질 수 있으므로, 이러한 특성이 주요 결과를 만들어낸 것은 아닌지 확인하기 위해 `baseline_usage`를 독립변수로 추가한 별도의 robustness model을 학습하였다.

Baseline Usage를 포함한 모델에서는 `baseline_usage`가 가장 높은 SHAP 중요도를 보였다.

그러나 핵심 변수들의 중요도는 baseline 추가 이후에도 크게 변하지 않았다.

| 변수 | Main Model | + Baseline Usage |
|---|---:|---:|
| Home Run | 0.0177 | 0.0175 |
| Score Difference | 0.0085 | 0.0084 |
| Launch Speed | 0.0076 | 0.0076 |

세 변수의 중요도 순서는 두 모델 모두 다음과 같이 동일하게 유지되었다.

**Home Run > Score Difference > Launch Speed**

따라서 주 분석에서 나타난 핵심 패턴은 **시즌 baseline 구사율의 구조적 영향을 추가로 고려한 경우에도 안정적으로 유지되었다.**

단, `baseline_usage`는 종속변수인 `usage_change`의 계산에 직접 포함되는 변수이므로 baseline을 포함한 모델의 설명력 증가를 일반적인 예측 성능 향상으로 해석하지 않았다.


---

## 5. 종합 결과 및 한계

GBM과 SHAP 분석을 통해 장타 허용 이후의 구종 회피 강도가 모든 상황에서 동일하지 않음을 확인하였다.

가장 높은 예측 중요도를 보인 변수는 **Home Run 여부**였으며, 홈런을 허용한 경우 이후 해당 구종을 더 강하게 회피하는 방향이 나타났다.

**Score Difference** 역시 높은 중요도를 보였으며, 점수차에 따라 회피 강도가 달라지는 비선형적 패턴이 관찰되었다.

또한 **Launch Speed**가 높을수록 이후 해당 구종을 더 강하게 회피하는 방향이 나타났다.

따라서 본 분석에서 가장 일관되게 나타난 핵심 요인은 다음과 같다.

**Home Run → Score Difference → Launch Speed**

반면 Count State, 주자 상황, Out State의 상대적 중요도는 낮게 나타났다.

다만 본 분석에는 다음과 같은 한계가 존재한다.

1. **낮은 모델 설명력**
   - 최종 GBM의 R²는 0.0352로 낮았다.
   - 따라서 실제 투수의 재대결 구종 선택은 본 모델에 포함된 변수 외에도 투수의 레퍼토리, 타자 특성, 포수의 사인, 경기 전략 등 다양한 요인의 영향을 받을 가능성이 있다.

2. **SHAP의 비인과적 해석**
   - SHAP 값은 GBM의 예측 구조를 설명하는 지표이며 각 변수의 인과효과를 의미하지 않는다.
   - 따라서 본 연구의 결과는 각 변수와 구종 회피 사이의 **예측적 연관성**으로 해석하였다.

3. **독립변수 간 상관관계**
   - Home Run, Launch Speed, Launch Angle, Estimated wOBA 등은 서로 관련된 정보를 포함한다.
   - 따라서 SHAP 중요도가 여러 변수 사이에 분산될 수 있으며, 변수별 Mean |SHAP|의 차이를 각 변수의 독립적인 효과 크기로 해석하지 않았다.

4. **투구 특성 변수의 해석**
   - Release Speed, Movement, Plate Location 등은 투수 및 구종의 고유한 특성을 상당 부분 반영할 수 있다.
   - 따라서 해당 변수들은 회피 행동 자체의 원인이라기보다 투구 특성을 통제하기 위한 변수의 성격으로 해석하였다.

결과적으로 본 분석은 특정 변수 하나가 장타 이후 구종 회피를 결정한다고 주장하기보다,

> **Home Run 여부, Score Difference, Launch Speed가 장타 이후 해당 구종의 사용 변화와 상대적으로 강하고 일관된 예측적 연관성을 보였다**

는 점을 주요 결과로 제시한다.
