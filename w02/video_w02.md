# 2주차 강의 비디오

## 1단계:신경망과 딥러닝 - 2. 신경망과 로지스틱회귀

### 동영상

* (C1W2L01) Binary Classification
* (C1W2L02) Logistic Regression
* (C1W2L03) Logistic Regression Cost Function
* (C1W2L04) Gradient Descent
* (C1W2L05) Derivatives
* (C1W2L06) More Derivative Examples
* (C1W2L07) Computation Graph
* (C1W2L08) Derivatives With Computation Graph
* (C1W2L09) Logistic Regression Gradient Descent
* (C1W2L10) Gradient Descent on m Examples

### 정리

* [노트북LM](https://notebook.google.com/notebook/cf01c4ee-33d7-4098-a465-1c79fc8119ae)
* [강의 노트](video_note_c1w2-1.pdf)

### 강의 내용 요약
제시된 자료는 **신경망 프로그래밍의 기초**가 되는 **이진 분류**와 **로지스틱 회귀** 알고리즘의 핵심 원리를 설명합니다. 데이터 효율성을 높이기 위한 **벡터화**와 행렬 표기법을 다루며, 모델 학습의 핵심인 **경사 하강법**과 **비용 함수**의 개념을 상세히 소개합니다. 특히 복잡한 미분 계산을 시각화하는 **계산 그래프**를 통해 **정방향 전파**와 **역전파**의 메커니즘을 직관적으로 이해하도록 돕습니다. 결과적으로 학습자는 **미분과 연쇄 법칙**을 활용해 매개변수를 최적화하고 성능을 개선하는 전체적인 과정을 학습하게 됩니다. 이는 대규모 데이터를 처리하는 현대 딥러닝 구현의 필수적인 이론적 토대를 제공합니다.


## 1단계:신경망과 딥러닝 - 3. 파이썬과 벡터화

### 동영상

* (C1W2L11) Vectorization
* (C1W2L12) More Vectorization Examples
* (C1W2L13) Vectorizing Logistic Regression
* (C1W2L14) Vectorizing Logistic Regression's Gradient Computation
* (C1W2L15) Broadcasting in Python
* (C1W2L16) A Note on Python/Numpy Vectors
* (C1W2L17) Quick Tour of Jupyter/Python Notebooks
* (C1W2L18) Explanation of Logistic Regression's Cost Function

### 정리

* [노트북LM](https://notebook.google.com/notebook/6b5617a2-c9f4-4162-bd98-157f785f54b6)
* [강의 노트](video_note_c1w2-2.pdf)


### 강의 내용 요약
**C1W2L11부터 C1W2L18까지의 동영상 강의 내용**을 순서대로 정돈하여, 혼자서도 개념과 구현 방식을 명확히 학습할 수 있도록 요약·정리해 드립니다.

---

### 1. C1W2L11: Vectorization (벡터화)
* **벡터화의 정의**: 코드에서 명시적인 `for` 루프를 제거하는 기법이자 딥러닝 프로그래밍의 핵심 기술입니다. 대용량 데이터셋을 학습시킬 때 실행 속도를 비약적으로 향상시킵니다.
* **원리 및 예시**: $z = w^T x + b$ 계산 ($w, x \in \mathbb{R}^{n_x}$)
  * **비벡터화 구현**: `for` 루프를 돌며 $i=1 \dots n_x$에 대해 `z += w[i] * x[i]`를 누적 계산합니다.
  * **벡터화 구현**: NumPy 내장 함수인 `np.dot(w, x) + b`를 사용합니다.
* **실행 속도 비교 (1,000,000차원 벡터 내적 실험)**:
  * 벡터화 연산(`np.dot`): 약 **1.5ms** 소요.
  * `for` 루프 연산: 약 **480~500ms** 소요 (약 **300배** 이상 느림).
* **하드웨어 병렬화**: CPU와 GPU 모두 **SIMD**(Single Instruction Multiple Data) 병렬 명령어를 지원하므로, `np.dot`과 같은 벡터화 함수를 사용하면 파이썬이 자동으로 병렬 처리를 수행합니다.

---

### 2. C1W2L12: More Vectorization Examples (벡터화 예제 더 보기)
* **기본 원칙**: 가능하면 `for` 루프 대신 NumPy 내장 함수를 적극 활용해야 합니다.
* **대표적 벡터화 사례**:
  1. **행렬-벡터 곱 ($u = A v$)**: 이중 `for` 루프 대신 `u = np.dot(A, v)`로 한 줄에 처리합니다.
  2. **원소별 지수/로그/절댓값 연산**: `for` 루프 대신 `np.exp(v)`, `np.log(v)`, `np.abs(v)`, `np.maximum(v, 0)`, `v**2`, `1/v` 등 벡터 단위 내장 함수를 사용합니다.
* **로지스틱 회귀 경사 계산의 1차 벡터화**:
  * 특성 개수($n_x$)에 관한 `for` 루프 제거: `dw`를 `np.zeros((n_x, 1))` 벡터로 초기화한 뒤 `dw += x^{(i)} * dz^{(i)}` 및 `dw /= m`으로 처리합니다.
  * (이 단계에서는 샘플 $m$개에 대한 outer `for` 루프만 남게 됩니다).

---

### 3. C1W2L13: Vectorizing Logistic Regression (로지스틱 회귀 벡터화)
* **목표**: $m$개의 전체 훈련 샘플에 대한 **정방향 전파(Forward Propagation)**를 `for` 루프 없이 한 번에 계산합니다.
* **데이터 행렬화**:
  * 입력 행렬 $X$: $m$개 샘플 $x^{(1)}, \dots, x^{(m)}$을 열 방향으로 쌓은 $(n_x, m)$ 차원 행렬.
* **선형 연산 벡터화 ($Z$)**:
  * $Z = [z^{(1)}, \dots, z^{(m)}]$는 $(1, m)$ 행 벡터가 됩니다.
  * **수식 및 코드**: $Z = w^T X + b \quad \rightarrow \quad$ `Z = np.dot(w.T, X) + b`.
  * 스칼라 값 $b$는 **브로드캐스팅**에 의해 자동으로 $(1, m)$ 행 벡터로 확장되어 더해집니다.
* **활성화 함수 벡터화 ($A$)**:
  * $A = [a^{(1)}, \dots, a^{(m)}] = [\sigma(z^{(1)}), \dots, \sigma(z^{(m)})]$.
  * **수식 및 코드**: $A = \sigma(Z)$ (벡터화된 시그모이드 함수 적용).

---

### 4. C1W2L14: Vectorizing Logistic Regression's Gradient Computation (로지스틱 회귀의 경사 계산 벡터화)
* **목표**: $m$개 전체 샘플에 대한 **역방향 전파(Backpropagation) 및 경사 계산**을 완전히 벡터화합니다.
* **오차 벡터화 ($dZ$)**:
  * $dZ = [dz^{(1)}, \dots, dz^{(m)}] = A - Y \quad (\text{차원: } (1, m))$. (여기서 $Y = [y^{(1)}, \dots, y^{(m)}]$).
* **가중치 및 편향 경사 벡터화 ($dw, db$)**:
  * $db = \frac{1}{m} \sum_{i=1}^m dz^{(i)} \quad \rightarrow \quad$ `db = (1/m) * np.sum(dZ)`.
  * $dw = \frac{1}{m} X dZ^T \quad \rightarrow \quad$ `dw = (1/m) * np.dot(X, dZ.T)` $\quad (\text{차원: } (n_x, 1))$.
* **경사 하강법 1회 반복의 완전 벡터화 구조**:
  1. $Z = \text{np.dot}(w^T, X) + b$
  2. $A = \sigma(Z)$
  3. $dZ = A - Y$
  4. $dw = \frac{1}{m} X dZ^T, \quad db = \frac{1}{m} \text{np.sum}(dZ)$
  5. 매개변수 갱신: $w := w - \alpha \, dw, \quad b := b - \alpha \, db$.
* **참고**: 경사 하강법 반복 횟수(예: 1,000회)를 위한 가장 바깥쪽 `for` 루프는 유지되어야 합니다.

---

### 5. C1W2L15: Broadcasting in Python (파이썬의 브로드캐스팅)
* **개념**: 차원이 서로 다른 행렬/벡터 간의 연산을 수행할 때, 파이썬 NumPy가 자동으로 차원을 맞추어 원소별(element-wise) 연산을 가능하게 하는 기법입니다.
* **실제 예제 (음식 칼로리 비율 계산)**:
  * $(3, 4)$ 크기의 음식 영양소 행렬 $A$에 대해 열별 합을 구함: `cal = A.sum(axis=0)` ($axis=0$은 세로 방향 합, 결과 차원: $(1, 4)$).
  * 백분율 계산: `percentage = 100 * A / cal.reshape(1, 4)`.
* **브로드캐스팅 동작 규칙**:
  * $(m, n)$ 행렬과 $(1, n)$ 행 벡터 연산 $\rightarrow$ 행 벡터를 세로로 $m$번 복사하여 $(m, n)$으로 확장 후 연산.
  * $(m, n)$ 행렬과 $(m, 1)$ 열 벡터 연산 $\rightarrow$ 열 벡터를 가로로 $n$번 복사하여 $(m, n)$으로 확장 후 연산.
  * $(m, 1)$ 또는 $(1, n)$ 벡터와 스칼라 실수 연산 $\rightarrow$ 실수를 전체 크기로 복사하여 연산.
* **권장 사항**: 차원의 모호함을 줄이고 버그를 방지하기 위해 `reshape()` 함수를 명시적으로 사용하는 것이 좋습니다 (상수 시간 $O(1)$ 연산으로 매우 가볍습니다).

---

### 6. C1W2L16: A Note on Python/Numpy Vectors (파이썬/NumPy 벡터에 관한 주의사항)
* **Rank-1 배열의 위험성**:
  * `a = np.random.randn(5)` 와 같이 차원을 지정하면 차원이 `(5,)`인 **Rank-1 배열**이 생성됩니다.
  * 행 벡터도 열 벡터도 아니므로, $a^T$가 $a$와 동일하게 출력되거나, `np.dot(a, a.T)`의 결과가 행렬이 아닌 스칼라 수치로 나오는 등 직관적이지 않은 버그를 유발합니다.
* **실무 권장 규칙**:
  1. 명시적 2D 행렬/벡터 표현 사용:
     * 열 벡터: `a = np.random.randn(5, 1)` $\rightarrow$ 차원 `(5, 1)`.
     * 행 벡터: `a = np.random.randn(1, 5)` $\rightarrow$ 차원 `(1, 5)`.
  2. `assert(a.shape == (5, 1))` 문을 사용하여 차원을 주기적으로 검증 (문서화 역할 및 저렴한 검증).
  3. Rank-1 배열이 생성된 경우 `a = a.reshape(5, 1)`을 통해 명시적으로 변환.

---

### 7. C1W2L17: Quick Tour of Jupyter/iPython Notebooks (Jupyter/iPython Notebook 둘러보기)
* **주요 구성 요소 및 사용법**:
  * **마크다운 셀**: 설명 텍스트를 포함하며, 더블 클릭 후 `Shift + Enter`를 누르면 깔끔한 글씨로 렌더링됩니다.
  * **코드 셀**: 실습 코드를 작성하는 회색 블록입니다. 코드는 주석 `### START CODE HERE ###`와 `### END CODE HERE ###` 사이에 작성합니다.
  * **실행 단축키**: 셀 선택 후 `Shift + Enter`.
  * **커널(Kernel) 관리**: 코드는 서버 커널에서 실행됩니다. 오랫동안 방치되거나 연결이 끊겨 오류가 발생하면 상단 메뉴의 **Kernel $\rightarrow$ Restart**를 눌러 재시작합니다.
  * **전제 셀 실행**: 모듈 임포트(`import numpy as np`)나 변수 초기화 코드가 있는 상단 셀은 직접 코드를 수정하지 않더라도 반드시 순서대로 실행해야 합니다.
  * **과제 제출**: 구현 완료 후 우측 상단의 **Submit Assignment** 버튼으로 제출합니다.

---

### 8. C1W2L18: Explanation of Logistic Regression's Cost Function (로지스틱 회귀 손실 함수 설명)
* **목적**: 로지스틱 회귀에서 특정 손실/비용 함수를 사용하는 확률론적 배경을 설명합니다.
* **확률 정의 ($y \in \{0, 1\}$ 이진 분류)**:
  * 예측값 $\hat{y} = p(y=1|x)$
  * $y=1$일 때: $p(y|x) = \hat{y}$
  * $y=0$일 때: $p(y|x) = 1 - \hat{y}$
* **하나의 수식으로 통합**:
  * $p(y|x) = \hat{y}^y \, (1 - \hat{y})^{(1-y)}$.
  * ($y=1$이면 $\hat{y}^1 (1-\hat{y})^0 = \hat{y}$, $y=0$이면 $\hat{y}^0 (1-\hat{y})^1 = 1-\hat{y}$가 됨).
* **로그 확률 전환**:
  * 로그 함수는 단조 증가 함수이므로 $p(y|x)$를 최대화하는 것은 $\log p(y|x)$를 최대화하는 것과 같습니다.
  * $\log p(y|x) = y \log \hat{y} + (1-y) \log (1-\hat{y}) = -\mathcal{L}(\hat{y}, y)$.
  * 손실 함수 $\mathcal{L}$에 음수가 붙은 이유는, 확률을 **최대화**하는 문제가 손실을 **최소화**하는 문제와 일치하기 때문입니다.
* **전체 훈련 세트 $m$개에 대한 비용 함수 $J(w, b)$ 도출**:
  * $m$개 샘플이 독립동일분포(i.i.d.)라 가정할 때 전체 우도(Likelihood)는 각 확률의 곱 $\prod_{i=1}^m p(y^{(i)}|x^{(i)})$.
  * 로그 우도: $\sum_{i=1}^m \log p(y^{(i)}|x^{(i)}) = -\sum_{i=1}^m \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$.
  * **최대 우도 추정(MLE)** 원리에 따라 로그 우도를 최대화하기 위해, 음수를 제거한 비용 함수 $J(w, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$를 최소화하게 됩니다.

---

