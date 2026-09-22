# 2주차

## 03-1 최근접 이웃 회귀
- https://colab.research.google.com/drive/1V7reROGNQhJiyQDr0AqfQKjdipENzwtK

~~~ python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)

knr.score(test_input, test_target)
~~~

### 결정 계수($R^2$)
결정 계수는 회귀 모델의 성능을 평가하는 데 사용되는 지표 중 하나입니다. 모델이 얼마나 데이터를 잘 설명하는지, 또는 종속 변수의 분산을 독립 변수가 얼마나 잘 설명하는지를 나타냅니다.

#### 개념:

* $R^2$는 0과 1 사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 더 잘 설명한다고 해석됩니다.
* $R^2$ 값이 1이면 모델이 종속 변수의 모든 변동을 완벽하게 설명한다는 의미입니다 (현실에서는 거의 불가능).
* $R^2$ 값이 0이면 모델이 종속 변수의 변동을 전혀 설명하지 못한다는 의미이며, 이는 모델의 예측이 평균을 예측하는 것보다 나을 것이 없다는 뜻입니다.
* 음수 값도 나올 수 있는데, 이는 모델이 무작위로 예측하거나 평균을 예측하는 것보다도 성능이 더 나쁘다는 것을 의미합니다. 이는 모델이 데이터에 전혀 맞지 않음을 강력히 시사합니다.
* 계산 방식 (간략히): $R^2 = 1 - (SS_{res} / SS_{tot})$
* $SS_{res}$ (잔차 제곱합): 실제 값과 모델이 예측한 값 사이의 차이(잔차)를 제곱하여 모두 더한 값입니다. 모델이 얼마나 오차를 가지고 있는지를 나타냅니다.
* $SS_{tot}$ (총 제곱합): 실제 값과 실제 값들의 평균 사이의 차이를 제곱하여 모두 더한 값입니다. 종속 변수의 전체 변동성을 나타냅니다.
결국, $R^2$는 모델이 설명하지 못하는 잔차의 변동($SS_{res}$)이 전체 변동($SS_{tot}$)에서 차지하는 비율을 1에서 뺀 값입니다. 즉, 모델이 설명하는 변동의 비율을 나타냅니다.

#### 장점과 한계점:

* 장점: 이해하기 쉽고, 모델의 설명력을 직관적으로 파악할 수 있습니다.
* 한계점:
  * 독립 변수의 개수가 늘어나면 $R^2$ 값은 항상 증가하거나 유지됩니다. 이는 새로운 변수가 모델의 설명력을 실제로 높이지 않아도 일어날 수 있기 때문에, 변수가 많다고 해서 무조건 좋은 모델이라고 할 수 없습니다. 이 문제를 보완하기 위해 '조정된 결정 계수(Adjusted $R^2$)'를 사용하기도 합니다.
  * 모델이 데이터에 과대적합(Overfitting)되었는지 여부는 $R^2$만으로는 판단하기 어렵습니다. 훈련 데이터에 대해서는 $R^2$가 매우 높게 나올 수 있지만, 새로운 데이터에 대해서는 성능이 좋지 않을 수 있습니다.

따라서 $R^2$는 회귀 모델의 성능을 평가하는 유용한 지표이지만, 이 값 하나만으로 모델의 좋고 나쁨을 최종적으로 판단하기보다는 다른 평가 지표(예: MAE, MSE, RMSE)와 함께 모델의 전반적인 특성을 고려하여 해석하는 것이 중요합니다.


### 과대적합(Overfitting)과 과소적합(Underfitting)
- 과대적합(Overfitting): 모델이 훈련 데이터에 너무 잘 맞춰져서, 새로운 데이터에 대한 일반화 성능이 떨어지는 현상입니다.
- 과소적합(Underfitting): 모델이 훈련 데이터에 충분히 학습되지 않아, 훈련 데이터와 새로운 데이터 모두에서 성능이 낮은 현상입니다.

~~~ python
# k-최근접 이웃 회귀 객체를 만듭니다
knr = KNeighborsRegressor()
# 5에서 45까지 x 좌표를 만듭니다
x = np.arange(5, 45).reshape(-1, 1)

# n = 1, 5, 10일 때 예측 결과를 그래프로 그립니다.
for n in [1, 5, 10]:
    # 모델 훈련
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    # 지정한 범위 x에 대한 예측 구하기
    prediction = knr.predict(x)
    # 훈련 세트와 예측 결과 그래프 그리기
    plt.scatter(train_input, train_target)
    plt.plot(x, prediction)
    plt.title('n_neighbors = {}'.format(n))
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()

~~~

확인 문제 코드는 n_neighbors 매개변수(이웃의 개수)를 1, 5, 10으로 변경하면서 K-최근접 이웃 회귀 모델이 훈련 데이터에 어떻게 적합되는지 시각적으로 보여줍니다. 이를 통해 모델의 과대적합과 과소적합 경향을 설명할 수 있습니다.

#### n_neighbors = 1 (과대적합 경향):

* 첫 번째 그래프(n_neighbors = 1)를 보면, 모델의 예측선이 훈련 데이터의 각 점을 거의 완벽하게 통과하거나 매우 가깝게 따라갑니다. 이는 모델이 훈련 데이터의 노이즈나 특정 패턴까지 학습하려 시도했음을 의미합니다.
* 이 경우 모델은 훈련 데이터에 너무 **과대적합(Overfitting)**되었다고 볼 수 있습니다. 훈련 데이터에 대해서는 거의 완벽한 예측을 할 수 있지만, 이 모델이 실제 세계의 새로운 데이터(훈련 데이터에 없던)에 대해서는 성능이 급격히 떨어질 가능성이 높습니다. 모델의 유연성이 너무 높아 분산이 높고 편향이 낮은 상태입니다.

#### n_neighbors = 10 (과소적합 경향):

* 세 번째 그래프(n_neighbors = 10)를 보면, 모델의 예측선이 훈련 데이터의 전반적인 추세를 따르기는 하지만, 각 데이터 포인트의 세부적인 변화는 잘 포착하지 못하고 상당히 평탄하게 그려집니다.
* 이는 모델이 데이터를 너무 단순하게 해석하여 **과소적합(Underfitting)**되었다고 볼 수 있습니다. 모델이 데이터의 복잡한 관계를 제대로 학습하지 못하고 일반화가 지나치게 된 상태입니다. 훈련 데이터에 대해서도 충분히 좋은 성능을 내지 못할 가능성이 높으며, 편향이 높고 분산이 낮은 상태입니다.

#### n_neighbors = 5 (적절한 균형의 가능성):

* 두 번째 그래프(n_neighbors = 5)는 n_neighbors = 1보다는 부드럽고 n_neighbors = 10보다는 데이터의 패턴을 더 잘 반영하는 중간 형태를 보여줍니다. 이 경우, 모델이 훈련 데이터의 주요 패턴을 잘 학습하면서도 과도한 노이즈에는 덜 민감하여, 새로운 데이터에 대해서도 비교적 좋은 성능을 낼 수 있는 적절한 균형점을 찾았을 가능성이 있습니다.

결론적으로, n_neighbors 값은 K-최근접 이웃 모델의 복잡도를 조절하는 하이퍼파라미터이며, 이 값을 어떻게 설정하느냐에 따라 모델이 과대적합되거나 과소적합될 수 있습니다. n_neighbors가 작으면 과대적합되기 쉽고, n_neighbors가 크면 과소적합되기 쉽습니다. 따라서 최적의 n_neighbors 값을 찾는 것이 중요합니다.

## 03-2 선형 회귀
- https://colab.research.google.com/drive/1qUPtMxHGKT3kXpMxz-cKqbyN212Rpi44

### k-최근접 이웃 회귀의 한계

~~~ python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
# k-최근접 이웃 회귀 모델을 훈련합니다
knr.fit(train_input, train_target)

# 50cm 농어의 이웃을 구합니다
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그립니다
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 100cm 농어의 이웃을 구합니다
distances, indexes = knr.kneighbors([[100]])

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그립니다
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 100cm 농어 데이터
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
~~~

- 50cm 농어의 무게를 예측했습니다. 결과는 [1033.33333333]이었습니다. 이 예측값은 훈련 세트에서 50cm에 가장 가까운 3개의 농어 무게를 평균한 값입니다.

- 100cm 농어의 무게를 예측했습니다. 놀랍게도 결과는 50cm 농어와 동일한 [1033.33333333]이었습니다. 시각화된 그래프(YIMtpZKPfQjc 셀의 출력 이미지)를 보면 100cm 농어의 가장 가까운 이웃 3개도 50cm 농어의 이웃과 동일한 점들임을 알 수 있습니다.

k-최근접 이웃 회귀의 한계: 이 결과는 k-최근접 이웃 회귀 모델의 주요 한계점 중 하나를 보여줍니다. 이 모델은 훈련 데이터의 범위를 벗어나는 새로운 데이터 포인트에 대해 예측할 때, 단순히 가장 가까운 이웃들의 평균을 사용합니다. 즉, 훈련 데이터에 존재하지 않는 새로운 패턴이나 트렌드를 **외삽(extrapolate)**할 수 없습니다. 100cm 농어처럼 훈련 데이터의 최대 길이를 훨씬 초과하는 값에 대해서도 모델은 훈련 데이터 내의 가장 큰 값들에 해당하는 이웃들의 평균을 그대로 반환하게 되어 현실적인 예측을 하지 못하게 됩니다. 이 모델은 훈련 데이터의 국소적인 패턴에만 의존하기 때문에, 데이터가 부족한 영역이나 훈련 데이터의 범위를 벗어나는 영역에서는 예측 성능이 떨어질 수 있습니다.

### 선형 회귀
~~~ python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# 50cm 농어에 대한 예측
print(lr.predict([[50]]))

print(lr.coef_, lr.intercept_)

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프를 그립니다
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
~~~

### 다항 회귀

~~~ python
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

print(train_poly.shape, test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

print(lr.coef_, lr.intercept_)

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
point = np.arange(15, 50)

# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
~~~

## 03-3 특성 공학(Feature Engineering)과 규제화(Regularization)
- https://colab.research.google.com/drive/1d4hQfK976D-mSKLpzNTB6-rX452zI07A

### 정의
* 특성 공학(Feature Engineering): 
  * 모델의 성능을 향상시키기 위해 원본 데이터를 변환하거나 새로운 특성을 생성하는 과정입니다. 예를 들어, 기존의 길이(length)와 무게(weight) 데이터를 사용하여 길이의 제곱(length^2)과 같은 새로운 특성을 만들어 모델에 입력할 수 있습니다.

* 규제화(Regularization):
  * 모델이 과대적합되는 것을 방지하기 위해 모델의 복잡성을 제한하는 기법입니다. 일반적으로 모델의 가중치에 패널티를 부여하여 큰 가중치가 되지 않도록 조정합니다.

* 특성 공학의 예시
  - 기존 데이터: 길이(length)와 무게(weight)
  - 새로운 특성 생성: 길이의 제곱(length^2), 길이와 무게의 곱(length * weight) 등
  - 이렇게 생성된 새로운 특성들은 모델이 데이터의 패턴을 더 잘 학습할 수 있도록 도와줍니다. 예를 들어, 길이와 무게 사이의 비선형 관계를 모델링할 때, 길이의 제곱(length^2)과 같은 특성을 추가하면 모델이 이러한 비선형 관계를 더 잘 포착할 수 있습니다.

- 규제화의 예시
  - L1 규제(Lasso): 가중치의 절댓값 합에 패널티를 부여하여 일부 가중치를 0으로 만들어 특성 선택(feature selection) 효과를 제공합니다.
  - L2 규제(Ridge): 가중치의 제곱합에 패널티를 부여하여 가중치가 너무 커지지 않도록 조정합니다. 이는 모델이 특정 특성에 과도하게 의존하지 않도록 도와줍니다.
  - Elastic Net: L1과 L2 규제를 결합한 방법으로, 두 규제의 장점을 모두 활용할 수 있습니다.

* 규제화의 필요성
  * 과대적합 방지: 모델이 훈련 데이터에 너무 잘 맞춰져서 새로운 데이터에 대한 일반화 성능이 떨어지는 것을 방지합니다.
  * 모델 단순화: 규제화를 통해 모델의 복잡성을 줄이고, 불필요한 특성을 제거하여 모델을 단순화할 수 있습니다.
  * 안정성 향상: 규제화를 통해 모델의 가중치가 너무 커지지 않도록 조정함으로써, 모델의 예측이 안정적이고 신뢰할 수 있도록 합니다.
  * 모델 해석 용이성: 규제화를 통해 모델의 가중치가 적절하게 조정되면, 모델의 예측에 대한 해석이 더 용이해집니다. 이는 특히 비즈니스 의사결정이나 과학적 연구에서 중요한 요소입니다.
  * 일반화 성능 향상: 규제화를 통해 모델이 새로운 데이터에 대해 더 나은 일반화 성능을 보일 수 있습니다.

### 사이킷런의 PolynomialFeatures

~~~ python
# PolynomialFeatures 임포트 (특성 공학)
from sklearn.preprocessing import PolynomialFeatures

# PolynomialFeatures 예시: 상수항 포함 (bias=True가 기본값)
poly = PolynomialFeatures()
# fit() 메서드는 입력 데이터의 특성 조합을 학습합니다.
# 여기서는 2와 3이라는 두 개의 특성이 있고, 이 특성들을 기반으로 새로운 다항 특성을 어떻게 생성할지 학습합니다.
# 실제 특성 값을 변환하는 것이 아니라, 어떤 특성 조합을 만들지 규칙을 파악하는 단계입니다.
poly.fit([[2, 3]])
# transform() 메서드는 fit()에서 학습한 규칙을 바탕으로 실제 데이터를 변환합니다.
# 원본 특성 [2, 3]을 바탕으로 상수항(1), 원본 특성(2, 3), 제곱 특성(2^2=4, 3^2=9), 
# 그리고 교차항(2*3=6)을 포함하는 새로운 특성 배열을 생성합니다.
print(poly.transform([[2, 3]]))

# 생성된 특성 이름 확인
poly.get_feature_names_out()

# PolynomialFeatures 예시: 상수항 제외 (include_bias=False)
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))

# 훈련 세트에 PolynomialFeatures 적용 (degree=2, bias 제외)
poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)

# 생성된 훈련 세트의 특성 개수 확인
print(train_poly.shape)

poly.get_feature_names_out()

# 테스트 세트에 동일한 PolynomialFeatures 적용
test_poly = poly.transform(test_input)
~~~

### 다중 회귀(Multiple Regression) 모델 훈련 및 평가

~~~ python
# LinearRegression 모델 훈련 (degree=2 특성 사용)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

# LinearRegression 모델 테스트 세트 스코어
print(lr.score(test_poly, test_target))

# 고차 다항 특성 생성 (degree=5, bias 제외)
poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# 5차 다항 특성으로 변환된 훈련 세트의 특성 개수 확인
print(train_poly.shape)

# LinearRegression 모델 훈련 (degree=5 특성 사용) - 과대적합 발생
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

# LinearRegression 모델 테스트 세트 스코어 (degree=5 특성 사용) - 과대적합으로 성능 저하
print(lr.score(test_poly, test_target))
~~~

### 릿지(Ridge) 회귀 평가:

1.  **과대적합 문제**: `degree=5`의 다항 특성으로 훈련된 일반 선형 회귀 모델(규제 없음)은 훈련 세트에서 `0.9999`라는 거의 완벽한 점수를 얻었지만, 테스트 세트에서는 `-144.40`이라는 매우 낮은(음수) 점수를 보여 극심한 **과대적합(Overfitting)** 문제를 겪었습니다. (셀 `UffVFVTGP8xj` 및 `GtITdlYFg7AY`)
2.  **릿지 적용 후 성능**: 특성 스케일링을 거친 후, 릿지 회귀를 적용하여 `alpha=0.1`일 때 가장 좋은 성능을 보였습니다. (셀 `HC2D_EX4orCj`의 그래프)
    *   훈련 세트 점수: `0.9903` (셀 `5S5vhi-vhjzT`)
    *   테스트 세트 점수: `0.9827` (셀 `5S5vhi-vhjzT`)
3.  **평가**: 릿지 회귀는 L2 규제를 통해 모델의 복잡성을 효과적으로 줄여 과대적합을 완화했습니다. 훈련 세트와 테스트 세트의 점수 차이가 크게 줄어들고, 테스트 세트 점수가 크게 개선되어 일반화 성능이 향상되었음을 알 수 있습니다. 릿지는 모든 특성의 계수를 0에 가깝게 줄이지만 완전히 0으로 만들지는 않습니다.

### 라쏘(Lasso) 회귀 평가:

1.  **라쏘 적용 후 성능**: 릿지와 마찬가지로 특성 스케일링 후 라쏘 회귀를 적용했으며, `alpha=10`일 때 좋은 테스트 세트 점수를 보였습니다. (셀 `7rkH8Dvzh9UI`의 그래프)
    *   훈련 세트 점수: `0.9888` (셀 `t4uFD9Flh_Dw`)
    *   테스트 세트 점수: `0.9824` (셀 `t4uFD9Flh_Dw`)
2.  **특성 선택(Feature Selection)**: 라쏘 회귀의 가장 큰 특징은 L1 규제를 사용하여 일부 특성의 계수를 아예 0으로 만들 수 있다는 것입니다. 노트북의 마지막 셀(`z_bQc3s8Uoai`)에서 `np.sum(lasso.coef_ == 0)`을 출력한 결과 `40`이라는 숫자가 나왔는데, 이는 `degree=5`로 생성된 총 55개의 특성 중 40개의 특성 계수를 0으로 만들었다는 의미입니다. 즉, 라쏘는 15개의 특성만을 사용해 예측을 수행합니다.
3.  **평가**: 라쏘 회귀 역시 릿지 회귀와 유사하게 과대적합을 효과적으로 방지하고 모델의 일반화 성능을 향상시켰습니다. 특히, 불필요하거나 중요도가 낮은 특성을 자동으로 제거하여 모델을 더 간결하게 만들고 해석 가능성을 높이는 특성 선택의 이점을 제공합니다.

### 결론:

두 규제 모델(릿지, 라쏘) 모두 `degree=5` 다항 특성에서 발생한 심각한 과대적합 문제를 해결하여 모델의 테스트 성능을 크게 향상시켰습니다. 두 모델 모두 약 `0.98` 수준의 테스트 세트 R-제곱 점수를 달성하여 유사한 예측 성능을 보였습니다.

*   **릿지**: 모든 특성을 유지하면서 계수를 줄여 모델의 복잡성을 제어합니다.
*   **라쏘**: 일부 특성을 완전히 제거하여 모델을 더 단순하게 만들고, 특성 선택의 효과를 제공합니다.