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