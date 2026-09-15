# 1주차

## hg-mldl2
### 01-3. 마켓과 머신러닝
- https://colab.research.google.com/drive/1QuurJCEuBvRlXnsBYeXtkFtW9Jc7WHCg

~~~ python
# 예제 코드
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)

kn.predict([[30, 600]])
kn.predict([[30, 600], [25, 100], [25, 50]])



kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

print(35/49)

~~~

- 마지막의 print(35/49)는 35를 49로 나누고 그 결과를 출력합니다. 앞선 코드의 맥락에서, 35는 도미(bream)의 개수를 의미하고 49는 전체 생선의 개수를 의미합니다. 따라서 이 계산은 전체 생선 중에서 도미가 차지하는 비율을 나타내며, 이는 n_neighbors를 49로 설정한 K-최근접 이웃 분류기(kn49)의 정확도와 동일한 값입니다. 즉, 이 분류기는 모든 샘플을 도미로 예측했을 때 얻을 수 있는 점수를 보여줍니다.

- kn과 kn49는 모두 KNeighborsClassifier 객체이지만, 가장 큰 차이점은 이웃의 개수를 의미하는 n_neighbors 매개변수 설정에 있습니다.

  - kn 객체는 n_neighbors 매개변수를 지정하지 않고 생성되었습니다. KNeighborsClassifier의 기본값은 n_neighbors=5이므로, kn은 예측할 때 가장 가까운 5개의 이웃 데이터를 참고합니다.

  - 반면, kn49 객체는 KNeighborsClassifier(n_neighbors=49)와 같이 n_neighbors를 명시적으로 49로 설정하여 생성되었습니다. 이는 kn49가 예측할 때 가장 가까운 49개의 이웃 데이터를 참고한다는 의미입니다.

- 이 n_neighbors 값의 차이로 인해 두 모델은 새로운 데이터에 대해 다른 예측 결과를 보일 수 있습니다. 특히 kn49의 경우, 데이터 전체 개수(도미 35개 + 빙어 14개 = 49개)와 동일한 n_neighbors 값을 사용했기 때문에, 모든 데이터를 도미(Class 1)로 예측하려는 경향이 강해집니다. 이는 kn49.score가 35/49와 같은 0.714...를 반환하는 것에서 확인할 수 있습니다. 모든 샘플을 도미로 예측하여 도미의 비율만큼만 맞추는 것입니다.

### 02-1. 훈련 데이터와 테스트 데이터
- https://colab.research.google.com/drive/1BJ9R0mxGuZ4SMW2nKej4IjyuqSSjaC88
