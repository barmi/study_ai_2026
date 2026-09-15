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

~~~ python
# zip() 함수는 여러 개의 이터러블(예: 리스트)에서 각각의 요소를 하나씩 가져와 튜플로 묶는 함수
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

# 잘못된 샘플링 방법

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

train_input = fish_data[:35]
train_target = fish_target[:35]

test_input = fish_data[35:]
test_target = fish_target[35:]

kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# numPy를 사용한 올바른 샘플링 방법
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 올바른 샘플링 방법
train_input, test_input, train_target, test_target = train_test_split(input_arr, target_arr, test_size=0.3, random_state=42)

# ---------
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn.fit(train_input, train_target)
kn.score(test_input, test_target)
kn.predict(test_input)

~~~

- 교재에서는 수동으로 numpy 배열을 섞고 인덱스를 나누어 훈련 데이터와 테스트 데이터를 샘플링하는 방법을 보여주고 있습니다. 하지만 실제로는 `train_test_split` 함수를 사용하는 것이 더 간단하고 직관적입니다. 이 함수는 데이터를 무작위로 섞고 지정된 비율로 훈련 세트와 테스트 세트로 나누어 줍니다.
- `train_test_split` 함수의 주요 매개변수:
  - `test_size`: 테스트 세트의 비율을 지정합니다. 예를 들어, `test_size=0.3`이면 전체 데이터의 30%를 테스트 세트로 사용합니다.
  - `random_state`: 난수 생성기의 시드를 설정하여 결과를 재현 가능하게 합니다. 같은 시드를 사용하면 항상 같은 방식으로 데이터를 섞습니다.
  - 참고: [scikit-learn train_test_split 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

