
# pytorch-faster-r-cnn implementation

### 공식 논문
- GIRSHICK, Ross. Fast r-cnn. In: _Proceedings of the IEEE international conference on computer vision_. 2015. p. 1440-1448.

### 참고자료
	- https://towardsdatascience.com/understanding-and-implementing-faster-r-cnn-a-step-by-step-guide-11acfff216b0
	- https://github.com/jwyang/faster-rcnn.pytorch

### 학습
	- python train.py

### Inference
- python detect.py --source path/of/image --weights path/of/weights

### 정리
- 내가 모델 아키텍쳐를 구현함에 있어 목표점이 될만한 코드였음(참고자료 첫번째 블로그). 각 파트별 세부적인 설명뿐 아니라 코드 벡터화도 무척 잘 되있는듯.
- 한번 시험삼아 voc2007의 train셋을 올리고 미니배치 사이즈를 128까지 변경해가며 학습시켜봤는데 사이즈별 학습 속도 차이가 거의 없었음.
  backbone network는 pre-trained 된 네트워크를 사용하니 순수하게 anchor loss 계산하는 부분이 배치 사이즈 상관없이 연산시간이 동일하다는 의미.
