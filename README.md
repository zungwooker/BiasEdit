# BiasEdit

### train/
* 다양하게 훈련하는 방법이 포함되어 있음
* Base, Ours에 대한 Vanilla, LfF, LfF+BE 모두 완성되어 있음

### StaB/
* 이미지를 생성하는 과정이 2-step으로 만들어져 있음
* 1-step: Tag로 통계내고 bias가 있는지 없는지, 그 bias가 무엇인지 탐지함
* 2-step: 탐지된 bias를 이용해서 instruction을 넣어주면(json) 이에 맞게 이미지를 생성함
* 이미지 생성을 split을 이용하여 생성할 수 있음