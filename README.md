# metafactor_DL
라이브러리는 에러난거 설치할 것

##########################################################

hotshot은 내가 만든 라이브러리니깐 설치 시도하지 말 것!!

##########################################################

일단 전부 돌아는 가게 만들어놨으니 이제 어떻게 돌리는지 시작

1.메일로 보낸 데이터 전부 다 받았다는 가정하에 최상단 셀 돌리고, "팩터값 계산" 탭 아래에 있는 것부터 한 방에 돌릴 것

2.1번 하고 난 뒤에 다시 돌릴 때는 "CausalAI(였지만 지금은 그냥 선작업)" 탭부터 돌리면 됨

3.손실함수 커스텀으로 만듬. 예측값 대로의 수익률과 label값의 수익률의 차이로 계산함. 그 덕에 model.fit, callback 못써서 코드가 짜증나짐

4.
4-1.완료 - 백테스팅 결과 PNL Daily로 나오도록 변경(훈련데이터 시점 수 : 120->3650)

4-3.완료 - PCA, PACF, 오버샘플링기존 코드 삭제

4-4.GAN 기반 data augment 모듈 오버샘플링 모듈 자리에 추가

5."딥-러닝"탭 첫 번째 셀에 하이퍼 파라미터 전부 선언하고 있으니 거기서 모델 튜닝할 것

(손튜닝 하는 이유 : 어느 데이터를 추가/제거할지 확실지 않아서 GridsearchCV 적용하기엔 애매함)

6.(후순위)어떤 데이터를 넣고, 뺄지 "딥-러닝"탭 두 번째 셀에 있는 목록 확인하고 여러 시도해볼 것

7.sequential 데이터로 변환하는 함수 검토해봐야 함.
