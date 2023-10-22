# 모델 설명
## feature
+ 각 패킷으로부터 delta_time, length, direction, protocol을 추출하여 4개의 feature를 생성
  + delta_time: 이전 패킷과의 타임스탬프 오차
  + length: 패킷의 길이
  + direction: 송신 또는 수신에 따라 0과 1로 설정
  + protocol: 프로토콜에 따라 0~1 사이의 값으로 정규화
+ 매 4개의 패킷에 대한 feature를 직렬화하여 4*4개의 feature를 생성

## 정규화 방식
+ delta_time: 1초를 넘을 경우 1초로 설정, 그렇지 않을 경우 0~1 사이의 값으로 정규화
+ length: 30을 넘을 경우 30으로 설정, 그렇지 않을 경우 0~1 사이의 값으로 정규화
+ direction: 송신 또는 수신에 따라 0과 1로 설정
+ protocol: 프로토콜에 따라 0~1 사이의 값으로 정규화

## label
1. 패킷과 연관된 단말 기기의 타입(Sensor, Actuator)
2. 패킷과 연관된 단말 기기의 제조사(Samsung, Aqara 등)
3. 패킷과 연관된 단말 기기의 모델명(SmartThings Multipurpose Sensor, Aqara Door/Window Sensor 등)

## model
1. OneVsOneClassifier
2. OneVsRestClassifier
3. RandomForestClassifier

# 모듈 설명
## log_conf.py
> 로깅에 관련된 모듈
+ init_logger(): 로깅 설정을 관리하는 함수

## flow.py
> Flow 객체 구현에 필요한 클래스들을 선언해둔 모듈
### FlowKey
+ FlowKey.sid: 송신 기기의 식별자
+ FlowKey.did: 수신 기기의 식별자
+ FlowKey.protocol: 패킷의 프로토콜
+ FlowKey.additional: 패킷 식별의 추가 정보
+ FlowKey.set_key(): 모든 attribute 초기화
### FlowValue
+ FlowValue.raw_time: 패킷에 기록된 실제 타임스탬프
+ FlowValue.direction: 패킷 전달 방향(0=정방향, 1=역방향)
+ FlowValue.length: 패킷의 길이
+ FlowValue.delta_time: 이전 패킷과의 타임스탬프 오차(첫 패킷은 0으로 설정)
+ FlowValue.protocol: 패킷의 프로토콜
+ FlowValue.set_raw_value(): delta_time을 제외한 모든 attribute 초기화
### Flows
+ Flows.value: {FlowKey:FlowValue} 형태의 Dictionary 타입 Flow 데이터
+ Flows.find(): FlowKey에 해당하는 Flow의 유무 반환
+ Flows.create(): 현재 등록되지 않은 FlowKey에 대한 새로운 Flow 생성
+ Flows.append(): 현재 등록된 Flow에 새로운 FlowValue를 append
+ Flows.sort(): 모든 Flow를 시간 순으로 정렬
+ Flows.tune(): 모든 FlowValue의 delta_time을 초기화
+ Flows.print() 모든 Flow를 텍스트 파일로 출력

## load_files.py
> 파일로 기록된 데이터의 입력에 대한 모듈
+ load_files(): pcapng 파일을 불러와 리스트 형태로 반환하는 함수
+ load_labels(): csv 파일을 불러와 리스트 형태로 반환하는 함수 

## preprocess.py
> 데이터 전처리에 관련된 모듈
+ normalize(): 데이터 정규화 함수

## learn.py
> 모델 학습에 관련된 모듈
+ ovo_run(): OneVsOneClassifier를 활용한 모델 학습 함수
+ ovr_run(): OneVsRestClassifier를 활용한 모델 학습 함수
+ rf_run(): RandomForestClassifier를 활용한 모델 학습 함수
+ learn(): 정의된 모델 학습 함수를 호출하는 함수

## evaluate.py
> 모델 평가에 관련된 모듈
+ evaluate(): 모델 평가 함수
+ make_heatmap(): 모델의 분류 결과를 히트맵으로 저장하는 함수
+ print_score(): 모델의 성능을 출력 및 저장하는 함수