# LTE-Maritime 항적 데이터 기반 항로 예측 시각화 시스템 연구 (선박해양플랜트연구소)

- 참여 인원 및 기간
    - 2명 (Data Engineer, Modeling 담당)
    - 2023.01 ~ 현재
- 데이터: AIS, LTE-M 선박 항적 데이터 (위치 시계열 데이터)
- 과업 목표
    - 고해상도, 대용량 선박 항적 데이터 전처리: 초 단위로 저장되는 위치 시계열 데이터에 대한 데이터 샘플링, feature engineering, time irregularity normalization 등
    - 데이터 기반 항적 예측: 위치 시계열 데이터 예측에 활용되는 Deep Learning Model 기반의 정확도 높은 모델 개발
    - 선박 충돌 방지 등 해사 안전 서비스에 활용 가능한 시각화 시스템 구현
- 사용기술
    - Deep Learning Modeling : PyTorch LSTM, Transformer
    - Front-end: Flask, HTML
- 현황
    - 데이터 전처리 (진행중)
    - BiLSTM 기반 베이스라인 모델 구현
    - Transformer 기반 모델 구현 (진행중)
    - 2023년 말 KCI 학술지 게재 (예정)
