# KLS_Shap
- 점검/진단 부적정률 기인인자 도출 서비스
  - 부실에 해당되는 인자들을 도출하여 보고서나 정책 변경에 활용
  
- 코드 설명
  @shap
  - shap_preprocessing
    - 데이터 타입변경 / 데이터 정제 / 데이터 변환
  - shap_model
    - classification model: lightGBM
    - Interpretable model: Shap
  - shap_main
    - 데이터 학습 및 output 도출
  - utill 
    - 보편적인 코드 실행
