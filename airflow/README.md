# Installataion

```
pip install apache-airflow
```


# How to Use

### 환경변수 설정
```
export AIRFLOW_HOME=/opt/ml/torchkpop/airflow
```

### DB init
```
airflow db init
```

### 사용자 생성
관련 문서 참조

### scheduler 실행
airflow scheduler

### !!만약 Unable to find any timezone configuration 에러가 뜬다면
```
apt-get install -y tzdata
```

# DAGS
01. file_mover.py
: /torchkpop/results 폴더 내의 결과 (ex: d3IPUJ42JO/60) 경로 중
csv 폴더만 남기고 삭제
sampled_images 에서 train 용 bbox 이미지들을 /torchkpop/body_embedding/train_images 폴더 내부로 복사