from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash_operator import BashOperator

from datetime import datetime
from prometheus_client.parser import text_string_to_metric_families
import requests
import json
import pytz
import pandas as pd

labels = ['container_cpu_cfs_throttled_seconds_total',
          'container_cpu_load_average_10s',
          'container_cpu_system_seconds_total',
          'container_cpu_usage_seconds_total'
          'container_cpu_user_seconds_total',
          'container_fs_io_time_seconds_total',
          'container_memory_usage_bytes',
          'container_memory_failcnt',
          'container_processes',
          'container_tasks_state'
          ]
containers = ['mysql',
              'jwilder/nginx-proxy',
              'registry.gitlab.com/indivd/insights/indivd-analytics:webapp-latest',
              'jrcs/letsencrypt-nginx-proxy-companion',
              'gcr.io/google-containers/cadvisor:latest',
              'grafana/grafana'
              ]

def create_request(**context):
    _time = datetime.now(pytz.timezone('Etc/GMT')).strftime("%Y-%m-%dT%H:%M:%S.781Z")
    _req = f"http://{context['host']}/api/v1/query?query={context['label']}&time={_time}"

    _metrics = requests.get(_req)
    return _metrics.content.decode("utf-8")


def parse_metrics(**context):
    label_metrics = {}
    for label in context['labels']:
        server_metrics = {}
        _res = json.loads(context['ti'].xcom_pull(task_ids=f'create_request_{label}'))
        for i in _res['data']['result']:
            try:
                if i['metric']['image'] in context['containers']:
                    server_metrics[i['metric']['image']] = int(i['value'][0]) / 1024 / 1024
            except:
                pass
            label_metrics[label] = server_metrics
    )
    all_server = []
    all_values = []
    print(label_metrics)
    for container in containers:
        one_server = []
        one_values = []
        for label in labels:
            one_values.append(label_metrics[label][container])
            one_server.append(container + '_' + label)
    all_server += one_server
    all_values += one_values
    df = pd.DataFrame(
        [all_values],
        columns=all_server,
    )
    print('label_metrics: ', len(label_metrics))
    print('server_metrics: ', len(server_metrics))

    _json = df.to_dict(orient='records')
    requests.post('{context['host']}:5000/data', json=_json)
    return label_metrics
  
  
with DAG("metrics",
         start_date=datetime(2021, 1, 1),
         schedule_interval="5 * * * *",
         catchup=False) as dag:
    create_request_labels = [
        PythonOperator(
            task_id=f"create_request_{label}",
            python_callable=create_request,
            provide_context=True,
            op_kwargs={
                "label": label,
                'host': host
            }
        ) for label in labels
    ]

    parse_metrics = PythonOperator(
        task_id="parse_metrics",
        python_callable=parse_metrics,
        provide_context=True,
        op_kwargs={
            'containers': containers,
            'labels': labels
        }
    )

    create_request_labels >> parse_metrics
