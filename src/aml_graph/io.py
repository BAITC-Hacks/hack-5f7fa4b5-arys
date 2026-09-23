from pathlib import Path
import pandas as pd

def _read(path):
    return pd.read_parquet(path) if path.suffix.lower()=='.parquet' else pd.read_csv(path)

def load_data(data_dir='data'):
    d=Path(data_dir)
    def pick(name):
        p=d/f'{name}.parquet'; c=d/f'{name}.csv'
        if p.exists(): return p
        if c.exists(): return c
        raise FileNotFoundError(name)
    nodes,edges,tx=_read(pick('nodes')),_read(pick('edges')),_read(pick('transactions'))
    tx['date']=pd.to_datetime(tx['date'])
    return nodes,edges,tx
