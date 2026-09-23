import numpy as np

def validate_data(nodes,edges,tx):
    req={'nodes':{'gid','depth','is_seed'},'edges':{'src','dst','sum_kzt','n_tx','depth'},'transactions':{'src','dst','date','sum_kzt'}}
    fs={'nodes':nodes,'edges':edges,'transactions':tx}
    errors=[]
    for n,c in req.items():
        m=c-set(fs[n].columns)
        if m: errors.append(f'{n}: missing {sorted(m)}')
        if fs[n].isna().any().any(): errors.append(f'{n}: NULL found')
    if nodes.gid.duplicated().any(): errors.append('duplicate gid')
    if (edges.src==edges.dst).any(): errors.append('self-loop found')
    if (edges.sum_kzt<=0).any() or (tx.sum_kzt<=0).any(): errors.append('non-positive amount')
    if errors: raise ValueError('; '.join(errors))
    a=tx.groupby(['src','dst']).agg(tx_sum=('sum_kzt','sum'),tx_n=('sum_kzt','size')).reset_index()
    z=edges.merge(a,on=['src','dst'],how='outer',indicator=True)
    if not (z._merge=='both').all(): raise ValueError('edge pair mismatch')
    if not np.allclose(z.sum_kzt,z.tx_sum,atol=.01): raise ValueError('edge amount mismatch')
    if not (z.n_tx.astype(int)==z.tx_n.astype(int)).all(): raise ValueError('edge count mismatch')
