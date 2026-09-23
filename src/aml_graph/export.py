from pathlib import Path
import numpy as np, pandas as pd

def export_results(f,edges,cmap,out_dir='outputs',top_n=50):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    nr=f[['gid','role','role_score','cluster_id','priority_score','evidence']].copy(); nr.role_score=nr.role_score.round(6); nr.priority_score=nr.priority_score.round(6); nr.to_csv(out/'nodes_roles.csv',index=False)
    e=edges.copy(); e['sc']=e.src.map(cmap); e['dc']=e.dst.map(cmap); internal=e[e.sc==e.dc].groupby('sc').sum_kzt.sum(); rows=[]
    for cid,g in f.groupby('cluster_id'):
        r=g.sort_values('priority_score',ascending=False); ns=int(g.is_seed.sum()); dom=g.role.value_counts().index[0]
        hyp=f'Связанная структура с {ns} seed; доминирующая роль: {dom}.' if ns>=2 else (f'Ветка вокруг 1 seed; доминирующая роль: {dom}.' if ns==1 else f'Наблюдаемое сообщество без seed; доминирующая роль: {dom}.')
        rows.append([int(cid),len(g),ns,round(float(internal.get(cid,0)),2),','.join(map(str,r.gid.head(5).astype('int64'))),hyp])
    pd.DataFrame(rows,columns=['cluster_id','n_nodes','n_seed','sum_kzt_internal','top_gids','hypothesis']).sort_values('cluster_id').to_csv(out/'clusters.csv',index=False)
    t=f.sort_values(['priority_score','gid'],ascending=[False,True]).head(top_n); pd.DataFrame({'rank':np.arange(1,len(t)+1),'gid':t.gid.astype('int64'),'role':t.role,'priority_score':t.priority_score.round(6),'why':t.evidence}).to_csv(out/'top_nodes.csv',index=False)
