import numpy as np, pandas as pd, networkx as nx

def pct(s):
    s=pd.Series(s); o=pd.Series(0.,index=s.index); m=s>0
    if m.any(): o.loc[m]=s.loc[m].rank(method='average',pct=True)
    return o

def build_features(nodes,edges,tx,louvain_seed=42):
    G=nx.DiGraph(); G.add_nodes_from(nodes.gid.astype('int64'))
    for r in edges.itertuples(index=False): G.add_edge(int(r.src),int(r.dst),weight=float(r.sum_kzt),n_tx=int(r.n_tx))
    f=nodes.copy(); f['gid']=f.gid.astype('int64')
    i=edges.groupby('dst').agg(in_degree=('src','nunique'),amount_in=('sum_kzt','sum')).rename_axis('gid')
    o=edges.groupby('src').agg(out_degree=('dst','nunique'),amount_out=('sum_kzt','sum')).rename_axis('gid')
    f=f.merge(i,on='gid',how='left').merge(o,on='gid',how='left')
    f[['in_degree','out_degree','amount_in','amount_out']]=f[['in_degree','out_degree','amount_in','amount_out']].fillna(0)
    f[['in_degree','out_degree']]=f[['in_degree','out_degree']].astype(int)
    f['flow_ratio']=np.where(f.amount_in>0,f.amount_out/f.amount_in,np.nan)
    f['pagerank']=f.gid.map(nx.pagerank(G,weight='weight'))
    f['betweenness']=f.gid.map(nx.betweenness_centrality(G,normalized=True,weight=None))
    seeds=set(f.loc[f.is_seed,'gid'].astype(int))
    f['seed_coverage']=f.gid.map(lambda v:len(nx.ancestors(G,int(v))&seeds)+(int(v) in seeds))
    ss={}
    for c in nx.strongly_connected_components(G):
        for v in c: ss[v]=len(c)
    f['scc_size']=f.gid.map(ss); f['cycle_member']=f.scc_size>1
    din=tx.groupby(['dst','date']).sum_kzt.sum().rename('din'); dout=tx.groupby(['src','date']).sum_kzt.sum().rename('dout')
    d=pd.concat([din.rename_axis(['gid','date']),dout.rename_axis(['gid','date'])],axis=1).fillna(0).reset_index()
    d['matched']=np.minimum(d.din,d.dout)
    t=d.groupby('gid').agg(matched=('matched','sum'),day_in=('din','sum'),day_out=('dout','sum'),overlap_days=('matched',lambda s:int((s>0).sum())))
    t['same_day_ratio']=t.matched/t[['day_in','day_out']].max(axis=1).replace(0,np.nan)
    f=f.merge(t[['same_day_ratio','overlap_days']],left_on='gid',right_index=True,how='left'); f[['same_day_ratio','overlap_days']]=f[['same_day_ratio','overlap_days']].fillna(0)
    UG=nx.Graph(); UG.add_nodes_from(G.nodes())
    for r in edges.itertuples(index=False):
        u,v,w=int(r.src),int(r.dst),float(r.sum_kzt)
        if UG.has_edge(u,v): UG[u][v]['weight']+=w
        else: UG.add_edge(u,v,weight=w)
    cs=sorted(nx.community.louvain_communities(UG,weight='weight',seed=louvain_seed),key=lambda c:(-len(c),min(c)))
    cmap={v:k for k,c in enumerate(cs,1) for v in c}; f['cluster_id']=f.gid.map(cmap).astype(int)
    f['cross_cluster_degree']=f.gid.map(lambda v:sum(cmap[n]!=cmap[v] for n in (set(G.predecessors(v))|set(G.successors(v)))))
    for a,b in [('in_degree','in_p'),('out_degree','out_p'),('amount_in','ain_p'),('amount_out','aout_p'),('betweenness','bt_p'),('pagerank','pr_p'),('seed_coverage','seed_p'),('cross_cluster_degree','bridge_p')]: f[b]=pct(f[a])
    f['degree_p']=pct(f.in_degree+f.out_degree); f['volume_p']=pct(f.amount_in+f.amount_out)
    f['retention_obs']=np.where(f.amount_in>0,np.clip(1-f.flow_ratio,0,1),0)
    r=f.flow_ratio.replace([0,np.inf,-np.inf],np.nan); f['flow_closeness']=np.where((f.amount_in>0)&(f.amount_out>0),np.exp(-np.abs(np.log(r))),0)
    f['degree_balance']=np.where((f.in_degree>0)&(f.out_degree>0),np.minimum(f.in_degree,f.out_degree)/np.maximum(f.in_degree,f.out_degree),0)
    return f,G,cmap
