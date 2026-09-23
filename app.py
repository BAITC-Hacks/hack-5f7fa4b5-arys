from pathlib import Path
import pandas as pd, streamlit as st, networkx as nx, plotly.graph_objects as go
st.set_page_config(page_title='AML Graph',layout='wide'); st.title('AML Graph — финансовая структура сети')
out=Path('outputs'); data=Path('data')
if not (out/'node_features.csv').exists(): st.error('Сначала выполните: python pipeline.py'); st.stop()
f=pd.read_csv(out/'node_features.csv'); top=pd.read_csv(out/'top_nodes.csv'); clusters=pd.read_csv(out/'clusters.csv'); edges=pd.read_csv(data/'edges.csv') if (data/'edges.csv').exists() else pd.read_parquet(data/'edges.parquet')
t1,t2,t3,t4=st.tabs(['Overview','TOP priorities','Node explorer','Clusters'])
with t1:
    a,b,c,d=st.columns(4); a.metric('Nodes',len(f)); b.metric('Edges',len(edges)); c.metric('Seeds',int(f.is_seed.sum())); d.metric('Clusters',f.cluster_id.nunique())
    st.bar_chart(f.role.value_counts())
with t2:
    st.dataframe(top,use_container_width=True,hide_index=True)
with t3:
    gid=st.selectbox('GID',f.gid.astype(str).tolist())
    g=int(gid); r=f[f.gid==g].iloc[0]; st.subheader(f'{g} — {r.role}'); st.write(r.evidence)
    c1,c2,c3,c4=st.columns(4); c1.metric('Priority',f'{r.priority_score:.3f}'); c2.metric('IN degree',int(r.in_degree)); c3.metric('OUT degree',int(r.out_degree)); c4.metric('Seed coverage',int(r.seed_coverage))
    ee=edges[(edges.src==g)|(edges.dst==g)].sort_values('sum_kzt',ascending=False); st.dataframe(ee,use_container_width=True,hide_index=True)
    nodes=set([g])|set(ee.src.astype(int))|set(ee.dst.astype(int)); H=nx.DiGraph(); H.add_nodes_from(nodes)
    for x in ee.itertuples(index=False): H.add_edge(int(x.src),int(x.dst),weight=float(x.sum_kzt))
    pos=nx.spring_layout(H,seed=42); ex=[]; ey=[]
    for u,v in H.edges(): ex += [pos[u][0],pos[v][0],None]; ey += [pos[u][1],pos[v][1],None]
    fig=go.Figure(); fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',hoverinfo='none'))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in H],y=[pos[n][1] for n in H],mode='markers+text',text=[str(n) for n in H],textposition='top center',hovertext=[str(n) for n in H]))
    fig.update_layout(showlegend=False,height=650); st.plotly_chart(fig,use_container_width=True)
with t4: st.dataframe(clusters.sort_values(['n_seed','n_nodes'],ascending=False),use_container_width=True,hide_index=True)
