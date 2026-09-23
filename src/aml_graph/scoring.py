import numpy as np
ROLES=['consolidator','transit','distributor','terminal','coordinator','peripheral']

def score_roles(f):
    f=f.copy(); f['consolidator_score']=.40*f.in_p+.20*f.seed_p+.18*f.ain_p+.12*f.retention_obs+.10*(1-f.out_p)
    f['distributor_score']=.46*f.out_p+.20*f.aout_p+.14*f.seed_p+.12*f.bt_p+.08*(1-f.in_p)
    f['transit_score']=.30*f.flow_closeness+.24*f.same_day_ratio.clip(0,1)+.20*f.bt_p+.14*f.degree_balance+.12*np.minimum(f.in_p,f.out_p)
    f['coordinator_score']=.34*f.bt_p+.20*f.seed_p+.14*f.pr_p+.14*f.degree_p+.10*f.bridge_p+.08*f.cycle_member.astype(float)
    term=(f.depth<4)&(~f.is_seed)&(f.amount_in>0)
    f['terminal_score']=term.astype(float)*(.42*(f.out_degree==0)+.28*f.ain_p+.18*f.in_p+.12*(1-f.out_p))
    act=np.maximum.reduce([f.in_p.values,f.out_p.values,f.bt_p.values,f.pr_p.values,f.seed_p.values]); f['peripheral_score']=np.clip(1-act,0,1)*.9+.1*((f.in_degree+f.out_degree)<=1)
    sc=f[[x+'_score' for x in ROLES]].copy(); sc.loc[f.in_degree<2,'consolidator_score']=0; sc.loc[f.out_degree<3,'distributor_score']=0
    sc.loc[~((f.in_degree>0)&(f.out_degree>0)&(f.flow_closeness>=.5)),'transit_score']=0; sc.loc[~(term&(f.out_degree<=1)),'terminal_score']=0
    ce=((f.bt_p>=.90)&(f.degree_p>=.90))|((f.seed_coverage>=7)&(f.bt_p>=.80)&(f.degree_p>=.80)); sc.loc[~ce,'coordinator_score']=0
    m=sc.to_numpy(); w=m.argmax(axis=1); f['role']=[ROLES[i] for i in w]; f['role_score']=m.max(axis=1).clip(0,1); return f

def score_priority(f,w,seed_mult=.55,isolated_mult=.10):
    f=f.copy(); f['priority_score']=w['seed']*f.seed_p+w['betweenness']*f.bt_p+w['degree']*f.degree_p+w['volume']*f.volume_p+w['pagerank']*f.pr_p+w['bridge']*f.bridge_p+w['role']*f.role_score
    f.loc[f.is_seed,'priority_score']*=seed_mult; f.loc[(f.in_degree+f.out_degree)==0,'priority_score']*=isolated_mult; f['priority_score']=f.priority_score.clip(0,1); return f

def add_evidence(f):
    def money(x): return f'{x/1e6:.2f}m KZT' if x>=1e6 else (f'{x/1e3:.0f}k KZT' if x>=1e3 else f'{x:.0f} KZT')
    def ev(r):
        if r.role=='consolidator': s=f'Признаки консолидации: {r.in_degree} плательщиков, IN {money(r.amount_in)}, охват {r.seed_coverage} seed; набл. OUT/IN {r.flow_ratio:.2f}.'
        elif r.role=='distributor': s=f'Признаки распределения: {r.out_degree} получателей, OUT {money(r.amount_out)}, охват {r.seed_coverage} seed; fan-out высокий.'
        elif r.role=='transit': s=f'Признаки транзита: OUT/IN {r.flow_ratio:.2f}, same-day match {r.same_day_ratio:.0%}, {r.in_degree}→{r.out_degree} контрагентов.'
        elif r.role=='terminal': s=f'Признаки конечного узла: depth={r.depth}, IN {money(r.amount_in)}, исходящих контрагентов {r.out_degree}; не boundary depth=4.'
        elif r.role=='coordinator': s=f'Кандидат в координаторы: {r.in_degree}→{r.out_degree} связей, охват {r.seed_coverage} seed, betweenness {r.betweenness:.4f}; межкластерных связей {r.cross_cluster_degree}.'
        else: s=f'Периферия: {r.in_degree} входящих и {r.out_degree} исходящих связей; структурная значимость ниже порогов других ролей.'
        return s[:200]
    f=f.copy(); f['evidence']=f.apply(ev,axis=1); return f
