import { useEffect, useState } from 'react'
import { PreviewChart, type PlotPoint, type PlotSeries } from './PreviewChart'
import { niceTicks } from './chartGeometry'

type Marker={scenario:string;name:string;value:number;percentile:number}
type Distribution={points:PlotPoint[];counts:number[];n:number;markers:Marker[]}
type Fan={day:number;q05:number;q25:number;q50:number;q75:number;q95:number}
type Ensemble={schema_version:number;real_market_data:boolean;research_status:string;contract:{paths:number;seed:number;sessions:number};invariants:{status:string;audited_sessions:number;strategy_paths:number};series:{id:string;name:string;color:string;fan:Fan[];distributions:Record<string,Distribution>}[]}
type Risk={real_market_data:boolean;research_status:string;versions:{id:string;nav:PlotPoint[];invariants:{status:string};summary:{total_return:number;max_drawdown:number;total_fees:number;fills:number;mean_predicted_daily_volatility:number;turnover:number;mean_concentration:number}}[];real_readiness:{status:string;latest_dates:Record<string,string>;holdout_start:string;reasons:string[]};snapshot_example:{X:{asset_ids:string[];factor_ids:string[];X:number[][];risk_basis_id:string};F_Delta:{quality:{min_eigenvalue:number;minimum_specific_variance:number};coverage:{observations:number};horizon_sessions:number;calibration_status:string}};calibration_acceptance:{known_unit_std:{status:string};known_double_std:{status:string}}}
const percent=(n:number)=>`${(n*100).toFixed(2)}%`
const colors=['#346779','#bd8396','#8a7199']
const names:Record<string,string>={baseline:'相同信号 · 无风险限制',exposure:'暴露限制',forecast:'暴露 + 方差限制'}
const reasonLabels:Record<string,string>={RISK_SET_CANDIDATE_NOT_FROZEN:'现有风险集仍为 candidate',CURRENT_BASIS_CALIBRATION_PENDING:'当前基底的真实风险校准证据尚未完成',RISK_SNAPSHOTS_BEHIND_MARKET_DATA:'风险历史仅到 2026-08-14，落后于市场数据',HISTORICAL_AVAILABLE_AT_EVIDENCE_REQUIRED:'历史表缺少逐条可得时间证据'}

function useArtifact<T>(url:string){
 const [data,setData]=useState<T|null>(null),[error,setError]=useState('')
 useEffect(()=>{const c=new AbortController();fetch(url,{signal:c.signal}).then(r=>{if(!r.ok)throw new Error(`加载失败 ${r.status}`);return r.json()}).then(d=>{if(d.real_market_data!==false||d.research_status!=='not_eligible')throw new Error('展示数据用途标识不匹配');setData(d)}).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>c.abort()},[url])
 return {data,error}
}

export function RiskEnsemblePanels({strategyId,reference,strategies,onStrategyChange}:{strategyId:string;reference:{name:string;color:string;rows:{day:number;wealth:number}[]};strategies:{id:string;name:string}[];onStrategyChange:(id:string)=>void}){
 const ensemble=useArtifact<Ensemble>('/demo-backtest/ensemble-v1/ensemble.json')
 const risk=useArtifact<Risk>('/demo-backtest/m3-v1/risk.json')
 const [metric,setMetric]=useState('max_drawdown'),[day,setDay]=useState(252)
 const item=ensemble.data?.series.find(s=>s.id===strategyId),dist=item?.distributions[metric]
 const format=metric==='sharpe'?(n:number)=>n.toFixed(2):percent
 const fan:PlotSeries[]=item?[
  {id:'q05-95',name:'5–95% 分位带',color:'#d7e4e8',stack:true,points:item.fan.map(r=>({x:r.day,y:r.q95,y0:r.q05}))},
  {id:'q25-75',name:'25–75% 分位带',color:'#abc5cf',stack:true,points:item.fan.map(r=>({x:r.day,y:r.q75,y0:r.q25}))},
  {id:'median',name:'逐日中位数',color:item.color,points:item.fan.map(r=>({x:r.day,y:r.q50}))},
  {id:'scenario',name:`当前情景 · ${reference.name}`,color:'#ac7a55',dash:true,points:reference.rows.map(r=>({x:r.day,y:r.wealth}))}
 ]:[]
 const support=dist?[...dist.points.map(p=>p.x),...dist.markers.map(m=>m.value)]:[]
 const distributionTicks=dist?niceTicks(Math.min(...support),Math.max(...support),4):[]
 return <>
 <div className="bp-section-heading" id="bp-ensemble"><span>04 / SYNTHETIC PATH ENSEMBLE</span><h2>模拟路径分布</h2><p>每个模拟行情分别运行三个策略与买入持有基准。策略选择与上方“成交与仓位”同步。</p></div>
 {ensemble.error?<p className="bp-note" role="alert">模拟分布：{ensemble.error}</p>:!item||!dist?<p className="bp-note">正在读取完整模拟产物…</p>:<>
 <section className="bp-surface"><div className="bp-controls"><label>策略 <select aria-label="选择模拟分布策略" value={strategyId} onChange={e=>onStrategyChange(e.target.value)}>{strategies.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select></label></div><div className="bp-chart-layout"><div><PreviewChart title={`${item.name} · 模拟净值扇形图`} series={fan} xTicks={[0,50,100,150,200,252]} selected={day} onSelect={setDay} height={320} hideReadout/><div className="bp-legend"><span><i style={{background:'#d7e4e8'}}/>5–95%</span><span><i style={{background:'#abc5cf'}}/>25–75%</span><span><i style={{background:item.color}}/>逐日中位数</span><span><i style={{background:'#ac7a55'}}/>当前手工情景</span></div><p className="bp-note">D{day}：5% {item.fan[day].q05.toFixed(3)} · 中位数 {item.fan[day].q50.toFixed(3)} · 95% {item.fan[day].q95.toFixed(3)}</p></div><aside className="bp-explainer"><h3>{item.name}</h3><dl><dt>市场路径</dt><dd>{ensemble.data!.contract.paths.toLocaleString()} × 252 日</dd><dt>执行账本</dt><dd>{ensemble.data!.invariants.strategy_paths.toLocaleString()} 条</dd><dt>逐日不变量</dt><dd>{ensemble.data!.invariants.audited_sessions.toLocaleString()} 日通过</dd><dt>固定主种子</dt><dd>{ensemble.data!.contract.seed}</dd></dl><p>每条路径使用独立确定性随机流，保留所有结果；费用、整手、成交和未成交均重新计算。</p><small>分位带按日截面计算，不是一条可实现路径，也不是实际收益的置信区间。概率仅相对于当前合成模型。</small></aside></div></section>
 <section className="bp-surface"><div className="bp-controls"><label>路径统计量 <select aria-label="选择路径分布统计量" value={metric} onChange={e=>setMetric(e.target.value)}><option value="max_drawdown">最大回撤</option><option value="total_return">累计收益</option><option value="sharpe">Sharpe</option></select></label><span>{item.name} · 全部 {dist.n} 条路径</span></div><div className="bp-chart-layout"><PreviewChart title={`${item.name} · ${metric} 分布密度`} series={[{id:'distribution',name:item.name,color:item.color,points:dist.points}]} xDomain={[distributionTicks[0],distributionTicks[distributionTicks.length-1]]} xTicks={distributionTicks} xLabel={format} format={n=>n.toFixed(1)} zero references={dist.markers.map((m,i)=>({x:m.value,label:String(i+1),color:colors[i]}))}/><aside className="bp-explainer"><h3>固定情景所在分位</h3><table className="bp-audit-table"><thead><tr><th>情景</th><th>统计量</th><th>分位</th></tr></thead><tbody>{dist.markers.map(m=><tr key={m.scenario}><th>{m.name}</th><td>{format(m.value)}</td><td>{m.percentile.toFixed(1)}%</td></tr>)}</tbody></table><p>分位 = 小于该值的路径数，加相等路径数的一半，再除以全部路径数。回撤分位越高，回撤越严重。</p><small>三个手工情景是压力参照。0% 或 100% 表示超出本次经验样本范围，不等于不可能或必然发生。</small></aside></div>
 <div className="bp-scenario-markers">{dist.markers.map((m,i)=><div key={m.scenario}><span style={{color:colors[i]}}>{i+1}. {m.name}</span><meter min={0} max={100} value={m.percentile} aria-label={`${m.name} 分位 ${m.percentile.toFixed(1)}%`}/><b>{m.percentile.toFixed(1)}%</b></div>)}</div>
 <p className="bp-note">直方密度覆盖全部样本；MDD 含 t0。Sharpe 使用样本日收益标准差、252 日年化、零基准收益。该层为合成行情完整重跑，未执行真实行情 block bootstrap。</p><a href="/demo-backtest/ensemble-v1/_MANIFEST.json" target="_blank" rel="noreferrer">模拟分布 manifest</a></section>
 </>}
 <div className="bp-section-heading" id="bp-risk"><span>05 / POINT-IN-TIME RISK</span><h2>时点风险接入</h2><p>X 为风险暴露，F 为因子协方差，Delta 为特异方差。工程验收与真实风险准入分别记录。</p></div>
 {risk.error?<p className="bp-note" role="alert">风险验收：{risk.error}</p>:!risk.data?<p className="bp-note">正在读取风险验收产物…</p>:<>
 <section className="bp-surface bp-chart-layout"><div><PreviewChart title="同一信号、行情与成本下的风险版本对照" xTicks={[0,21,42,63,84]} series={risk.data.versions.map((v,i)=>({id:v.id,name:names[v.id],color:colors[i],points:v.nav,dash:v.id==='forecast'}))}/><div className="bp-legend">{risk.data.versions.map((v,i)=><span key={v.id}><i style={{background:colors[i]}}/>{names[v.id]}</span>)}</div></div><aside className="bp-explainer"><h3>合成账本验收通过</h3><p>84 个交易日，80 日合成历史预热。相同滞后 5 日排名信号，每 21 日从 80% 至 0% 的离散仓位依次检查。</p><p>行业与规模暴露限制先约束候选目标；风险版本再检查一日方差上限。现金留存、订单和实际费用均由执行账本生成。</p><small>本样本额外方差限制未触发，后两条曲线重合。不能据此判定额外风险限制无效。</small></aside></section>
 <section className="bp-surface"><div style={{overflowX:'auto'}}><table className="bp-audit-table"><caption>风险版本对照 · 合成工程样本</caption><thead><tr><th>版本</th><th>累计收益</th><th>最大回撤</th><th>收盘持仓日波动估计</th><th>单边换手</th><th>费用 / 元</th><th>成交</th></tr></thead><tbody>{risk.data.versions.map(v=><tr key={v.id}><th>{names[v.id]}</th><td>{percent(v.summary.total_return)}</td><td>{percent(v.summary.max_drawdown)}</td><td>{percent(v.summary.mean_predicted_daily_volatility)}</td><td>{percent(v.summary.turnover)}</td><td>{v.summary.total_fees.toFixed(2)}</td><td>{v.summary.fills}</td></tr>)}</tbody></table></div><p className="bp-note">行业权重上限 60%，规模绝对暴露上限 0.4；方差上限 0.000025 / 日，均仅用于此工程 fixture。买单只预留决策时已有现金，不预支次日卖出款。风险栏使用前一日快照评估收盘实际持仓，是持仓诊断，不是该日收益的前瞻校准结果。未把预测成本再次从净值扣除。</p></section>
 <section className="bp-surface bp-chart-layout"><div><h3>真实风险数据：待补证据</h3><ul>{risk.data.real_readiness.reasons.map(r=><li key={r}>{reasonLabels[r]??r}</li>)}</ul><p className="bp-note">真实风险拟合与校准未执行；Holdout 自 {risk.data.real_readiness.holdout_start} 起保持封存。此页合成验收不会放行真实风险策略。</p><a href="/demo-backtest/m3-v1/real-readiness.json" target="_blank" rel="noreferrer">查看真实数据接入诊断</a></div><aside className="bp-explainer"><h3>独立校准检查</h3><p>已验证标准化收益标准差为 1 时通过、为 2 时失败；并拒绝未来信息、期限错配、缺列、非有限值和非正特异方差。</p><small>这里的已知答案测试仅验证实现。真实风险是否校准，仍需独立样本及分层检验。</small></aside></section>
 <details className="bp-method"><summary>查看风险暴露快照与数值检查</summary><div style={{overflowX:'auto'}}><table className="bp-audit-table"><thead><tr><th>因子</th>{risk.data.snapshot_example.X.asset_ids.map(a=><th key={a}>{a}</th>)}</tr></thead><tbody>{risk.data.snapshot_example.X.factor_ids.map((f,j)=><tr key={f}><th>{f}</th>{risk.data!.snapshot_example.X.X.map((row,i)=><td key={i}>{row[j].toFixed(2)}</td>)}</tr>)}</tbody></table></div><p>此合成矩阵只检验行业与六个风格列的接口，不是新采用的正式 Barra 因子清单。真实 candidate 基底保持原样。</p><p>F 对称且半正定；Delta 严格为正。样例最小特异方差 {risk.data.snapshot_example.F_Delta.quality.minimum_specific_variance.toExponential(4)}；使用 {risk.data.snapshot_example.F_Delta.coverage.observations} 条已可得历史观测；预测期限 1 个交易日，不自动换算为其他期限。</p><a href="/demo-backtest/m3-v1/_MANIFEST.json" target="_blank" rel="noreferrer">风险验收 manifest</a></details>
 </>}
 </>
}
