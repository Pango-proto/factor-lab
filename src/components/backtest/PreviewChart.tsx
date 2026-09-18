import { useEffect, useId, useRef, useState } from 'react'
import { niceTicks, scaleX } from './chartGeometry'
export type PlotPoint = { x:number;y:number;y0?:number;value?:number }
export type PlotSeries = {id:string;name:string;color:string;points:PlotPoint[];step?:boolean;dash?:boolean;stack?:boolean}
type Marker = {x:number;y:number;side:string;quantity:number;unfilled?:boolean;reason?:string}
type Props = {
 series:PlotSeries[];title:string;format?:(n:number)=>string;height?:number;xLabel?:(n:number)=>string;
 selected?:number;onSelect?:(n:number)=>void;markers?:Marker[];area?:boolean;bars?:boolean;
 fixedDomain?:[number,number];xDomain?:[number,number];xTicks?:number[];zero?:boolean;hideReadout?:boolean;
 horizontalBand?:[number,number];spans?:{from:number;to:number;label:string}[];
 references?:{x:number;label:string;color:string}[];
}
export function PreviewChart({series,title,format=n=>n.toFixed(2),height=280,
 xLabel=n=>n===0?'t0':`D${Math.round(n).toString().padStart(3,'0')}`,selected,onSelect,markers=[],
 area,bars,fixedDomain,xDomain,xTicks,zero,hideReadout,horizontalBand,spans=[],references=[]}:Props){
 const uid=useId().replace(/:/g,''), container=useRef<HTMLDivElement>(null)
 const [width,setWidth]=useState(900)
 useEffect(()=>{if(!container.current)return;const o=new ResizeObserver(([e])=>setWidth(Math.max(320,e.contentRect.width)));o.observe(container.current);return()=>o.disconnect()},[])
 const left=68,right=26,top=34,bottom=36
 if(width<600)height=Math.min(height,230)
 const all=series.flatMap(s=>s.points)
 if(!all.length)return <div className="bp-chart-empty">没有可展示的数据</div>
 const xmin=xDomain?.[0]??Math.min(...all.map(p=>p.x)),xmax=xDomain?.[1]??Math.max(...all.map(p=>p.x))
 let ymin=fixedDomain?.[0]??Math.min(...all.map(p=>p.y0??p.y),...markers.map(p=>p.y),...(horizontalBand??[]))
 let ymax=fixedDomain?.[1]??Math.max(...all.map(p=>p.y),...markers.map(p=>p.y),...(horizontalBand??[]))
 if(zero){ymin=Math.min(0,ymin);ymax=Math.max(0,ymax)}
 const nonnegative=ymin>=0,nonpositive=ymax<=0
 if(!fixedDomain){const pad=Math.max((ymax-ymin)*.12,Math.abs(ymax)*.005,.0001);ymin-=pad;ymax+=pad}
 let ticks=niceTicks(ymin,ymax,fixedDomain?6:4)
 if(zero&&nonnegative)ticks=ticks.filter(n=>n>=0)
 if(zero&&nonpositive)ticks=ticks.filter(n=>n<=0)
 ymin=ticks[0];ymax=ticks[ticks.length-1]
 const sx=(x:number)=>scaleX(x,xmin,xmax,left,width-right)
 const sy=(y:number)=>top+(ymax-y)/(ymax-ymin||1)*(height-top-bottom)
 const path=(p:PlotPoint[],step=false)=>p.map((v,i)=>`${i?(step?`H${sx(v.x).toFixed(2)} V`:'L'):'M'}${i&&step?'':`${sx(v.x).toFixed(2)},`}${sy(v.y).toFixed(2)}`).join(' ')
 const xt=xTicks??Array.from({length:6},(_,i)=>xmin+(xmax-xmin)*i/5)
 const selectedValues=series.map(s=>({...s,point:s.points.find(p=>p.x===selected)}))
 return <div className="bp-chart" ref={container}><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}
 onPointerMove={onSelect?e=>{const r=e.currentTarget.getBoundingClientRect();const x=(e.clientX-r.left)/r.width*width;onSelect(Math.max(xmin,Math.min(xmax,Math.round(xmin+(x-left)/(width-left-right)*(xmax-xmin)))))}:undefined}>
 <title>{title}</title><defs><clipPath id={uid}><rect x={left-7} y={top-8} width={width-left-right+14} height={height-top-bottom+16}/></clipPath></defs>
 {ticks.map(v=><g key={v}><line x1={left} x2={width-right} y1={sy(v)} y2={sy(v)} stroke="#e9edef"/><text x={left-10} y={sy(v)+4} textAnchor="end">{format(v)}</text></g>)}
 {xt.filter(x=>x>=xmin&&x<=xmax).map((x,i)=><text data-axis="x" data-value={x} key={i} x={sx(x)} y={height-10} textAnchor="middle">{xLabel(x)}</text>)}
 <g clipPath={`url(#${uid})`}>
 {horizontalBand&&<rect x={left} width={width-left-right} y={sy(horizontalBand[1])} height={sy(horizontalBand[0])-sy(horizontalBand[1])} fill="#9ab4bf" opacity=".15"/>}
 {horizontalBand?.map(v=><line key={v} x1={left} x2={width-right} y1={sy(v)} y2={sy(v)} stroke="#93aab4" strokeDasharray="4 4"/>)}
 {spans.filter(s=>s.to>=xmin&&s.from<=xmax).map(s=><g key={s.label}><rect x={sx(Math.max(xmin,s.from))} width={sx(Math.min(xmax,s.to))-sx(Math.max(xmin,s.from))} y={top} height={height-top-bottom} fill="#dec390" opacity=".17"/><text x={sx(Math.max(xmin,s.from))+5} y={top+13}>{s.label}</text></g>)}
 {series.map(s=><g key={s.id}>
 {s.stack?<path d={`${path(s.points)} ${s.points.slice().reverse().map(p=>`L${sx(p.x)},${sy(p.y0??0)}`).join(' ')} Z`} fill={s.color} opacity=".72"/>:<>
 {area&&<path d={`${path(s.points)} L${sx(xmax)},${sy(0)} L${sx(xmin)},${sy(0)} Z`} fill={s.color} opacity=".09"/>}
 {bars?s.points.map((p,i)=><line key={i} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y0??0)} y2={sy(p.y)} stroke={s.color} strokeWidth="2"/>):<path data-series={s.id} d={path(s.points,s.step)} fill="none" stroke={s.color} strokeWidth="1.8" strokeDasharray={s.dash?'5 4':undefined} strokeLinejoin="round"/>}</>}
 </g>)}
 {references.map((r,i)=><g key={r.label}><line x1={sx(r.x)} x2={sx(r.x)} y1={top+18} y2={height-bottom} stroke={r.color} strokeDasharray="3 4"/><text x={sx(r.x)+(i%2?-5:5)} y={top+12} textAnchor={i%2?'end':'start'}>{r.label}</text></g>)}
 {markers.map((m,i)=><path key={i} data-day={m.x} data-marker={m.unfilled?'unfilled':'filled'} transform={`translate(${sx(m.x)},${sy(m.y)})`} d={m.side==='buy'?'M0,-7 L6,5 L-6,5 Z':'M0,7 L6,-5 L-6,-5 Z'} fill={m.unfilled?'white':m.side==='buy'?'#428b7d':'#c86f83'} stroke={m.unfilled?'#846840':'white'} strokeWidth={m.unfilled?1.7:1.2}><title>{xLabel(m.x)} {m.unfilled?'未成交 ':''}{m.side==='buy'?'买入':'卖出'} {m.quantity} 股 · {m.y.toFixed(2)} 元 {m.reason??''}</title></path>)}
 {selected!==undefined&&selected>=xmin&&selected<=xmax&&<g><line x1={sx(selected)} x2={sx(selected)} y1={top} y2={height-bottom} stroke="#8ba1aa" strokeDasharray="3 4"/>{selectedValues.map(s=>s.point&&<circle key={s.id} cx={sx(s.point.x)} cy={sy(s.point.y)} r="3" fill={s.color} stroke="white"/>)}</g>}
 </g></svg>
 {onSelect&&!hideReadout&&<div className="bp-chart-readout"><span>{selected===undefined?'悬停查看':xLabel(selected)}</span>{selectedValues.map(s=>s.point&&<span key={s.id}><i style={{background:s.color}}/>{s.name} <b>{format(s.point.value??s.point.y)}</b></span>)}</div>}
 </div>
}
