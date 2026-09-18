export function scaleX(value:number, min:number, max:number, left:number, right:number) {
  return max>min ? left+(value-min)/(max-min)*(right-left) : (left+right)/2
}
export function niceTicks(min:number,max:number,count=4) {
  const range=max-min || Math.max(Math.abs(min)*.01,.0001)
  const rough=range/count, magnitude=10**Math.floor(Math.log10(rough))
  const step=([1,2,2.5,5,10].find(n=>n*magnitude>=rough)??10)*magnitude
  const lo=Math.floor(min/step)*step, hi=Math.ceil(max/step)*step
  return Array.from({length:Math.round((hi-lo)/step)+1},(_,i)=>Number((lo+i*step).toPrecision(12)))
}
