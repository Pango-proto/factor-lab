import test from 'node:test'
import assert from 'node:assert/strict'
import { scaleX, niceTicks } from '../frontend/components/backtest/chartGeometry.ts'

test('fractional return domain spans the full plot, not one unit',()=>{
 assert.equal(scaleX(-.06,-.06,.06,68,874),68)
 assert.equal(scaleX(.06,-.06,.06,68,874),874)
 assert.equal(scaleX(0,-.06,.06,68,874),471)
})
test('t0 and D1 are different coordinates at both zoom levels',()=>{
 for(const end of [10,252]){
  assert.equal(scaleX(0,0,end,68,874),68)
  assert.ok(scaleX(1,0,end,68,874)>68)
 }
 assert.ok(scaleX(1,0,10,68,874)-68>80)
})
test('nice percent ticks avoid 3.3 percent and singleton scales stay finite',()=>{
 assert.deepEqual(niceTicks(-.05,.05),[-.05,-.025,0,.025,.05])
 assert.equal(scaleX(10,10,10,68,874),471)
})
