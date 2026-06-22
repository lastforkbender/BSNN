import { useState, useEffect, useRef, useCallback } from "react";

// ── TRON DESIGN TOKENS ────────────────────────────────────────────────────────
const C = {
  bg:      "#020810",
  panel:   "#040d1a",
  grid:    "#071428",
  cyan:    "#00e5ff",
  blue:    "#0066ff",
  purple:  "#7b00ff",
  gold:    "#ffcc00",
  red:     "#ff2244",
  dim:     "#0a2a3a",
  muted:   "#1a4060",
  text:    "#b0e0f0",
  bright:  "#e0f8ff",
};

// ── COMPLEX MATH UTILITIES ────────────────────────────────────────────────────
const cMul = (a, b) => ({
  re: a.re*b.re - a.im*b.im,
  im: a.re*b.im + a.im*b.re
});
const cAbs = (a) => Math.sqrt(a.re**2 + a.im**2);
const cPhase = (a) => Math.atan2(a.im, a.re);
const randC = () => ({ re: (Math.random()-0.5)*2, im: (Math.random()-0.5)*2 });

// ── B-SPLINE EVALUATOR (de Boor) ──────────────────────────────────────────────
function deBoor(k, degree, t, knots, ctrl) {
  const d = ctrl.slice(k-degree, k+1);
  for (let r=1; r<=degree; r++) {
    for (let j=degree; j>=r; j--) {
      const left=j+k-degree, right=j+1+k-degree;
      const denom = knots[right]-knots[left];
      const alpha = denom===0 ? 0 : (t-knots[left])/denom;
      d[j] = { re: (1-alpha)*d[j-1].re + alpha*d[j].re,
               im: (1-alpha)*d[j-1].im + alpha*d[j].im };
    }
  }
  return d[degree];
}
function evalBSpline(ctrl, nPts=60, degree=3) {
  const n=ctrl.length, m=n+degree+1;
  const knots=Array.from({length:m},(_,i)=>
    i<=degree ? 0 : i>=m-degree-1 ? 1 : (i-degree)/(n-degree));
  const pts=[];
  for(let s=0;s<nPts;s++){
    const t=s/(nPts-1)*(1-1e-9);
    let k=degree;
    while(k<n-1 && knots[k+1]<=t) k++;
    pts.push(deBoor(k,degree,t,knots,ctrl));
  }
  return pts;
}

// ── BLOCH SPHERE STATE ────────────────────────────────────────────────────────
const blochXYZ = (theta,phi,r=1) => ({
  x: r*Math.sin(theta)*Math.cos(phi),
  y: r*Math.sin(theta)*Math.sin(phi),
  z: r*Math.cos(theta)
});

// ── SIMULATED BSNN ENGINE ─────────────────────────────────────────────────────
function initBSNN(nLayers=3, mcPerLayer=4) {
  const makeMC = (layer,id) => ({
    mc_id: id, generation:0, layer, score:Math.random()*0.3,
    bloch_theta: Math.random()*Math.PI,
    bloch_phi:   Math.random()*2*Math.PI,
    spline_count: Math.max(2,4-layer),
    agent_pairs: [[0,1],[2,3]],
    loss_history: [], improvement_rate:0,
    is_subconscious:false,
    nodes: Array.from({length:4},()=>({
      id:Math.random().toString(36).slice(2,6),
      ctrl: Array.from({length:6},randC),
      angle: Math.random()*Math.PI*2
    }))
  });
  return {
    generation:0, global_best:0,
    layers: Array.from({length:nLayers},(_,l)=>
      Array.from({length:mcPerLayer},(_,i)=>makeMC(l,`L${l}M${i}`))),
    apex:null,
    kan:{ stored:0, mean_score:0, max_score:0, top_concepts:[] },
    manifold: Array.from({length:32},()=>({
      theta:Math.random()*Math.PI,
      phi:Math.random()*2*Math.PI,
      r:0.8+Math.random()*0.2
    })),
    subconscious_spawned:false
  };
}

function evolveStep(state) {
  const s = JSON.parse(JSON.stringify(state));
  s.generation++;
  let globalBest = s.global_best;

  s.layers = s.layers.map(layer => layer.map(mc => {
    // self-improve: gradient on theta/phi + loss
    const dtheta = (Math.random()-0.5)*0.1;
    const dphi   = (Math.random()-0.5)*0.1;
    mc.bloch_theta = Math.max(0,Math.min(Math.PI, mc.bloch_theta+dtheta));
    mc.bloch_phi   = (mc.bloch_phi + dphi + 2*Math.PI) % (2*Math.PI);
    const loss = Math.max(0, (mc.loss_history[mc.loss_history.length-1]||1)*
                 (0.85+Math.random()*0.2));
    const coupled = loss * (1-0.3) - 0.3 * Math.sin(mc.bloch_theta)*0.5;
    const score = 1/(1+Math.abs(coupled));
    mc.improvement_rate = score - mc.score;
    mc.score = score;
    mc.loss_history = [...mc.loss_history.slice(-9), parseFloat(coupled.toFixed(4))];
    mc.generation = s.generation;
    // rotate spline nodes
    mc.nodes = mc.nodes.map(n=>({...n, angle:(n.angle+0.05*(1-score))%(Math.PI*2),
      ctrl: n.ctrl.map(c=>cMul(c,{re:Math.cos(0.05),im:Math.sin(0.05)*0.1}))
    }));
    if(score > globalBest) globalBest=score;
    return mc;
  }));

  // evolve manifold
  s.manifold = s.manifold.map(m=>({
    theta:(m.theta+0.01*(Math.random()-0.5)+Math.PI)%Math.PI,
    phi:  (m.phi+0.02*(Math.random()-0.5)+2*Math.PI)%(2*Math.PI),
    r:    Math.max(0.6,Math.min(1,m.r+(Math.random()-0.5)*0.02))
  }));

  // merge apex
  const allMCs = s.layers.flat().sort((a,b)=>b.score-a.score);
  s.apex = {
    mc_id:"APEX",
    score: parseFloat(((allMCs[0]?.score||0)*0.9+globalBest*0.1).toFixed(4)),
    parents:[allMCs[0]?.mc_id,allMCs[1]?.mc_id]
  };

  // subconscious spawn check
  const meanScore = allMCs.reduce((a,m)=>a+m.score,0)/allMCs.length;
  if(meanScore>0.3 && Math.random()>0.7){
    const newMC={
      mc_id:`SC_${s.generation}`, generation:s.generation, layer:0,
      score:meanScore*0.8, bloch_theta:Math.random()*Math.PI,
      bloch_phi:Math.random()*2*Math.PI, spline_count:2,
      agent_pairs:[[0,1]], loss_history:[], improvement_rate:0,
      is_subconscious:true,
      nodes:Array.from({length:2},()=>({
        id:Math.random().toString(36).slice(2,6),
        ctrl:Array.from({length:4},randC), angle:Math.random()*Math.PI*2
      }))
    };
    s.layers[0].push(newMC);
    s.subconscious_spawned=true;
  } else s.subconscious_spawned=false;

  // KAN stats
  s.kan={
    stored: Math.min((s.kan.stored||0)+allMCs.length, 256),
    mean_score: parseFloat(meanScore.toFixed(4)),
    max_score:  parseFloat(globalBest.toFixed(4)),
    top_concepts: allMCs.slice(0,3).map(m=>({idx:0,score:m.score,sim:m.score*0.9}))
  };
  s.global_best = parseFloat(globalBest.toFixed(4));
  return s;
}

// ── CANVAS: BLOCH SPHERE ──────────────────────────────────────────────────────
function BlochCanvas({ manifold, highlight }) {
  const ref = useRef(null);
  useEffect(()=>{
    const cv=ref.current; if(!cv) return;
    const ctx=cv.getContext("2d");
    const W=cv.width, H=cv.height, cx=W/2, cy=H/2, R=Math.min(W,H)*0.38;
    ctx.clearRect(0,0,W,H);
    // outer glow sphere
    const grad=ctx.createRadialGradient(cx,cy,R*0.1,cx,cy,R);
    grad.addColorStop(0,C.cyan+"08"); grad.addColorStop(1,"transparent");
    ctx.fillStyle=grad; ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2); ctx.fill();
    // sphere outline
    ctx.strokeStyle=C.cyan+"40"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2); ctx.stroke();
    // latitude/longitude lines
    for(let i=1;i<4;i++){
      const y=cy+R*(i/4*2-1)*0.85;
      const rr=Math.sqrt(R*R-(y-cy)**2);
      ctx.strokeStyle=C.blue+"25"; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.arc(cx,y,rr,0,Math.PI*2); ctx.stroke();
    }
    for(let i=0;i<6;i++){
      const a=i*Math.PI/3;
      ctx.strokeStyle=C.blue+"20"; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(cx,cy-R); ctx.lineTo(cx,cy+R); ctx.stroke();
    }
    // axes
    const axColor=[[C.cyan,"X"],[C.purple,"Y"],[C.gold,"Z"]];
    [[1,0,0],[0,1,0],[0,0,1]].forEach(([x,y,z],[i])=>{
      const px=cx+x*R; const py=cy-z*R;
      ctx.strokeStyle=axColor[i][0]+"80"; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(px,py); ctx.stroke();
      ctx.fillStyle=axColor[i][0]; ctx.font="10px monospace";
      ctx.fillText(axColor[i][1], px+4, py+4);
    });
    // manifold points
    manifold.forEach((m,idx)=>{
      const {x,y,z}=blochXYZ(m.theta,m.phi,m.r);
      const px=cx+x*R, py=cy-z*R;
      const size = highlight===idx ? 5 : 2.5;
      const col = highlight===idx ? C.gold : C.cyan;
      const grd=ctx.createRadialGradient(px,py,0,px,py,size*2);
      grd.addColorStop(0,col+"dd"); grd.addColorStop(1,"transparent");
      ctx.fillStyle=grd; ctx.beginPath(); ctx.arc(px,py,size*2,0,Math.PI*2); ctx.fill();
      ctx.fillStyle=col; ctx.beginPath(); ctx.arc(px,py,size,0,Math.PI*2); ctx.fill();
    });
    // theta/phi arc
    if(manifold.length>0){
      const m=manifold[0];
      ctx.strokeStyle=C.gold+"80"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.arc(cx,cy,R*0.4,-Math.PI/2,-Math.PI/2+m.theta); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle=C.gold; ctx.font="9px monospace";
      ctx.fillText(`θ=${(m.theta/Math.PI).toFixed(2)}π`,cx+R*0.45+2,cy-R*0.1);
      ctx.fillText(`φ=${(m.phi/Math.PI).toFixed(2)}π`,cx-R*0.6,cy+R*0.5);
    }
  },[manifold,highlight]);
  return <canvas ref={ref} width={220} height={220} style={{display:"block"}} />;
}

// ── CANVAS: SPLINE PATH ───────────────────────────────────────────────────────
function SplineCanvas({ nodes, mcId, score }) {
  const ref = useRef(null);
  useEffect(()=>{
    const cv=ref.current; if(!cv) return;
    const ctx=cv.getContext("2d"); const W=cv.width,H=cv.height;
    ctx.clearRect(0,0,W,H);
    // background grid
    ctx.strokeStyle=C.dim; ctx.lineWidth=0.5;
    for(let x=0;x<W;x+=20){ ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke(); }
    for(let y=0;y<H;y+=20){ ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke(); }
    // spline paths (Tron channels)
    const colors=[C.cyan,C.blue,C.purple,C.gold];
    nodes.forEach((node,ni)=>{
      const col=colors[ni%colors.length];
      const pts=evalBSpline(node.ctrl,50,3);
      if(pts.length<2) return;
      // glow
      ctx.save();
      ctx.shadowColor=col; ctx.shadowBlur=8;
      ctx.strokeStyle=col+"aa"; ctx.lineWidth=1.5;
      ctx.beginPath();
      pts.forEach((p,i)=>{
        const px=(p.re+2)/4*W, py=(p.im+2)/4*H;
        i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
      });
      ctx.stroke(); ctx.restore();
      // control points
      node.ctrl.forEach(c=>{
        const px=(c.re+2)/4*W, py=(c.im+2)/4*H;
        ctx.fillStyle=col+"66"; ctx.beginPath(); ctx.arc(px,py,2,0,Math.PI*2); ctx.fill();
      });
    });
    // score bar
    const sw=W*score;
    ctx.fillStyle=C.blue+"33"; ctx.fillRect(0,H-4,W,4);
    ctx.fillStyle=score>0.7?C.gold:score>0.4?C.cyan:C.purple;
    ctx.fillRect(0,H-4,sw,4);
  },[nodes,score]);
  return <canvas ref={ref} width={160} height={100}
    style={{display:"block",border:`1px solid ${C.muted}`}} />;
}

// ── LOSS SPARKLINE ────────────────────────────────────────────────────────────
function Sparkline({ data }) {
  const ref=useRef(null);
  useEffect(()=>{
    const cv=ref.current; if(!cv||!data.length) return;
    const ctx=cv.getContext("2d"); const W=cv.width,H=cv.height;
    ctx.clearRect(0,0,W,H);
    const mn=Math.min(...data), mx=Math.max(...data), range=mx-mn||1;
    ctx.strokeStyle=C.cyan+"cc"; ctx.lineWidth=1.5;
    ctx.shadowColor=C.cyan; ctx.shadowBlur=4;
    ctx.beginPath();
    data.forEach((v,i)=>{
      const x=i/(data.length-1)*W;
      const y=H-(v-mn)/range*(H-4)-2;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
  },[data]);
  return <canvas ref={ref} width={80} height={30}
    style={{display:"block",verticalAlign:"middle"}} />;
}

// ── META-CONTROLLER CARD ──────────────────────────────────────────────────────
function MCCard({ mc, selected, onClick }) {
  const border = mc.is_subconscious ? C.purple : mc.layer===0 ? C.blue : C.cyan;
  const glow   = mc.score>0.8 ? C.gold : border;
  return (
    <div onClick={onClick} style={{
      background:  C.panel,
      border:      `1px solid ${selected?C.gold:border+"66"}`,
      boxShadow:   selected ? `0 0 12px ${glow}88` : "none",
      borderRadius:4, padding:"8px", cursor:"pointer", marginBottom:6,
      transition:"all 0.2s"
    }}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4}}>
        <span style={{color:C.cyan,fontFamily:"monospace",fontSize:11,fontWeight:"bold"}}>
          {mc.mc_id}
          {mc.is_subconscious && <span style={{color:C.purple,marginLeft:4}}>[SC]</span>}
        </span>
        <span style={{color:mc.score>0.7?C.gold:C.text,fontFamily:"monospace",fontSize:10}}>
          {(mc.score*100).toFixed(1)}%
        </span>
      </div>
      <div style={{display:"flex",gap:8,alignItems:"center"}}>
        <SplineCanvas nodes={mc.nodes} mcId={mc.mc_id} score={mc.score} />
        <div style={{flex:1}}>
          <div style={{color:C.muted,fontSize:9,fontFamily:"monospace",marginBottom:2}}>LOSS</div>
          <Sparkline data={mc.loss_history} />
          <div style={{color:C.text,fontSize:9,fontFamily:"monospace",marginTop:4}}>
            θ={( mc.bloch_theta/Math.PI).toFixed(2)}π φ={(mc.bloch_phi/Math.PI).toFixed(2)}π
          </div>
          <div style={{color:mc.improvement_rate>=0?C.cyan:C.red,fontSize:9,fontFamily:"monospace"}}>
            Δ {mc.improvement_rate>=0?"+":""}{mc.improvement_rate.toFixed(4)}
          </div>
          <div style={{color:C.muted,fontSize:8,fontFamily:"monospace"}}>
            L{mc.layer} G{mc.generation} SA×{mc.agent_pairs.length*2}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── MAIN APP ──────────────────────────────────────────────────────────────────
export default function BSNNApp() {
  const [bsnn, setBSNN]       = useState(()=>initBSNN(3,4));
  const [running, setRunning] = useState(false);
  const [speed, setSpeed]     = useState(1500);
  const [selectedMC, setSel]  = useState(null);
  const [log, setLog]         = useState([]);
  const [blochHL, setBlochHL] = useState(0);
  const [tab, setTab]         = useState("network");   // network|manifold|kan|log
  const timerRef = useRef(null);

  const step = useCallback(()=>{
    setBSNN(prev=>{
      const next=evolveStep(prev);
      setLog(l=>[
        { gen:next.generation, best:next.global_best,
          sub:next.subconscious_spawned,
          apex:next.apex?.score||0,
          kan:next.kan?.stored||0 },
        ...l.slice(0,99)
      ]);
      setBlochHL(h=>(h+1)%next.manifold.length);
      return next;
    });
  },[]);

  useEffect(()=>{
    if(running){ timerRef.current=setInterval(step,speed); }
    else clearInterval(timerRef.current);
    return ()=>clearInterval(timerRef.current);
  },[running,speed,step]);

  const allMCs = bsnn.layers.flat();
  const selMC  = selectedMC!=null ? allMCs.find(m=>m.mc_id===selectedMC) : null;

  const tabBtn = (id,label)=>(
    <button onClick={()=>setTab(id)} style={{
      background: tab===id ? C.blue+"44" : "transparent",
      border:     `1px solid ${tab===id?C.cyan:C.muted}`,
      color:      tab===id ? C.bright : C.text,
      padding:"4px 12px", cursor:"pointer", borderRadius:2,
      fontFamily:"monospace", fontSize:11, transition:"all 0.15s"
    }}>{label}</button>
  );

  return (
    <div style={{ background:C.bg, minHeight:"100vh", color:C.text,
                  fontFamily:"monospace", fontSize:12, overflow:"hidden" }}>

      {/* ── HEADER ── */}
      <div style={{ background:C.panel, borderBottom:`1px solid ${C.cyan}33`,
                    padding:"10px 16px", display:"flex", alignItems:"center", gap:12 }}>
        <div style={{ color:C.cyan, fontSize:16, fontWeight:"bold", letterSpacing:3,
                      textShadow:`0 0 12px ${C.cyan}` }}>
          ⬡ BSNN
        </div>
        <div style={{ color:C.muted, fontSize:10 }}>
          B-SPLINE SELF-IMPROVING NEURAL NETWORK · BLOCH MANIFOLD COGNITIVE SURFACE
        </div>
        <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center"}}>
          <span style={{color:C.gold,fontSize:10}}>BEST {(bsnn.global_best*100).toFixed(1)}%</span>
          <span style={{color:C.muted,fontSize:10}}>G{bsnn.generation}</span>
          <span style={{color:C.cyan,fontSize:10}}>KAN:{bsnn.kan?.stored||0}</span>
        </div>
      </div>

      {/* ── CONTROLS ── */}
      <div style={{ background:C.grid, borderBottom:`1px solid ${C.blue}33`,
                    padding:"6px 16px", display:"flex", gap:8, alignItems:"center" }}>
        <button onClick={()=>setRunning(r=>!r)} style={{
          background: running ? C.red+"33" : C.cyan+"33",
          border:     `1px solid ${running?C.red:C.cyan}`,
          color:      running ? C.red : C.cyan,
          padding:"4px 14px", cursor:"pointer", borderRadius:2,
          fontFamily:"monospace", fontSize:11, letterSpacing:1
        }}>{running?"■ HALT":"▶ EVOLVE"}</button>

        <button onClick={step} style={{
          background:C.blue+"22", border:`1px solid ${C.blue}`,
          color:C.blue, padding:"4px 10px", cursor:"pointer",
          borderRadius:2, fontFamily:"monospace", fontSize:11
        }}>STEP</button>

        <button onClick={()=>{ setBSNN(initBSNN(3,4)); setLog([]); setSel(null); }} style={{
          background:"transparent", border:`1px solid ${C.muted}`,
          color:C.muted, padding:"4px 10px", cursor:"pointer",
          borderRadius:2, fontFamily:"monospace", fontSize:11
        }}>RESET</button>

        <span style={{color:C.muted,fontSize:10,marginLeft:8}}>SPEED</span>
        {[2000,1000,500,200].map(s=>(
          <button key={s} onClick={()=>setSpeed(s)} style={{
            background: speed===s ? C.purple+"44":"transparent",
            border:`1px solid ${speed===s?C.purple:C.muted}`,
            color: speed===s?C.bright:C.muted,
            padding:"3px 8px", cursor:"pointer", borderRadius:2,
            fontFamily:"monospace", fontSize:10
          }}>{1000/s<1?`${s}ms`:`${(1000/s).toFixed(1)}Hz`}</button>
        ))}

        <div style={{marginLeft:"auto",display:"flex",gap:6}}>
          {tabBtn("network","NETWORK")}
          {tabBtn("manifold","BLOCH")}
          {tabBtn("kan","KAN")}
          {tabBtn("log","LOG")}
        </div>
      </div>

      {/* ── MAIN CONTENT ── */}
      <div style={{ display:"flex", height:"calc(100vh - 88px)", overflow:"hidden" }}>

        {/* ── LEFT: LAYER PANEL ── */}
        <div style={{ width:220, background:C.panel, borderRight:`1px solid ${C.blue}22`,
                      overflowY:"auto", padding:"8px", flexShrink:0 }}>
          {bsnn.apex && (
            <div style={{ background:C.gold+"11", border:`1px solid ${C.gold}44`,
                          borderRadius:4, padding:"6px", marginBottom:8 }}>
              <div style={{color:C.gold,fontSize:10,fontWeight:"bold",marginBottom:2}}>
                ⬟ APEX CONTROLLER
              </div>
              <div style={{color:C.bright,fontSize:11}}>
                {(bsnn.apex.score*100).toFixed(1)}% ← {bsnn.apex.parents?.join(" + ")}
              </div>
            </div>
          )}
          {bsnn.subconscious_spawned && (
            <div style={{ background:C.purple+"22", border:`1px solid ${C.purple}66`,
                          borderRadius:4, padding:"4px 6px", marginBottom:8,
                          fontSize:9, color:C.purple }}>
              ✦ SUBCONSCIOUS SPAWN G{bsnn.generation}
            </div>
          )}
          {bsnn.layers.map((layer,li)=>(
            <div key={li}>
              <div style={{ color:C.muted, fontSize:9, letterSpacing:2,
                            marginBottom:4, marginTop:li>0?8:0 }}>
                LAYER {li} · {layer.length} MC
              </div>
              {layer.map(mc=>(
                <MCCard key={mc.mc_id} mc={mc}
                  selected={selectedMC===mc.mc_id}
                  onClick={()=>setSel(s=>s===mc.mc_id?null:mc.mc_id)} />
              ))}
            </div>
          ))}
        </div>

        {/* ── CENTER / RIGHT PANELS ── */}
        <div style={{ flex:1, display:"flex", flexDirection:"column", overflow:"hidden" }}>

          {tab==="network" && (
            <div style={{ flex:1, padding:12, overflowY:"auto" }}>
              {/* Selected MC detail */}
              {selMC ? (
                <div style={{ background:C.panel, border:`1px solid ${C.cyan}44`,
                              borderRadius:4, padding:12, marginBottom:12 }}>
                  <div style={{color:C.cyan,fontSize:13,fontWeight:"bold",marginBottom:8}}>
                    ⬡ {selMC.mc_id}
                    {selMC.is_subconscious&&<span style={{color:C.purple}}> [SUBCONSCIOUS]</span>}
                    <span style={{color:C.muted,fontSize:10,marginLeft:8}}>
                      Layer {selMC.layer} · Gen {selMC.generation}
                    </span>
                  </div>
                  <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
                    <div>
                      <div style={{color:C.muted,fontSize:9,marginBottom:4}}>SPLINE NODES</div>
                      <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
                        {selMC.nodes.map(n=>(
                          <div key={n.id} style={{background:C.grid,padding:"4px 8px",
                                borderRadius:2,fontSize:9,color:C.text}}>
                            {n.id} ∠{(n.angle/Math.PI).toFixed(2)}π
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div style={{color:C.muted,fontSize:9,marginBottom:4}}>BLOCH STATE</div>
                      <div style={{fontSize:11,color:C.gold}}>
                        θ={( selMC.bloch_theta/Math.PI).toFixed(3)}π
                        &nbsp;φ={(selMC.bloch_phi/Math.PI).toFixed(3)}π
                      </div>
                    </div>
                    <div>
                      <div style={{color:C.muted,fontSize:9,marginBottom:4}}>AGENT PAIRS</div>
                      {selMC.agent_pairs.map(([a,b],i)=>(
                        <div key={i} style={{color:C.purple,fontSize:10}}>
                          SA{a} ⟷ SA{b}
                        </div>
                      ))}
                    </div>
                    <div>
                      <div style={{color:C.muted,fontSize:9,marginBottom:4}}>LOSS HISTORY</div>
                      <Sparkline data={selMC.loss_history} />
                    </div>
                  </div>
                </div>
              ):(
                <div style={{color:C.muted,fontSize:10,marginBottom:12}}>
                  ← select a meta-controller to inspect
                </div>
              )}

              {/* Score heatmap grid */}
              <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginBottom:8}}>
                SCORE TOPOLOGY · ALL LAYERS
              </div>
              <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                {bsnn.layers.map((layer,li)=>(
                  <div key={li} style={{background:C.grid,borderRadius:4,padding:8,
                                        border:`1px solid ${C.blue}22`}}>
                    <div style={{color:C.muted,fontSize:8,marginBottom:6}}>L{li}</div>
                    {layer.map(mc=>{
                      const s=mc.score;
                      const col=s>0.8?C.gold:s>0.6?C.cyan:s>0.4?C.blue:C.purple;
                      return (
                        <div key={mc.mc_id} onClick={()=>setSel(mc.mc_id)}
                          style={{display:"flex",alignItems:"center",gap:6,marginBottom:3,cursor:"pointer"}}>
                          <div style={{width:60,height:8,background:C.dim,borderRadius:2}}>
                            <div style={{width:`${s*100}%`,height:"100%",
                                         background:col,borderRadius:2,
                                         boxShadow:`0 0 4px ${col}`}} />
                          </div>
                          <span style={{color:C.muted,fontSize:8}}>{mc.mc_id}</span>
                          <span style={{color:col,fontSize:8}}>{(s*100).toFixed(0)}%</span>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>

              {/* Tron spline channel visualization */}
              <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginTop:12,marginBottom:8}}>
                TRON SPLINE CHANNELS
              </div>
              <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                {allMCs.slice(0,8).map(mc=>(
                  <div key={mc.mc_id} style={{background:C.grid,borderRadius:4,padding:4}}>
                    <SplineCanvas nodes={mc.nodes} mcId={mc.mc_id} score={mc.score} />
                    <div style={{color:C.muted,fontSize:8,textAlign:"center",marginTop:2}}>
                      {mc.mc_id}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab==="manifold" && (
            <div style={{ flex:1, padding:12, overflowY:"auto" }}>
              <div style={{display:"flex",gap:16,flexWrap:"wrap",alignItems:"flex-start"}}>
                <div>
                  <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginBottom:8}}>
                    BLOCH SPHERE COGNITIVE MANIFOLD
                  </div>
                  <div style={{background:C.grid,borderRadius:8,padding:8,
                               border:`1px solid ${C.blue}33`,
                               boxShadow:`0 0 20px ${C.cyan}11`}}>
                    <BlochCanvas manifold={bsnn.manifold} highlight={blochHL} />
                  </div>
                </div>
                <div style={{flex:1}}>
                  <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginBottom:8}}>
                    DIFFERENTIATION MODULE · θ/φ RECURSIVE REFINEMENT
                  </div>
                  <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                    {bsnn.manifold.slice(0,16).map((m,i)=>(
                      <div key={i} style={{
                        background:C.grid,borderRadius:4,padding:"4px 8px",
                        border:`1px solid ${i===blochHL?C.gold:C.muted}22`,
                        boxShadow:i===blochHL?`0 0 8px ${C.gold}44`:"none"
                      }}>
                        <div style={{color:C.muted,fontSize:7}}>M{i}</div>
                        <div style={{color:C.gold,fontSize:9}}>
                          θ={(m.theta/Math.PI).toFixed(2)}π
                        </div>
                        <div style={{color:C.cyan,fontSize:9}}>
                          φ={(m.phi/Math.PI).toFixed(2)}π
                        </div>
                        <div style={{color:C.text,fontSize:8}}>r={m.r.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{marginTop:12,background:C.grid,borderRadius:4,padding:8,
                               border:`1px solid ${C.purple}33`}}>
                    <div style={{color:C.purple,fontSize:9,letterSpacing:2,marginBottom:6}}>
                      FIDELITY-LOSS COUPLER
                    </div>
                    <div style={{color:C.text,fontSize:10}}>
                      L<sub>total</sub> = (1−α)·loss − α·F(ρ, σ<sub>target</sub>)
                    </div>
                    <div style={{color:C.muted,fontSize:9,marginTop:4}}>
                      α = 0.3 · target_state θ=π/4 φ=π/4
                    </div>
                    <div style={{color:C.cyan,fontSize:10,marginTop:4}}>
                      SVD rank compression → angular Bloch alignment active
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {tab==="kan" && (
            <div style={{ flex:1, padding:12, overflowY:"auto" }}>
              <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginBottom:8}}>
                KAN INNER-VECTOR CONCEPT STORE · simdjson EVOLUTIONARY PROCESSING
              </div>
              <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
                {[
                  ["STORED",bsnn.kan?.stored||0,C.cyan],
                  ["MEAN SCORE",((bsnn.kan?.mean_score||0)*100).toFixed(1)+"%",C.gold],
                  ["MAX SCORE",((bsnn.kan?.max_score||0)*100).toFixed(1)+"%",C.green||C.cyan],
                  ["KAN DIM","32",C.purple],
                  ["CAPACITY","256",C.blue],
                ].map(([k,v,col])=>(
                  <div key={k} style={{background:C.grid,border:`1px solid ${col}33`,
                                        borderRadius:4,padding:"8px 12px",minWidth:100}}>
                    <div style={{color:C.muted,fontSize:8,marginBottom:2}}>{k}</div>
                    <div style={{color:col,fontSize:16,fontWeight:"bold"}}>{v}</div>
                  </div>
                ))}
              </div>

              <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginTop:12,marginBottom:6}}>
                TOP CONCEPTS
              </div>
              {(bsnn.kan?.top_concepts||[]).map((c,i)=>(
                <div key={i} style={{background:C.grid,border:`1px solid ${C.blue}22`,
                                      borderRadius:4,padding:"6px 10px",marginBottom:4,
                                      display:"flex",gap:12}}>
                  <span style={{color:C.muted,fontSize:9}}>#{i}</span>
                  <span style={{color:C.cyan,fontSize:9}}>
                    SCORE {(c.score*100).toFixed(1)}%
                  </span>
                  <span style={{color:C.text,fontSize:9}}>
                    SIM {(c.sim*100).toFixed(1)}%
                  </span>
                  <div style={{flex:1}}>
                    <div style={{background:C.dim,height:4,borderRadius:2}}>
                      <div style={{width:`${c.score*100}%`,height:"100%",
                                   background:C.cyan,borderRadius:2}} />
                    </div>
                  </div>
                </div>
              ))}

              <div style={{marginTop:12,background:C.grid,borderRadius:4,padding:8,
                           border:`1px solid ${C.muted}22`,fontSize:9,color:C.muted}}>
                <div style={{color:C.text,marginBottom:4}}>EVOLUTIONARY LOOP</div>
                simdjson cart processes KAN stats → if mean_score &gt; 0.3 → subconscious MC spawn
                <br/>loss_history convergence watched per MC · new agents formed from discrete pairing
                <br/>SVD probability parallelism: all compressions compete for best angular alignment
              </div>
            </div>
          )}

          {tab==="log" && (
            <div style={{ flex:1, padding:12, overflowY:"auto" }}>
              <div style={{color:C.muted,fontSize:9,letterSpacing:2,marginBottom:8}}>
                EVOLUTION LOG
              </div>
              {log.length===0 && (
                <div style={{color:C.muted,fontSize:10}}>no events yet — run EVOLVE</div>
              )}
              {log.map((e,i)=>(
                <div key={i} style={{
                  display:"flex",gap:12,padding:"3px 0",
                  borderBottom:`1px solid ${C.dim}`,fontSize:10,
                  color: i===0?C.bright:C.text
                }}>
                  <span style={{color:C.muted,minWidth:28}}>G{e.gen}</span>
                  <span style={{color:C.gold}}>▲{(e.best*100).toFixed(1)}%</span>
                  <span style={{color:C.cyan}}>APEX {(e.apex*100).toFixed(1)}%</span>
                  <span style={{color:C.blue}}>KAN {e.kan}</span>
                  {e.sub && <span style={{color:C.purple}}>✦ SC</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
