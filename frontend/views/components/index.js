// views/components/index.js — Reusable UI components
export const Badge=(t,v='g')=>`<span class="bg bg-${v}">${t}</span>`;
export const DotBadge=(t,v='g',pulse=false)=>`<span class="bg bg-${v}"><span class="dot dot-${v}${pulse?' dot-pulse':''}"></span>${t}</span>`;
export const Btn=(label,onclick,v='g',size='',disabled=false)=>`<button class="btn btn-${v}${size?' btn-'+size:''}" onclick="${onclick}"${disabled?' disabled':''}>${label}</button>`;
export const Card=(label,body,style='')=>`<div class="cd"${style?` style="${style}"`:''}>` + (label?`<div class="cd-label">${label}</div>`:'')+body+'</div>';
export const Toggle=(on,onclick)=>`<div class="toggle" style="background:${on?'var(--g)':'var(--brd)'}" onclick="${onclick}"><div class="knob" style="background:${on?'var(--bg)':'var(--mx)'};left:${on?'21px':'3px'}"></div></div>`;
export const Bar=(pct,color='var(--g)')=>`<div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>`;
export const StatBox=(val,label,color='var(--g)')=>`<div class="stat"><div class="stat-v" style="color:${color}">${val}</div><div class="stat-l">${label}</div></div>`;
export const SettingRow=(label,desc,ctrl)=>`<div class="srow"><div><div class="srow-label">${label}</div>${desc?`<div class="srow-desc">${desc}</div>`:''}</div>${ctrl}</div>`;
export const LogEntry=(e,op=1)=>{const cc=e.conf>.9?'var(--g)':e.conf>.7?'var(--a)':'var(--r)';return`<div class="log-entry" style="opacity:${op}"><div class="fr g5">${Badge(e.gesture,'g')}${e.combo?Badge('⚡'+e.combo,'p'):''}${e.model?`<span style="font-size:8px;color:var(--dm)">[${e.model}]</span>`:''}</div><div class="fr" style="gap:10px"><span style="font-size:11px;font-weight:700;color:${cc}">${(e.conf*100).toFixed(1)}%</span><span style="font-size:9px;color:var(--dm)">${e.time.toLocaleTimeString()}</span></div></div>`};
export const FingerBars=(names,colors)=>`<div class="fgrid">${names.map((n,i)=>`<div class="fcol"><div class="fbar-w"><div class="fbar-f" id="fb${i}" style="height:0%;background:${colors[i]}"></div></div><div class="fname" style="color:${colors[i]}">${n}</div><div class="fval" id="fv${i}">0%</div></div>`).join('')}</div>`;
