"""New collection interface; reuse the frozen mannequin renderer without altering v4."""

from motionlab.noticeability.protocol import BODY_PARTS, PRIMARY_QUESTION, REASON_TAGS
from motionlab.perceptual.subjective import _HTML

# ruff: noqa: E501, RUF001
_RENDERER = _HTML[_HTML.index("function add(") : _HTML.index("window.__motionlabInspectionState")]

HTML = (
    r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion noticeability pilot</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#12151b;color:#e5ebf5;font:16px system-ui}main{width:min(1040px,calc(100% - 28px));margin:24px auto 56px;padding:18px;border:1px solid #2b3444}button,select,input{background:#263345;color:inherit;border:1px solid #53657c;border-radius:6px;padding:10px;margin:4px}button{cursor:pointer}button:disabled{opacity:.35;cursor:default}#viewportWrap{position:relative;background:#12151b;border:1px solid #313a48;border-radius:12px;overflow:hidden;aspect-ratio:3/2;touch-action:none}canvas{width:100%;height:100%;display:block}fieldset{border:1px solid #53657c;margin:12px 0}#status{color:#ffd080;min-height:24px}#error{color:#ffaaaa;white-space:pre-wrap}#context{max-width:900px}label{display:inline-block;margin:4px}[hidden]{display:none!important}.answers button{min-width:120px;font-size:19px}.selected{outline:2px solid #93cfff}
</style></head><body><main>
<h2>Motion noticeability</h2>
<p>Watch once at normal speed. Answer what you noticed during that viewing. Inspection is optional and comes afterward.</p>
<div id="login"><label>Rater ID <input id="rater" value="123" autocomplete="off"></label><button id="begin">Begin / resume</button></div>
<p id="context"></p><p id="status"></p><p id="error" role="alert"></p>
<div id="viewportWrap" hidden><canvas id="view" width="960" height="640" hidden></canvas></div>
<button id="watch" hidden>Watch once at 1×</button>
<section id="primary" hidden><h3>__QUESTION__</h3><div class="answers" id="primaryAnswers"><button data-notice="NO">NO</button><button data-notice="MAYBE">MAYBE / UNSURE</button><button data-notice="YES">YES</button></div><p>Your first answer is saved immediately and cannot be edited.</p></section>
<fieldset id="inspection" hidden><legend>Optional inspection — first answer already saved</legend>
<button id="replay">Replay</button><button id="pause">Pause / resume</button>
<select id="speed" aria-label="Playback speed"><option>.25</option><option>.5</option><option selected>1</option><option>1.5</option><option>2</option></select>
<button id="zoomOut">Zoom −</button><button id="zoomIn">Zoom +</button><button id="reset">Reset camera</button>
<select id="preset" aria-label="Camera view"><option value="three_quarter">3/4 view</option><option value="front">Front</option><option value="side">Side</option></select>
<label><input type="checkbox" id="overlay">Skeleton overlay</label>
<p>Scroll to zoom; drag to orbit; Shift-drag or right-drag to pan (where supported by the original render version).</p>
<h3>After inspecting it, can you identify an unintended problem?</h3><div class="answers" id="inspectedAnswers"><button data-notice="NO">NO</button><button data-notice="MAYBE">MAYBE</button><button data-notice="YES">YES</button></div>
<p>What seemed wrong? Optional — YES does not require a diagnosis.</p><div id="tags">__TAGS__</div>
<label>Body part <select id="part"><option value="">Not specified</option>__PARTS__</select></label>
<label>Confidence <select id="confidence"><option value="">Not specified</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<button id="mark">Mark interval point</button><button id="clearMark">Clear interval</button><span id="interval"></span>
<p><button id="next">Save optional inspection and next</button></p></fieldset>
<button id="skip" hidden>Skip interrupted presentation and continue</button>
</main><script>
const $=id=>document.getElementById(id);
let trial=null,motions=[],current=0,frame=0,camera={},stage='login',playing=false,phaseTime=0,rate=1,last=0,primaryID=null,inspectStart=0,inspected=null,marks=[],drag=null,telem=null;
const clone=x=>JSON.parse(JSON.stringify(x));
function status(s){$('status').textContent=s}
function blank(){let c=$('view');c.getContext('2d').clearRect(0,0,c.width,c.height);c.hidden=true;$('viewportWrap').hidden=true}
async function api(route,data={}){let r=await fetch('/api/'+route,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});let v=await r.json();if(!r.ok)throw Error(v.error||'Request failed');return v}
async function safely(fn){$('error').textContent='';try{await fn()}catch(e){$('error').textContent=e.message}}
function telemetry(){return {...clone(telem),total_inspection_time_ms:performance.now()-inspectStart,final_playback_rate:rate,final_camera_state:clone(camera)}}
function showStage(s){stage=s;$('watch').hidden=s!=='ready';$('primary').hidden=s!=='question';$('inspection').hidden=s!=='inspection';$('skip').hidden=s!=='interrupted'}
async function next(){playing=false;blank();showStage('loading');let v=await api('next');if(!v.trial){status(v.message);return}trial=v.trial;motions=[v.motion];current=0;frame=0;phaseTime=0;rate=1;primaryID=null;camera=clone(trial.viewing_settings.default_camera);inspected=null;marks=[];$('interval').textContent='';$('context').textContent=trial.evaluation_goal+' '+trial.style_context;$('speed').value='1';$('preset').value='three_quarter';$('overlay').checked=false;$('part').value='';$('confidence').value='';document.querySelectorAll('#tags input').forEach(x=>x.checked=false);document.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));status('Trial '+(trial.display_index)+' of '+trial.total+'. Ready for one normal-speed presentation.');showStage('ready')}
$('begin').onclick=()=>safely(async()=>{await api('login',{rater_id:$('rater').value});$('login').hidden=true;await next()});
$('watch').onclick=()=>safely(async()=>{showStage('starting');$('viewportWrap').hidden=false;$('view').hidden=false;$('viewportWrap').scrollIntoView({block:'center'});try{await api('start',{trial_id:trial.trial_id})}catch(e){blank();showStage('interrupted');throw e}if(stage!=='starting')return;showStage('initial');phaseTime=0;frame=0;last=performance.now();playing=true;draw();status('Watch once — normal speed and camera. Controls unlock after your first answer.')});
for(const b of document.querySelectorAll('#primaryAnswers button'))b.onclick=()=>safely(async()=>{if(stage!=='question')return;showStage('saving');let v;try{v=await api('spontaneous',{trial_id:trial.trial_id,spontaneous_notice:b.dataset.notice})}catch(e){showStage('question');throw e}primaryID=v.observation_id;inspectStart=performance.now();telem={format_version:'motionlab.inspection_telemetry.v1',replay_count:[0],speed_change_events:[],playback_wall_time_by_speed_ms:{'0.25':0,'0.5':0,'1':0,'1.5':0,'2':0},camera_change_events:[],zoom_change_count:0,view_change_count:0,overlay_change_count:0};showStage('inspection');$('viewportWrap').hidden=false;$('view').hidden=false;draw();status('First answer saved: '+b.dataset.notice+'. Inspect if useful, or save and continue.')});
for(const b of document.querySelectorAll('#inspectedAnswers button'))b.onclick=()=>{inspected=b.dataset.notice;document.querySelectorAll('#inspectedAnswers button').forEach(x=>x.classList.toggle('selected',x===b))};
$('next').onclick=()=>safely(async()=>{if(stage!=='inspection')return;playing=false;$('next').disabled=true;try{await api('inspection',{primary_observation_id:primaryID,inspected_notice:inspected,inspection_telemetry:telemetry(),optional_reason_tags:[...document.querySelectorAll('#tags input:checked')].map(x=>x.value),optional_body_part:$('part').value||null,optional_interval:marks.length?[...marks].sort((a,b)=>a-b):null,optional_confidence:$('confidence').value?Number($('confidence').value):null});await next()}finally{$('next').disabled=false}});
$('skip').onclick=()=>safely(async()=>{await api('interrupt',{trial_id:trial.trial_id});await next()});
function interrupt(){if(!['initial','starting'].includes(stage))return;playing=false;blank();showStage('interrupted');status('Presentation interrupted. No human answer was recorded; skip this exposed trial.');api('interrupt',{trial_id:trial.trial_id}).catch(e=>$('error').textContent=e.message)}
document.addEventListener('visibilitychange',()=>{if(document.hidden)interrupt()});window.addEventListener('blur',interrupt);
$('replay').onclick=()=>{if(stage!=='inspection')return;telem.replay_count[0]++;phaseTime=0;playing=true;last=performance.now()};
$('pause').onclick=()=>{if(stage==='inspection'){playing=!playing;last=performance.now()}};
$('speed').onchange=()=>{if(stage!=='inspection')return;if(telem.speed_change_events.length>=512)return;let newRate=Number($('speed').value);telem.speed_change_events.push({elapsed_ms:performance.now()-inspectStart,from_rate:rate,to_rate:newRate});rate=newRate};
function changeCamera(kind,fn){if(stage!=='inspection'||telem.camera_change_events.length>=512)return;let from=clone(camera);fn();telem.camera_change_events.push({elapsed_ms:performance.now()-inspectStart,kind,from,to:clone(camera)});if(kind==='zoom')telem.zoom_change_count++;if(kind==='view')telem.view_change_count++;if(kind==='overlay')telem.overlay_change_count++;draw()}
function zoom(factor){changeCamera('zoom',()=>{let z=trial.viewing_settings.zoom_limits;camera.zoom=Math.max(z.minimum,Math.min(z.maximum,camera.zoom*factor))})}
$('zoomOut').onclick=()=>zoom(.8);$('zoomIn').onclick=()=>zoom(1.25);
$('reset').onclick=()=>changeCamera('reset',()=>{camera=clone(trial.viewing_settings.default_camera);$('overlay').checked=false;$('preset').value='three_quarter'});
$('preset').onchange=()=>changeCamera('view',()=>{let key=$('preset').value;Object.assign(camera,trial.viewing_settings.camera_presets[key],{preset:key})});
$('overlay').onchange=()=>changeCamera('overlay',()=>camera.skeleton_overlay=$('overlay').checked);
$('mark').onclick=()=>{marks.push(Math.min(phaseTime,trial.duration_s));marks=marks.slice(-2);$('interval').textContent=marks.map(x=>x.toFixed(2)+'s').join(' — ')};
$('clearMark').onclick=()=>{marks=[];$('interval').textContent=''};
$('view').addEventListener('wheel',e=>{e.preventDefault();if(stage==='inspection')zoom(Math.exp(-Math.max(-1200,Math.min(1200,e.deltaY*(e.deltaMode===1?16:1)))*.0015))},{passive:false});
$('view').oncontextmenu=e=>e.preventDefault();
$('view').onpointerdown=e=>{if(stage!=='inspection')return;drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button!==0,from:clone(camera)};$('view').setPointerCapture(e.pointerId)};
$('view').onpointermove=e=>{if(!drag||stage!=='inspection')return;let dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){if(!trial.viewing_settings.pan?.enabled)return;let scale=trial.viewing_settings.pixels_per_meter*camera.zoom;camera.pan_x_m=Math.max(-20,Math.min(20,drag.from.pan_x_m+dx/scale));camera.pan_y_m=Math.max(-20,Math.min(20,drag.from.pan_y_m-dy/scale))}else{camera.yaw_degrees=drag.from.yaw_degrees+dx*.35;let limit=trial.viewing_settings.orbit.maximum_pitch_degrees;camera.pitch_degrees=Math.max(-limit,Math.min(limit,drag.from.pitch_degrees+dy*.35));camera.preset='orbit'}draw()};
function endDrag(){if(!drag)return;let to=clone(camera),from=drag.from,kind=drag.pan?'pan':'orbit';drag=null;camera=from;if(kind==='pan'&&!trial.viewing_settings.pan?.enabled)return;changeCamera(kind,()=>camera=to)}
$('view').onpointerup=endDrag;$('view').onpointercancel=endDrag;
async function tick(t){let dt=Math.max(0,(t-last)/1000);last=t;if(playing){if(stage==='initial'&&dt>.3){interrupt()}else{phaseTime+=dt*rate;if(stage==='inspection')telem.playback_wall_time_by_speed_ms[String(rate)]+=dt*1000;if(phaseTime>=trial.duration_s){playing=false;phaseTime=trial.duration_s;if(stage==='initial'){blank();showStage('confirming');try{await api('complete',{trial_id:trial.trial_id});showStage('question');status('Answer based only on that initial viewing.')}catch(e){showStage('interrupted');$('error').textContent=e.message}}}frame=Math.min(motions[0].positions.length-1,Math.floor(phaseTime*motions[0].fps));if(['initial','inspection'].includes(stage))draw()}}requestAnimationFrame(tick)}
window.__noticeabilityState=()=>({stage,frame,playing,camera:clone(camera),primaryID,trial});
__RENDERER__
requestAnimationFrame(tick);
</script></body></html>""".replace("__QUESTION__", PRIMARY_QUESTION)
    .replace(
        "__TAGS__",
        "".join(
            f'<label><input type="checkbox" value="{tag}">{tag}</label>' for tag in REASON_TAGS
        ),
    )
    .replace("__PARTS__", "".join(f"<option>{part}</option>" for part in BODY_PARTS))
    .replace("__RENDERER__", _RENDERER)
)
