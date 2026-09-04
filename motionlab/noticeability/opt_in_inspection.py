"""Frozen, explicit-opt-in inspection UI; the repeated-view primary protocol is unchanged."""

from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from motionlab.dataset.io import file_sha256
from motionlab.noticeability import repeated_protocol, repeated_server
from motionlab.noticeability.repeated_continuation import prepare_repeated_view_pilot
from motionlab.noticeability.repeated_ui import HTML as REPEATED_HTML


def _replace_once(html: str, old: str, new: str) -> str:
    if html.count(old) != 1:
        raise ValueError("frozen repeated-view UI changed; review the opt-in presentation revision")
    return html.replace(old, new, 1)


def build_html() -> str:
    html = _replace_once(
        REPEATED_HTML,
        "Detailed inspection comes afterward.",
        "Inspection stays hidden unless you choose Inspect after answering.",
    )
    html = _replace_once(
        html,
        '<fieldset id="inspection" hidden>',
        '<section id="postAnswer" hidden><button id="continue" disabled>Next animation</button>'
        '<button id="openInspection" disabled>Inspect (optional)</button></section>'
        '<fieldset id="inspection" hidden>',
    )
    html = _replace_once(
        html,
        "$('inspection').hidden=s!=='inspection';",
        "$('inspection').hidden=s!=='inspection';$('postAnswer').hidden=s!=='answered';"
        "$('continue').disabled=s!=='answered';$('openInspection').disabled=s!=='answered';",
    )
    html = _replace_once(
        html,
        "primaryID=v.observation_id;rate=1;inspectStart=performance.now();",
        "primaryID=v.observation_id;rate=1;inspectStart=0;",
    )
    html = _replace_once(
        html,
        "showStage('inspection');$('viewportWrap').hidden=false;$('view').hidden=false;draw();"
        "status('Answer saved: '+b.dataset.notice+' after '+viewings+' viewing(s). "
        "Inspect if useful, or save and continue.')",
        "showStage('answered');blank();status('Answer saved: '+b.dataset.notice+"
        "'. Continue to the next animation, or choose Inspect if you want a closer look.')",
    )
    html = _replace_once(
        html,
        "$('skip').onclick=",
        """
$('continue').onclick=()=>safely(async()=>{
  if(stage!=='answered')return;
  try{await next()}catch(e){showStage('answered');throw e}
});
$('openInspection').onclick=()=>{
  if(stage!=='answered')return;
  inspectStart=performance.now();showStage('inspection');
  $('viewportWrap').hidden=false;$('view').hidden=false;draw();
  status('Optional inspection. Your original answer is already saved.');
};
$('skip').onclick=""",
    )
    return html


HTML = build_html()


def presentation_identity() -> dict[str, Any]:
    return {
        "format_version": "motionlab.inspection_presentation.opt_in.v1",
        "mode": "explicit_opt_in",
        "implementation_hash": file_sha256(Path(__file__)),
        "html_hash": repeated_protocol.identity("opt-in-inspection-ui", HTML),
        "inspection_time_starts": "on_explicit_open",
        "continue_without_inspection": "no_inspection_amendment",
    }


def prepare_pilot(parent: Path, output: Path) -> dict[str, Any]:
    return prepare_repeated_view_pilot(
        parent, output, inspection_presentation=presentation_identity()
    )


def make_server(manifest_path: Path, *, port: int = 8765) -> ThreadingHTTPServer:
    manifest = repeated_protocol.validate_pilot(manifest_path)
    # This revision is itself part of the frozen manifest. Never change a running UI silently.
    if manifest.get("inspection_presentation") != presentation_identity():
        raise ValueError("opt-in inspection implementation differs from the frozen presentation")
    server = repeated_server.make_server(manifest_path, port=port)
    base = cast(type, server.RequestHandlerClass)

    def do_get(self: Any) -> None:
        if self.path != "/":
            cast(Any, base).do_GET(self)
            return
        try:
            self.local()
        except ValueError as exc:
            self.reply({"error": str(exc)}, 400)
            return
        data = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    server.RequestHandlerClass = type("OptInInspectionHandler", (base,), {"do_GET": do_get})
    return server


def serve_pilot(manifest_path: Path, *, port: int = 8765) -> None:
    with make_server(manifest_path, port=port) as server:
        print(
            f"Noticeability pilot (opt-in inspection): http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        server.serve_forever()
