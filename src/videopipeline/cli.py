from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, List, Optional

from .analysis_chat import compute_chat_analysis
from .analysis_highlights import compute_highlights_analysis
from .doctor import run_doctor
from .exporter import ExportSpec, HookTextSpec, LayoutPipSpec, run_ffmpeg_export
from .layouts import get_facecam_rect
from .metadata import build_metadata, derive_hook_text, write_metadata
from .profile import load_profile
from .project import add_selection_from_candidate, create_or_load_project, get_project_data
from .transcribe import TranscribeConfig, load_transcript_json, save_transcript_json, transcribe_segment
from .publisher.accounts import AccountStore
from .publisher.queue import PublishWorker
from .publisher.secrets import delete_tokens, store_tokens
from .publisher.state import accounts_path
from .publisher.connectors.tiktok import build_authorize_url, build_pkce_pair, exchange_code
from .publisher.presets import AccountPreset


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _default_out_path(video_path: Path) -> Path:
    return Path("outputs") / video_path.stem / "candidates_highlights.json"


def cmd_suggest(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)
    profile = load_profile(args.profile)
    analysis_cfg = profile.get("analysis", {})

    payload = compute_highlights_analysis(
        proj,
        audio_cfg=analysis_cfg.get("audio", {}),
        motion_cfg=analysis_cfg.get("motion", {}),
        scenes_cfg=analysis_cfg.get("scenes", {}),
        highlights_cfg=analysis_cfg.get("highlights", {}),
        include_chat=True,
    )

    out_path = args.out or _default_out_path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    candidates = payload.get("candidates", [])

    print(f"Wrote: {out_path}")
    print(f"Project: {proj.project_dir}")
    print()
    print(f"{'Rank':>4}  {'Peak':>10}  {'Start':>10}  {'End':>10}  {'Score':>7}  {'Audio':>7}  {'Motion':>7}  {'Chat':>7}")
    for c in candidates:
        breakdown = c.get("breakdown", {})
        print(
            f"{c.get('rank'):>4}  {_fmt_time(c.get('peak_time_s')):>10}  {_fmt_time(c.get('start_s')):>10}  {_fmt_time(c.get('end_s')):>10}  {c.get('score', 0.0):>7.2f}  {breakdown.get('audio', 0.0):>7.2f}  {breakdown.get('motion', 0.0):>7.2f}  {breakdown.get('chat', 0.0):>7.2f}"
        )


def cmd_studio(args: argparse.Namespace) -> None:
    from .studio.app import create_app

    app = create_app(video_path=args.video, profile_path=args.profile)

    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="warning")


def cmd_export(args: argparse.Namespace) -> None:
    # Headless export (useful in scripts/automation)
    spec = ExportSpec(
        video_path=args.video,
        start_s=args.start,
        end_s=args.end,
        output_path=args.out,
        template=args.template,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        normalize_audio=args.normalize_audio,
        subtitles_ass=args.subtitles_ass,
    )

    def on_prog(frac: float, msg: str) -> None:
        pct = int(frac * 100)
        print(f"\r{pct:3d}% {msg:10s}", end="", flush=True)

    run_ffmpeg_export(spec, on_progress=on_prog)
    print("\nDone:", spec.output_path)


def cmd_export_top(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)
    profile = load_profile(args.profile)
    proj_data = get_project_data(proj)

    highlights = proj_data.get("analysis", {}).get("highlights", {})
    candidates = highlights.get("candidates") or []
    if not candidates:
        analysis_cfg = profile.get("analysis", {})
        payload = compute_highlights_analysis(
            proj,
            audio_cfg=analysis_cfg.get("audio", {}),
            motion_cfg=analysis_cfg.get("motion", {}),
            scenes_cfg=analysis_cfg.get("scenes", {}),
            highlights_cfg=analysis_cfg.get("highlights", {}),
            include_chat=True,
        )
        candidates = payload.get("candidates", [])

    top_n = max(1, int(args.top))
    template = args.template or profile.get("export", {}).get("template", "vertical_blur")

    facecam = get_facecam_rect(proj_data.get("layout", {}))
    pip_cfg = profile.get("layout", {}).get("pip", {})
    hook_cfg = profile.get("overlay", {}).get("hook_text", {})
    export_cfg = profile.get("export", {})

    created: list[dict[str, Any]] = []
    for cand in candidates[:top_n]:
        sel_id = add_selection_from_candidate(proj, candidate=cand, template=template, title="")
        created.append(
            {
                "id": sel_id,
                "start_s": cand["start_s"],
                "end_s": cand["end_s"],
                "template": template,
                "candidate_rank": cand.get("rank"),
                "candidate_score": cand.get("score"),
                "candidate_peak_time_s": cand.get("peak_time_s"),
            }
        )

    for idx, selection in enumerate(created, start=1):
        start_s = float(selection["start_s"])
        end_s = float(selection["end_s"])
        out_path = proj.exports_dir / (
            f"{selection['id']}_{template}_{export_cfg.get('width', 1080)}x{export_cfg.get('height', 1920)}.mp4"
        )

        segments = None
        if args.with_captions:
            cfg = TranscribeConfig(
                model_size=str(profile.get("captions", {}).get("model_size", "small")),
                language=profile.get("captions", {}).get("language"),
                device=str(profile.get("captions", {}).get("device", "cpu")),
                compute_type=str(profile.get("captions", {}).get("compute_type", "int8")),
            )
            tjson = proj.analysis_dir / "transcripts" / f"{selection['id']}_{int(start_s)}_{int(end_s)}.json"
            if tjson.exists():
                segments = load_transcript_json(tjson)
            else:
                segments = transcribe_segment(proj.video_path, start_s=start_s, end_s=end_s, cfg=cfg)
                save_transcript_json(tjson, segments, cfg)
            ass_path = proj.analysis_dir / "subtitles" / f"{selection['id']}.ass"
            from .subtitles import write_ass

            subtitles_ass = write_ass(
                segments,
                ass_path,
                playres_x=int(export_cfg.get("width", 1080)),
                playres_y=int(export_cfg.get("height", 1920)),
            )
        else:
            subtitles_ass = None

        hook_spec = None
        if hook_cfg.get("enabled", False):
            hook_text = hook_cfg.get("text") or derive_hook_text(selection, segments)
            if hook_text:
                hook_spec = HookTextSpec(
                    enabled=True,
                    duration_seconds=float(hook_cfg.get("duration_seconds", 2.0)),
                    text=str(hook_text),
                    font=str(hook_cfg.get("font", "auto")),
                    fontsize=int(hook_cfg.get("fontsize", 64)),
                    y=int(hook_cfg.get("y", 120)),
                )

        spec = ExportSpec(
            video_path=proj.video_path,
            start_s=start_s,
            end_s=end_s,
            output_path=out_path,
            template=template,
            width=int(export_cfg.get("width", 1080)),
            height=int(export_cfg.get("height", 1920)),
            fps=int(export_cfg.get("fps", 30)),
            crf=int(export_cfg.get("crf", 20)),
            preset=str(export_cfg.get("preset", "veryfast")),
            normalize_audio=bool(export_cfg.get("normalize_audio", False)),
            subtitles_ass=subtitles_ass,
            layout_facecam=facecam,
            layout_pip=LayoutPipSpec(**pip_cfg) if pip_cfg else None,
            hook_text=hook_spec,
        )

        def on_prog(frac: float, msg: str) -> None:
            pct = int(frac * 100)
            print(f"\r[{idx}/{len(created)}] {pct:3d}% {msg:10s}", end="", flush=True)

        run_ffmpeg_export(spec, on_progress=on_prog)
        metadata = build_metadata(
            selection=selection,
            output_path=out_path,
            template=template,
            with_captions=args.with_captions,
            segments=segments,
        )
        write_metadata(out_path.with_suffix(".metadata.json"), metadata)
        print("\nDone:", out_path)


def cmd_doctor(_: argparse.Namespace) -> None:
    rep = run_doctor()
    print("VideoPipeline doctor\n")
    for name, data in rep.checks.items():
        print(f"- {name}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
    print("\nOK" if rep.ok else "\nNOT OK (fix missing requirements above)")


def cmd_attach_chat(args: argparse.Namespace) -> None:
    proj = create_or_load_project(args.video)
    proj.analysis_dir.mkdir(parents=True, exist_ok=True)
    dest = proj.chat_raw_path
    dest.write_bytes(Path(args.chat).read_bytes())

    hop_s = args.hop
    if hop_s is None:
        proj_data = get_project_data(proj)
        hop_s = (
            proj_data.get("analysis", {})
            .get("audio", {})
            .get("config", {})
            .get("hop_seconds", 0.5)
        )

    compute_chat_analysis(
        proj,
        chat_path=dest,
        hop_s=float(hop_s),
        smooth_s=float(args.smooth),
    )
    print(f"Attached chat: {dest}")
    print(f"Project: {proj.project_dir}")


def cmd_accounts_list(_: argparse.Namespace) -> None:
    store = AccountStore()
    accounts = store.list()
    if not accounts:
        print("No accounts configured.")
        return
    for acct in accounts:
        print(f"{acct.id}  {acct.platform}  {acct.label}")


def cmd_accounts_add_youtube(args: argparse.Namespace) -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    scopes = args.scopes.split(",") if args.scopes else ["https://www.googleapis.com/auth/youtube.upload"]
    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets), scopes=scopes)
    creds = flow.run_local_server(port=args.redirect_port)
    tokens = {
        "access_token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }
    store = AccountStore()
    account = store.add(platform="youtube", label=args.label or "YouTube")
    store_tokens("youtube", account.id, tokens)
    print(f"Added YouTube account: {account.id}")
    print(f"Accounts file: {accounts_path()}")


def cmd_accounts_add_tiktok(args: argparse.Namespace) -> None:
    verifier, challenge = build_pkce_pair()
    state = uuid.uuid4().hex
    redirect_uri = f"http://127.0.0.1:{args.redirect_port}/callback"
    scopes = args.scopes or "user.info.basic,video.upload"
    url = build_authorize_url(
        client_key=args.client_key,
        redirect_uri=redirect_uri,
        scopes=scopes,
        state=state,
        code_challenge=challenge,
    )

    code_holder: dict[str, str] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/callback"):
                query = self.path.split("?", 1)[-1]
                params = dict(pair.split("=", 1) for pair in query.split("&") if "=" in pair)
                if params.get("state") != state:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Invalid state.")
                    return
                code = params.get("code")
                if code:
                    code_holder["code"] = code
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"Authorization received. You can close this window.")
                    done.set()
                    return
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing code.")

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", args.redirect_port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    webbrowser.open(url)
    print("Waiting for TikTok authorization...")
    done.wait(timeout=180)
    server.shutdown()

    code = code_holder.get("code")
    if not code:
        raise RuntimeError("TikTok authorization timed out.")

    token_payload = exchange_code(
        client_key=args.client_key,
        client_secret=args.client_secret,
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
    )
    token_payload["client_key"] = args.client_key
    token_payload["client_secret"] = args.client_secret
    token_payload["expires_at"] = time.time() + float(token_payload.get("expires_in", 0))

    store = AccountStore()
    presets = AccountPreset(default_privacy="private").to_dict()
    account = store.add(platform="tiktok", label=args.label or "TikTok", presets=presets)
    store_tokens("tiktok", account.id, token_payload)
    print(f"Added TikTok account: {account.id}")


def cmd_accounts_remove(args: argparse.Namespace) -> None:
    store = AccountStore()
    account = store.get(args.account_id)
    if not account:
        print("Account not found.")
        return
    store.remove(args.account_id)
    delete_tokens(account.platform, account.id)
    print(f"Removed account: {args.account_id}")


def cmd_publish(args: argparse.Namespace) -> None:
    store = AccountStore()
    account = store.get(args.account)
    if not account:
        raise RuntimeError("account_not_found")
    worker = PublishWorker(account_store=store)
    job = worker.queue_job(
        platform=account.platform,
        account_id=account.id,
        file_path=args.export,
        metadata_path=args.meta,
    )
    print(f"Queued job: {job.id}")
    if not args.async_mode:
        worker.run_once(job.id)
        job = worker.job_store.get_job(job.id)
        print(f"Job status: {job.status}")


def cmd_publish_project(args: argparse.Namespace) -> None:
    project_path = Path(args.project)
    if not project_path.exists():
        project_path = Path("outputs") / "projects" / args.project
    project_file = project_path / "project.json"
    if not project_file.exists():
        raise RuntimeError("project_not_found")

    proj_data = json.loads(project_file.read_text(encoding="utf-8"))
    exports = proj_data.get("exports") or []
    if not exports:
        print("No exports to publish.")
        return

    store = AccountStore()
    account = store.get(args.account)
    if not account:
        raise RuntimeError("account_not_found")
    worker = PublishWorker(account_store=store)
    queued = 0
    for exp in exports:
        output = Path(exp.get("output") or "")
        meta = output.with_suffix(".metadata.json")
        if not output.exists() or not meta.exists():
            continue
        worker.queue_job(
            platform=account.platform,
            account_id=account.id,
            file_path=output,
            metadata_path=meta,
        )
        queued += 1
    print(f"Queued {queued} publish jobs.")


def cmd_jobs_list(_: argparse.Namespace) -> None:
    worker = PublishWorker()
    jobs = worker.job_store.list_jobs()
    for job in jobs:
        print(
            f"{job.id}  {job.status}  {job.platform}  {job.account_id}  "
            f"{job.progress:.0%}  attempts={job.attempts}"
        )


def cmd_jobs_retry(args: argparse.Namespace) -> None:
    worker = PublishWorker()
    job = worker.job_store.retry(args.job_id)
    print(f"Job {job.id} queued for retry.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="vp", description="VideoPipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("suggest", help="Suggest highlight candidates from combined signals.")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--profile", type=Path, default=None)
    p.set_defaults(func=cmd_suggest)

    s = sub.add_parser("studio", help="Launch the local review + export studio web app.")
    s.add_argument("video", type=Path, nargs="?", default=None, help="Video file to open (optional, launches Home if omitted)")
    s.add_argument("--profile", type=Path, default=None, help="Path to a YAML profile (e.g. profiles/gaming.yaml)")
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--reload", action="store_true", help="Auto-reload server on code changes (dev)")
    s.add_argument("--no-open", action="store_true", help="Do not open a browser automatically")
    s.set_defaults(func=cmd_studio)

    e = sub.add_parser("export", help="Headless export of a specific clip range.")
    e.add_argument("video", type=Path)
    e.add_argument("--start", type=float, required=True)
    e.add_argument("--end", type=float, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--template", type=str, default="vertical_blur")
    e.add_argument("--width", type=int, default=1080)
    e.add_argument("--height", type=int, default=1920)
    e.add_argument("--fps", type=int, default=30)
    e.add_argument("--crf", type=int, default=20)
    e.add_argument("--preset", type=str, default="veryfast")
    e.add_argument("--normalize-audio", action="store_true")
    e.add_argument("--subtitles-ass", type=Path, default=None)
    e.set_defaults(func=cmd_export)

    et = sub.add_parser("export-top", help="Create selections from top candidates and export them.")
    et.add_argument("video", type=Path)
    et.add_argument("--top", type=int, default=10)
    et.add_argument("--template", type=str, default=None)
    et.add_argument("--profile", type=Path, default=None)
    et.add_argument("--with-captions", action="store_true")
    et.set_defaults(func=cmd_export_top)

    d = sub.add_parser("doctor", help="Check local system dependencies (ffmpeg, optional whisper).")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("attach-chat", help="Attach chat JSON and compute chat features.")
    c.add_argument("video", type=Path)
    c.add_argument("chat", type=Path)
    c.add_argument("--hop", type=float, default=None)
    c.add_argument("--smooth", type=float, default=3.0)
    c.set_defaults(func=cmd_attach_chat)

    accounts = sub.add_parser("accounts", help="Manage publishing accounts.")
    accounts_sub = accounts.add_subparsers(dest="accounts_cmd", required=True)

    accounts_list = accounts_sub.add_parser("list", help="List configured accounts.")
    accounts_list.set_defaults(func=cmd_accounts_list)

    accounts_add = accounts_sub.add_parser("add", help="Add a new publishing account.")
    accounts_add_sub = accounts_add.add_subparsers(dest="platform", required=True)

    youtube_add = accounts_add_sub.add_parser("youtube", help="Add a YouTube account via OAuth.")
    youtube_add.add_argument("--client-secrets", type=Path, required=True)
    youtube_add.add_argument("--label", type=str, default=None)
    youtube_add.add_argument("--scopes", type=str, default=None)
    youtube_add.add_argument("--redirect-port", type=int, default=8080)
    youtube_add.set_defaults(func=cmd_accounts_add_youtube)

    tiktok_add = accounts_add_sub.add_parser("tiktok", help="Add a TikTok account via OAuth.")
    tiktok_add.add_argument("--client-key", type=str, required=True)
    tiktok_add.add_argument("--client-secret", type=str, required=True)
    tiktok_add.add_argument("--label", type=str, default=None)
    tiktok_add.add_argument("--redirect-port", type=int, default=3455)
    tiktok_add.add_argument("--scopes", type=str, default=None)
    tiktok_add.set_defaults(func=cmd_accounts_add_tiktok)

    accounts_remove = accounts_sub.add_parser("remove", help="Remove an account.")
    accounts_remove.add_argument("account_id", type=str)
    accounts_remove.set_defaults(func=cmd_accounts_remove)

    publish = sub.add_parser("publish", help="Queue a publish job for an exported clip.")
    publish.add_argument("--account", type=str, required=True)
    publish.add_argument("--export", type=Path, required=True)
    publish.add_argument("--meta", type=Path, required=True)
    publish.add_argument("--async", dest="async_mode", action="store_true")
    publish.set_defaults(func=cmd_publish)

    publish_project = sub.add_parser("publish-project", help="Queue publish jobs for a project.")
    publish_project.add_argument("--project", type=str, required=True)
    publish_project.add_argument("--account", type=str, required=True)
    publish_project.add_argument("--all-exports", action="store_true")
    publish_project.set_defaults(func=cmd_publish_project)

    jobs = sub.add_parser("jobs", help="Manage publish jobs.")
    jobs_sub = jobs.add_subparsers(dest="jobs_cmd", required=True)

    jobs_list = jobs_sub.add_parser("list", help="List recent publish jobs.")
    jobs_list.set_defaults(func=cmd_jobs_list)

    jobs_retry = jobs_sub.add_parser("retry", help="Retry a publish job.")
    jobs_retry.add_argument("job_id", type=str)
    jobs_retry.set_defaults(func=cmd_jobs_retry)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
