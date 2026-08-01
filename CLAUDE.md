# Repo conventions for Claude Code

## Giving the user shell commands to run manually

The user has clipboard/copy problems in some of their terminal environments
(e.g. cloud console/VNC sessions to remote servers). When a command needs to
be run manually by the user rather than executed directly by Claude, write it
to `TO_RUN.sh` at the repo root (overwrite it each time — it's a scratch
handoff file, not a history log) instead of relying on them copying it out of
the chat. Tell them the file exists and what it does; they'll copy from it in
an editor.
