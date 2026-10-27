#!/usr/bin/env python3
"""Mark the Asana tasks linked from a GitHub issue or PR body as complete.

Run by ``.github/workflows/asana_close.yml`` when a PR is merged or an issue is
closed as completed. Replaces a third-party action so the only code touching
the Asana token is this file.

Inputs come from the environment (never interpolated into a shell):

- ``BODY``       the issue / PR description to scan for task links
- ``ASANA_PAT``  an Asana personal access token (REST API). This is NOT the
                 ``ASANA_SECRET`` used by Asana's own GitHub actions, which
                 only authenticates against the github.integrations.asana.plus
                 proxy.

Both Asana URL generations are recognised::

    https://app.asana.com/0/{project}/{task}[/f]
    https://app.asana.com/1/{workspace}[/project/{project}]/task/{task}

Tasks the token cannot see (403/404) are reported as warnings and skipped; an
auth failure or server error fails the job so a broken token gets noticed.
Standard library only, so the workflow needs no ``pip install``.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://app.asana.com/api/1.0"

TASK_LINK = re.compile(
    r"app\.asana\.com/(?:"
    r"0/\d+/(?P<old>\d+)"
    r"|1/\d+/(?:project/\d+/)?task/(?P<new>\d+)"
    r")(?![\d])"
)


def find_task_gids(text):
    """Return the task gids linked in ``text``, de-duplicated in order."""
    gids = []
    for m in TASK_LINK.finditer(text or ""):
        gid = m.group("old") or m.group("new")
        if gid not in gids:
            gids.append(gid)
    return gids


def put_completed(gid, token):
    """PUT ``completed: true`` on one task; return ``(status, body)``."""
    req = urllib.request.Request(
        f"{API}/tasks/{gid}",
        data=json.dumps({"data": {"completed": True}}).encode(),
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def complete_all(gids, token, put=put_completed):
    """Complete each task; return the process exit code."""
    failed = False
    for gid in gids:
        status, body = put(gid, token)
        if 200 <= status < 300:
            print(f"Completed Asana task {gid}")
        elif status in (403, 404):
            print(f"::warning::Asana task {gid} not found or not visible to the token ({status}); skipped")
        else:
            print(f"::error::Completing Asana task {gid} failed ({status}): {body[:500]}")
            failed = True
    return 1 if failed else 0


def main():
    gids = find_task_gids(os.environ.get("BODY"))
    if not gids:
        print("No Asana task links found; nothing to do.")
        return 0
    token = os.environ.get("ASANA_PAT")
    if not token:
        print("::error::ASANA_PAT is not set; cannot complete linked Asana tasks.")
        return 1
    return complete_all(gids, token)


if __name__ == "__main__":
    sys.exit(main())
