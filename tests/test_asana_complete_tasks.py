"""Unit tests for ``scripts/asana_complete_tasks.py``.

Covers task-link parsing across Asana's old (``/0/``) and new (``/1/``) URL
formats and the per-task outcome handling. No network: the HTTP call is
replaced with a fake.
"""

import pytest

import scripts.asana_complete_tasks as act


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://app.asana.com/0/1201/1209", ["1209"]),
        ("https://app.asana.com/0/1201/1209/f", ["1209"]),
        ("https://app.asana.com/0/0/1209", ["1209"]),
        ("https://app.asana.com/1/111/project/1201/task/1209", ["1209"]),
        ("https://app.asana.com/1/111/project/1201/task/1209?focus=true", ["1209"]),
        ("https://app.asana.com/1/111/task/1209", ["1209"]),
        # Markdown link, as the Asana GitHub integration writes them
        ("Task: [Fix seam](https://app.asana.com/0/1201/1209/f)", ["1209"]),
    ],
)
def test_parses_task_links(text, expected):
    assert act.find_task_gids(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "no links here",
        "https://app.asana.com/0/1201/list",  # project view, not a task
        "https://app.asana.com/1/111/project/1201/board",
        "https://example.com/0/1201/1209",  # not Asana
        "https://app.asana.com/1/111/home",
    ],
)
def test_ignores_non_task_text(text):
    assert act.find_task_gids(text) == []


def test_dedupes_in_order():
    text = (
        "https://app.asana.com/0/1/22 and https://app.asana.com/0/1/11 "
        "and again https://app.asana.com/1/9/project/1/task/22"
    )
    assert act.find_task_gids(text) == ["22", "11"]


def test_complete_all_reports_each_outcome(capsys):
    statuses = {"1": 200, "2": 404, "3": 403}

    def fake_put(gid, token):
        assert token == "tok"
        return statuses[gid], ""

    assert act.complete_all(["1", "2", "3"], "tok", put=fake_put) == 0
    out = capsys.readouterr().out
    assert "Completed Asana task 1" in out
    assert "::warning::" in out and "task 2" in out and "task 3" in out


@pytest.mark.parametrize("status", [401, 500])
def test_complete_all_fails_on_auth_or_server_error(status, capsys):
    rc = act.complete_all(["1", "2"], "tok", put=lambda gid, token: (status, "boom"))
    assert rc == 1
    assert "::error::" in capsys.readouterr().out


def test_main_without_links_is_a_noop(monkeypatch, capsys):
    monkeypatch.setenv("ASANA_PAT", "tok")
    monkeypatch.setenv("BODY", "nothing linked")
    assert act.main() == 0
    assert "No Asana task links" in capsys.readouterr().out


def test_main_requires_token_when_links_present(monkeypatch, capsys):
    monkeypatch.delenv("ASANA_PAT", raising=False)
    monkeypatch.setenv("BODY", "https://app.asana.com/0/1/2")
    assert act.main() == 1
    assert "ASANA_PAT" in capsys.readouterr().out
