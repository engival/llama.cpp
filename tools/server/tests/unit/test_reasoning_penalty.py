import pytest
from utils import *

server = ServerPreset.tinyllama2()

# the tiny vocab has no single-token spelling for most marker words: use words that resolve on it
SERVER_WORDS = "the,and,=e"

# the stories model has no thinking tags: pass them with the request so the sampler is created
TAGS = {
    "reasoning_budget_start_tag": "<think>",
    "reasoning_budget_end_tags": ["</think>"],
}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.reasoning_penalty_words = SERVER_WORDS
    server.reasoning_penalty_step = 0.5


def check_stats(stats: dict):
    for key in ("triggers", "markers", "tokens_in_think"):
        assert type(stats[key]) == int and stats[key] >= 0
    assert type(stats["hits"]) == dict


def test_reasoning_penalty_null_uses_server_default():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "reasoning_penalty": None,
        **TAGS,
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]["reasoning_penalty"]
    assert settings["words"] == SERVER_WORDS.split(",")
    assert settings["step"] == 0.5
    check_stats(res.body["reasoning_penalty"])


def test_reasoning_penalty_empty_words_disables():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "reasoning_penalty": {"words": []},
        **TAGS,
    })
    assert res.status_code == 200
    assert res.body["generation_settings"]["reasoning_penalty"]["words"] == []
    assert "reasoning_penalty" not in res.body


def test_reasoning_penalty_default_words():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "reasoning_penalty": {"words": "default"},
        **TAGS,
    })
    assert res.status_code == 200
    words = res.body["generation_settings"]["reasoning_penalty"]["words"]
    assert len(words) == 50
    assert "Wait" in words


def test_reasoning_penalty_invalid_words():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "reasoning_penalty": {"words": "all"},
    })
    assert res.status_code == 400


def test_reasoning_penalty_partial_object_inherits_defaults():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "reasoning_penalty": {"max": 1.5, "window": 16},
        **TAGS,
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]["reasoning_penalty"]
    assert settings["words"] == SERVER_WORDS.split(",")
    assert settings["step"] == 0.5
    assert settings["max"] == 1.5
    assert settings["window"] == 16
    check_stats(res.body["reasoning_penalty"])


def test_reasoning_penalty_no_tags_no_stats():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert "reasoning_penalty" not in res.body


def test_reasoning_penalty_stream():
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 8,
        "stream": True,
        **TAGS,
    })
    n_final = 0
    for data in res:
        if data["stop"]:
            n_final += 1
            check_stats(data["reasoning_penalty"])
        else:
            assert "reasoning_penalty" not in data
    assert n_final == 1


def test_reasoning_penalty_oai_completions():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": "I believe the meaning of life is",
        "max_tokens": 8,
        **TAGS,
    })
    assert res.status_code == 200
    check_stats(res.body["reasoning_penalty"])


def test_reasoning_penalty_oai_completions_stream():
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/completions", data={
        "prompt": "I believe the meaning of life is",
        "max_tokens": 8,
        "stream": True,
        **TAGS,
    })
    finals = [data for data in res if "reasoning_penalty" in data]
    assert len(finals) == 1
    check_stats(finals[0]["reasoning_penalty"])
