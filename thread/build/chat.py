"""The world-plan model: Qwen3.5 35B-A3B (MLX build, 4 bit), run with mlx-lm. It writes the list of things, the style
guide and the material choices. setup.sh downloads the weights; at run time nothing goes to the network.
Callers in thread/.venv use Chat (a conversation) or ask_many (several questions, one load). Both run this file in
spike/.venv, which has MLX, as ONE process that leaves memory when its last answer is in: one large model at a time.
  spike/.venv/bin/python chat.py      one JSON a line in: {"messages": [{"role", "content"}...], "max_tokens", "temperature"}
                                      one JSON a line out: {"text", "s", "gen_tokens", "peak_gb", "finish"}; the first line out is {"load_s"}
A request may also carry "tools" (function descriptions, which the model's own chat template lays out: it then answers
with <tool_call> blocks, see tool_calls) and "keep": true, for a conversation that grows turn by turn: the model's work
on the part of the conversation it has already read is kept, so each turn costs only what is new."""
import sys, os, json, subprocess, time
import where

MODEL, REVISION = "mlx-community/Qwen3.5-35B-A3B-4bit", "1e20fd8d42056f870933bf98ca6211024744f7ec"
# the art director's eyes (RFC pillar 4): Qwen3-VL 32B, a model trained for looking at pictures, when asked for by name
DIRECTOR, DIRECTOR_REVISION = "mlx-community/Qwen3-VL-32B-Instruct-4bit", "6e5644d3ea4b953b5221ffd02339bf897041038a"

class Chat:
    """One model process held for a conversation: `with Chat() as c: c.say(messages)`."""
    def __init__(self, vision=False, director=False):
        """vision: the same weights loaded with their picture tower (mlx-vlm), so a message's content may be a list of
        {"type": "text", "text"} and {"type": "image", "path"} parts. Text answers are the same; `keep` is not used."""
        t = time.time()
        self.p = subprocess.Popen([where.PY_IMG, "-B", os.path.abspath(__file__)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={**os.environ, "CHAT_VISION": "1" if vision or director else "0", "CHAT_DIRECTOR": "1" if director else "0"})
        line = self.p.stdout.readline()
        if not line: raise SystemExit("The world-plan model could not be started. Open the app again to finish setup, or run ./setup.sh.")
        self.load_s = json.loads(line)["load_s"]
    def say(self, messages, max_tokens=5200, temperature=0.1, tools=None, keep=False):
        self.p.stdin.write(json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "tools": tools, "keep": keep}) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line: raise SystemExit("The world-plan model stopped before it had answered.")
        return json.loads(line)
    def close(self):
        try: self.p.stdin.close(); self.p.wait(timeout=60)
        except Exception: self.p.kill()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

def tool_calls(text):
    """The calls in an answer, as the model's template has it write them:
    <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call> -> [(NAME, {KEY: VALUE})].
    A call cut short by the token limit (no closing tags) is still read, so its code can be judged and answered."""
    import re
    out = []
    for m in re.finditer(r"<function=([\w.-]+)>(.*?)(?:</function>|\Z)", text, re.S):
        args = {k: v.strip("\n") for k, v in re.findall(r"<parameter=([\w.-]+)>\n?(.*?)(?:</parameter>|\Z)", m.group(2), re.S)}
        out.append((m.group(1), args))
    return out

def common(a, b):
    """How many leading items two lists share."""
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def ask_many(prompts, max_tokens=3000, temperature=0.2):
    """Several questions, each on its own, with one load of the model. "{{GUIDE}}" in a prompt is replaced by the answer
    to the last prompt that stood without it. -> ([text, ...], {"load_s", "s": [...], "wall_s"})"""
    t = time.time(); out, secs, guide = [], [], ""
    with Chat() as c:
        for p in prompts:
            own = "{{GUIDE}}" not in p
            x = c.say([{"role": "user", "content": p.replace("{{GUIDE}}", guide)}], max_tokens, temperature); out.append(x["text"]); secs.append(x["s"])
            if own: guide = x["text"]
        load = c.load_s
    return out, {"load_s": load, "s": secs, "wall_s": round(time.time() - t, 1)}

def serve_vision():
    """The same service with eyes: Qwen3.5's own picture tower, through mlx-vlm."""
    import mlx.core as mx
    from mlx_vlm import load, generate
    t = time.time()
    name, rev = (DIRECTOR, DIRECTOR_REVISION) if os.environ.get("CHAT_DIRECTOR") == "1" else (MODEL, REVISION)
    try: model, proc = load(name, revision=rev)
    except Exception as e: sys.exit(f"The world-plan model could not be loaded with its picture tower ({type(e).__name__}: {str(e)[:200]}). Open the app again to finish setup, or run ./setup.sh.")
    print(json.dumps({"load_s": round(time.time() - t, 1)}), flush=True)
    for line in sys.stdin:
        if not line.strip(): continue
        q = json.loads(line); images = []; msgs = []
        for m in q["messages"]:
            if isinstance(m["content"], list):
                images += [p["path"] for p in m["content"] if p.get("type") == "image"]
                msgs.append({**m, "content": [{"type": "image"} if p.get("type") == "image" else {"type": "text", "text": p.get("text", "")} for p in m["content"]]})
            else: msgs.append(m)
        text = proc.apply_chat_template(msgs, tools=q.get("tools") or None, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t = time.time(); n = q.get("max_tokens", 5000)
        r = generate(model, proc, text, image=images or None, max_tokens=n, temperature=q.get("temperature", 0.1), verbose=False)
        got = int(getattr(r, "generation_tokens", 0) or 0)
        print(json.dumps({"text": r.text, "s": round(time.time() - t, 1), "gen_tokens": got, "peak_gb": round(mx.get_peak_memory() / 2**30, 2), "finish": "length" if got >= n else "stop", "prompt_tokens": int(getattr(r, "prompt_tokens", 0) or 0), "reused": 0}), flush=True)

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"                  # the weights are on disk already; never call out
    if os.environ.get("CHAT_VISION") == "1": serve_vision(); sys.exit(0)
    import mlx.core as mx
    mx.set_cache_limit(2 * 2**30)
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    t = time.time()
    try: model, tok = load(MODEL, revision=REVISION)
    except Exception as e: sys.exit(f"The world-plan model could not be loaded ({type(e).__name__}: {str(e)[:200]}). If it is not downloaded, open the app again to finish setup, or run ./setup.sh.")
    print(json.dumps({"load_s": round(time.time() - t, 1)}), flush=True)
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
    cache, seen = None, []                              # for "keep": the model's state, and the tokens it stands for
    for line in sys.stdin:
        if not line.strip(): continue
        q = json.loads(line)
        text = tok.apply_chat_template(q["messages"], tools=q.get("tools") or None, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t = time.time(); parts, last, prompt, extra, reused = [], None, text, {}, 0
        if q.get("keep"):
            ids = tok.encode(text, add_special_tokens=False); n = min(common(seen, ids), len(ids) - 1)       # at least one token must be new
            if cache is None or (n < len(seen) and not can_trim_prompt_cache(cache)): cache, n = make_prompt_cache(model), 0
            elif n < len(seen): trim_prompt_cache(cache, len(seen) - n)
            prompt, extra, seen = ids[n:], {"prompt_cache": cache}, list(ids); reused = n
        for r in stream_generate(model, tok, prompt, max_tokens=q.get("max_tokens", 5000), sampler=make_sampler(temp=q.get("temperature", 0.1)), **extra):
            parts.append(r.text); last = r
            if q.get("keep") and r.token is not None: seen.append(int(r.token))
        if q.get("keep"):                               # hold `seen` to exactly what the state covers; if that cannot be told, start again next turn
            have = getattr(cache[0], "offset", None)
            if isinstance(have, int) and have <= len(seen): seen = seen[:have]
            else: cache, seen = None, []
        print(json.dumps({"text": "".join(parts), "s": round(time.time() - t, 1), "gen_tokens": last.generation_tokens, "peak_gb": round(last.peak_memory, 2), "finish": last.finish_reason, "prompt_tokens": last.prompt_tokens, "reused": reused}), flush=True)
